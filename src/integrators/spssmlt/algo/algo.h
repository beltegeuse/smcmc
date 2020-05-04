#ifndef MITSUBA_ALGO_H
#define MITSUBA_ALGO_H

#include <mutex>

#include "../spssmlt.h"
#include "../rendering.h"

MTS_NAMESPACE_BEGIN

// These are the different MH step that we can perform
enum EOperationType {
    /// Large step mutation
    ELargeOp,
    /// Local mutation
    ELocalOp,
    /// Replica exchange
    EREOp
};

/**
 * This class make easier to perform MH step for the different algorithm variants
 * Note that the underlying algorithm are implemented inside algo_classic.h and algo_re.h
 */
class Rendering {
public:
    Rendering(SPSSMLTConfiguration &config_,
              Scene *scene, const Vector2i &cropSize_) : config(config_) {
      ref<Scheduler> scheduler = Scheduler::getInstance();
      nCores = scheduler->getCoreCount();
      cropSize = cropSize_;

      // Number of path sampler
      pathSamplers.reserve(nCores);
      for (int idCore = 0; idCore < nCores; idCore++) {
        pathSamplers.push_back(new PathTracingTilde(scene, config, cropSize));
      }
    }

    /// Output meaning full informations
    virtual void dumpInfo(Scene *scene, Film *film, int iteration) {}

    /// Generate the list of tile
    void initTiles(std::vector<AbstractTile *> &tildes) {
        ref<SPSSMLTSampler> mltSampler = new SPSSMLTSampler(config);
        if (config.randomSeed != -1) {
            mltSampler->setRandom(new Random(config.randomSeed));
        }

        // Create tildes array by creating a new sampler as well
        tildes.reserve(cropSize.x * cropSize.y);//Number of tiles should never exceed this
        for (int y = 0; y < cropSize.y; y += 1) {
            for (int x = 0; x < cropSize.x; x += 1) {
                ref<Sampler> clonedSampler = mltSampler->clone();
                clonedSampler->incRef(); // To be sure that the sampler is not destroyed
                auto tile = new AbstractTile(Point2i(x, y),
                                             dynamic_cast<SPSSMLTSampler *>(clonedSampler.get()));
                tile->sampler->HachisukaMut = config.amcmc;
                tile->sampler->aStar = config.aStar;
                tildes.push_back(tile);
            }
        }
    }

    ReconstructionOptions reconstruct(std::vector<AbstractTile *> &tildes, size_t sampleCount) {
        // Set the initial scale to 1 / nbSamples
        // before each alignment procedure
        for (AbstractTile *tile : tildes) {
            tile->resetScale();
            tile->scaleNbSamples();
        }

        std::unique_ptr<TileSolver> solver = std::unique_ptr<TileSolver>([&]() -> TileSolver * {
            if (config.alignAlgo == ENoAlign) {
                // We do nothing with this solver
                return nullptr;
            } else if (config.alignAlgo == EIterativeReweight) {
                return new TileIterativeReweighted(config, cropSize, tildes);
            } else if (config.alignAlgo == ECovariateLog) {
                return new TileCovariateGradLog(config, cropSize, tildes);
            } else if(config.alignAlgo == ECovariateLogVal) {
                return new TileCovariateGradLogVal(config, cropSize, tildes);
            } else if (config.alignAlgo == EReference) {
                return new TileCovariateRef(config, cropSize, tildes);
            } else {
                SLog(EError, "Do not found the correct solver");
                return nullptr;
            }
        }());

        if (solver == nullptr) {
            return ReconstructionOptions{.needGlobalRescale = false, .res = nullptr};
        } else {
            return solver->solve();
        }
    }

    std::vector<std::vector<AbstractTile *>> generateTasks(int iteration, std::vector<AbstractTile *> &tildes) {
        // Most basic way to generate the tasks
        std::vector<std::vector<AbstractTile *>> estimatedTasks;
        estimatedTasks.reserve(tildes.size());
        for (AbstractTile *t : tildes) {
            estimatedTasks.emplace_back(1, t);
        }
        return estimatedTasks;
    }

    /// Do the rendering for the set of tiles
    virtual void compute(std::vector<AbstractTile *> &tildes,
                         std::vector<std::vector<AbstractTile *>> &&tasks,
                         size_t sampleCount) = 0;

    /**
     * Initialize the different tiles chains
     * Note that this function could be called at each iteration.
     * However, if SPPInit is 0, the method does not perform any further initialization
     */
    void initializeChains(std::vector<AbstractTile *> &tildes, int iteration) {
      if (config.SPPInit == 0 && iteration == 0) {
        SLog(EWarn, "No initialization for the MC");
        return;
      }
      if (config.nbMCMC == 0 && iteration != 0) {
        SLog(EInfo, "No MCMC global, just skip");
        return;
      }

      if (config.initAlgo == EInitNaive) {
        SLog(EInfo, "Initialize all chains by brute forcing...");
        // Initialize all chains with uniform random generation...
        BlockScheduler initializeChains(tildes.size(), nCores, 32);
        BlockScheduler::ComputeBlockFunction initfunc = [&](int tileID, int threadID) {
            AbstractTile *tile = tildes[tileID];
            SPSSMLTSampler *sampler = tile->sampler;
            Random *random = sampler->getRandom();

            // Generate the tile value and the importance associated to it
            SAssert(tile->size() > 0); // Impossible to initialize if no pixel inside the tile
            sampler->setLargeStep(true);

            // Retry until we found some thing ...
            size_t nbTry = 0;
            while (nbTry < config.SPPInit) {
              // Generate the proposed path
              auto id_pix = this->pathSamplers[threadID]->Lpix(tile->proposed, sampler, tile->pixel(0));
              tile->impProp = importance(tile->proposed, id_pix.index);
              if (config.useInitialization) {
                tile->accumNorm(tile->impProp, tile->proposed);
              }

              // Randomly accept this proposed path
              // based on the current one
              if (tile->impProp != 0.0) {
                bool acc = [&]() -> bool {
                    if (tile->impCurr == 0)
                      return true;
                    else
                      return std::min(1.0, tile->impProp / tile->impCurr) > random->nextFloat();
                }();

                if (acc) {
                  sampler->accept();
                  tile->impCurr = tile->impProp;
                  tile->current = tile->proposed;
                  tile->OriInit = true;
                } else {
                  sampler->reject();
                }
              } else {
                sampler->reject();
              }
              nbTry += 1;
            }
        };
        initializeChains.run(initfunc);
      } else if (config.initAlgo == EInitBruteForce) {
        SLog(EInfo, "Initialize all chains by brute forcing...");
        // Initialize all chains
        BlockScheduler initializeChains(tildes.size(), nCores, 32);
        BlockScheduler::ComputeBlockFunction initfunc = [&](int tileID, int threadID) {
            AbstractTile *tile = tildes[tileID];
            SPSSMLTSampler *sampler = tile->sampler;

            // Generate the tile value and the importance associated to it
            SAssert(tile->size() > 0); // Impossible to initialize if no pixel inside the tile
            sampler->setLargeStep(true);

            // Retry until we found some thing ...
            size_t nbTry = 0;
            while (tile->impCurr == 0 && nbTry < config.SPPInit) {
              auto id_pix = this->pathSamplers[threadID]->Lpix(tile->current, sampler, tile->pixel(0));
              tile->impCurr = importance(tile->current, id_pix.index);
              if (config.useInitialization) {
                tile->accumNorm(tile->impCurr, tile->current);
              }
              sampler->accept();
              nbTry += 1;
            }
        };
        initializeChains.run(initfunc);
      } else if (config.initAlgo == EInitMCMC) {
        SAssert(tildes[0]->size() == 5);

        struct MCMCInitState {
            Point2i pixel = Point2i(0);
            Float imp = 0.f;
            SPSSMLTSampler *sampler = nullptr;
            std::vector<Spectrum> values = std::vector<Spectrum>(5, Spectrum(0.f));
        };

        SLog(EInfo, "Initialize all chains by MCMC");
        // Create SPSSMLT samplers needed for the parallisation layer
        auto mcmc_states = [&]() -> std::vector<MCMCInitState> {
            std::vector<MCMCInitState> mcmc_states(nCores);
            for (size_t i = 0; i < nCores; i++) {
              // This code support random seed as the sampler of each tile are random
              ref<Sampler> clonedSampler = tildes[0]->sampler->clone();
              clonedSampler->incRef();
              mcmc_states[i].sampler = static_cast<SPSSMLTSampler *>(clonedSampler.get());
            }
            return mcmc_states;
        }();

        // Initialize all the samplers with a valid state
        // Note that a proper resampling step might improve this initialization procedure
        // For now, a burning procedure is used here.
        SLog(EInfo, " - Initialize the samplers...");
        for (int stateID = 0; stateID < nCores; stateID++) {
          MCMCInitState &state = mcmc_states[stateID];
          SPSSMLTSampler *sampler = state.sampler;

          sampler->setLargeStep(true);
          int nbTry = 0;
          while (state.imp == 0) {
            if (nbTry != 0) {
              sampler->accept();
            }
            auto res_pix = pathSamplers[stateID]->Lpix(state.values,
                                                       sampler);
            state.pixel = res_pix.coord;
            state.imp = importance(state.values, res_pix.index);
            nbTry++;
          }
          sampler->accept();
        }

        // Here we use a burn in period (10k samples)
        // Note that we could also used resampling approach (Veach startup bias elimination)
        {
            const auto nbSamplesBurnin = 100000;
            BlockScheduler initializeChains(nCores, nCores, 1);
            BlockScheduler::ComputeBlockFunction initfunc = [&](int tileID, int threadID) {
                MCMCInitState &state = mcmc_states[threadID];
                Random *random = state.sampler->getRandom();
                auto pathSampler = pathSamplers[threadID].get();

                std::vector<Spectrum> proposed(5, Spectrum(0.f));
                for (int mutationCtr = 0; mutationCtr < nbSamplesBurnin; mutationCtr++) {
                    bool largeStep = random->nextFloat() < config.pLarge;
                    state.sampler->setLargeStep(largeStep);

                    auto res_pix = pathSampler->Lpix(proposed, state.sampler);
                    Float impProp = importance(proposed, res_pix.index);

                    if (largeStep && config.useInitialization) {
                        tildes[res_pix.coord.y * cropSize.x + res_pix.coord.x]->accumNorm(impProp, proposed);
                    }

                    // MCMC rules
                    Float a = std::min((Float) 1.0f, impProp / state.imp);
                    if (a > random->nextFloat()) {
                        state.imp = impProp;
                        state.pixel = res_pix.coord;
                        state.values = proposed;
                        state.sampler->accept();
                    } else {
                        state.sampler->reject();
                    }

                    mutationCtr++;
                }
            };
            initializeChains.run(initfunc);
            // A still open research question is: Does our SMCMC approach performs
            // better with more independent global MCMC chains?
        }

        // Run in parallel the different chains for a certain amount of time (4 SPP for example).
        {
          static std::mutex io_mutex;
          SLog(EInfo, " - MCMC walking...");
          size_t sampleCount = cropSize.x * cropSize.y / nCores;
          sampleCount *= config.SPPInit;

          size_t nb_init_expected = config.percentageInit * cropSize.x * cropSize.y;
          size_t nb_init = 0;

          if (iteration != 0) {
            nb_init_expected = 0;
            sampleCount = cropSize.x * cropSize.y / nCores;
            sampleCount *= config.nbMCMC;
          }

          BlockScheduler initializeChains(nCores, nCores, 1);
          BlockScheduler::ComputeBlockFunction initfunc = [&](int tileID, int threadID) {
              MCMCInitState &state = mcmc_states[threadID];
              Random *random = state.sampler->getRandom();
              auto pathSampler = pathSamplers[threadID].get();

              auto change = [&tildes, this, &nb_init](MCMCInitState &state) -> void {
                  AbstractTile *tile = tildes[state.pixel.y * cropSize.x + state.pixel.x];
                  if (tile->impCurr == 0) {
                    std::lock_guard<std::mutex> lk(io_mutex);
                    nb_init += 1;

                    // Swap the tile initial state
                    tile->impCurr = state.imp;
                    tile->sampler->copy_from(state.sampler);
                    tile->current = state.values;
                    tile->OriInit = true;
                  } else {
                    // As the two TF are the same RE is always accepted
                    // TODO: Can be super slow (due to the mutex usage). Need to found another implementation
                    //  However, it is still not critical as initialization takes only a fraction of the rendering
                    //  time.
                    std::lock_guard<std::mutex> lk(io_mutex);
                    std::swap(state.sampler, tile->sampler);
                    std::swap(state.sampler->amcmc, tile->sampler->amcmc);
                    std::swap(state.imp, tile->impCurr);
                    state.values.swap(tile->current);
                  }
              };

              std::vector<Spectrum> proposed(5, Spectrum(0.f));
              for (int mutationCtr = 0;
                   mutationCtr < sampleCount && (nb_init_expected == 0 || nb_init < nb_init_expected);) {
                bool largeStep = random->nextFloat() < config.pLarge;
                state.sampler->setLargeStep(largeStep);

                auto res_pix = pathSampler->Lpix(proposed, state.sampler);
                Float impProp = importance(proposed, res_pix.index);

                if (largeStep && config.useInitialization) {
                  tildes[res_pix.coord.y * cropSize.x + res_pix.coord.x]->accumNorm(impProp, proposed);
                }

                // MCMC rules
                Float a = std::min((Float) 1.0f, impProp / state.imp);
                if (a > random->nextFloat()) {
                  state.imp = impProp;
                  state.pixel = res_pix.coord;
                  state.values = proposed;
                  state.sampler->accept();
                } else {
                  // FIXME: Assest if this idea is good or not.
                  //    For example. do we still want to perform RE here?
                  state.sampler->reject();
                }

                change(state);
                mutationCtr++;
              }
          };
          initializeChains.run(initfunc);

          SLog(EInfo, "Nb pixels initialized: %i", nb_init);
          SLog(EInfo, "Pourcentage: %f (excepted: %f)", nb_init / (Float) (cropSize.x * cropSize.y),
               config.percentageInit);

        }
      } else {
        SLog(EError, "Initialization is wrong");
      }
    }

protected:
    void output(Scene *scene, Film *film, int currentIteration, const std::string &suffix,
                std::function<Spectrum(size_t)> &f) {
      auto cropSize = film->getCropSize();
      ref<Bitmap> bitmap = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat, cropSize);
      auto bitmap_ptr = (Spectrum *) bitmap->getData();
      for (size_t t = 0; t < cropSize.x * cropSize.y; t++) {
        bitmap_ptr[t] = f(t);
      }

      develop(scene, film, bitmap, currentIteration, suffix);
    }

    void develop(Scene *scene, Film *film, const Bitmap *bitmap,
                 int currentIteration, const std::string &suffixName = "_") {
      std::stringstream ss;
      ss << scene->getDestinationFile().string() << suffixName
         << currentIteration;
      std::string path = ss.str();
      film->setBitmap(bitmap);
      film->setDestinationFile(path, 0);
      film->develop(scene, 0.f);
      // in the last rendering pass, post process is called on scene which can call film develop once again
      // to prevent the curent file being overwritten with init bitmap (being displayed), we set the
      // file name to empty here
      film->setDestinationFile("", 0);
    }

protected:

    // The different importance function tested.
    inline Float importance(const std::vector<Spectrum> &res, size_t index) const {
      if (config.imp == ELuminance) {
        auto lum = 0.0;
        for (size_t t = 0; t < res.size(); t++) { lum += res[t].getLuminance(); }
        return lum;
      } else if (config.imp == EMax) {
        auto max_rgb = 0.0;
        for (size_t t = 0; t < res.size(); t++) {
          max_rgb += res[t].max();
        }
        return max_rgb;
      } else if (config.imp == EMaxAll) {
        // This is the target function that we used in practice
        // as it reduce the color shift bias in earlier iterations.
        auto max_rgb = 0.0;
        for (size_t t = 0; t < res.size(); t++) {
          max_rgb = std::max(max_rgb, res[t].max());
        }
        return max_rgb;
      } else if (config.imp == ESum) {
        auto sum = 0.0;
        for (size_t t = 0; t < res.size(); t++) {
          sum += res[t][0] + res[t][1] + res[t][2];
        }
        return sum;
      } else {
        SLog(EError, "Impossible to compute the target function");
        return 0.0;
      }
    }

protected:
    /// Helpers functions to perfom different MH opertations
    void independentChainExploration(std::vector<AbstractTile *> &tiles,
                                     size_t nbSamples,
                                     RenderingTechniqueTilde *pathSampler,
                                     Random *random) {
      for (AbstractTile *tile : tiles) {
        for (int mutationCtr = 0; mutationCtr < nbSamples; ++mutationCtr) {
          if (tile->impCurr != 0.0) {
            tile->newSample();
            mutate(tile, random, pathSampler);
            Float a = tile->impCurr == 0.0 ? 1.0 : std::min((Float) 1.0f, tile->impProp / tile->impCurr);
            bool acc = classicalMCMC(*tile, a, random->nextFloat());
            if (!tile->sampler->isLargeStep()) {
              tile->nbSmallMut++;
              if (acc) {
                tile->nbSmallMutAcc++;
              }
            }
          } else {
            if(config.useUniformInit) {
              mutate(tile, random, pathSampler);
              Float a = tile->impCurr == 0.0 ? 1.0 : std::min((Float) 1.0f, tile->impProp / tile->impCurr);
              bool acc = classicalMCMC(*tile, a, random->nextFloat());
            } else {
              // Do nothing...
              // Only the MC estimator at the same rate
              if(random->nextFloat() < config.pLarge) {
                mutate(tile, random, pathSampler);
                tile->reject(0.0);
              }
            }

          }
        }
        tile->flush();
      }
    }

    /**
     *
     * @param tile
     * @param random
     * @param pathSampler
     */
    void mutate(AbstractTile *tile, Random *random, RenderingTechniqueTilde *pathSampler) {
      EOperationType typeOp = random->nextFloat() < config.pLarge ? ELargeOp : ELocalOp;
      if (tile->impCurr == 0.0) {
          // In case we still do not initialiazed properly the chain
          typeOp = ELargeOp;
      }
      tile->sampler->setLargeStep(typeOp == ELargeOp);

      auto res_pix = pathSampler->Lpix(tile->proposed, tile->sampler, tile->pixel(0));
      tile->impProp = importance(tile->proposed, res_pix.index);

      if (typeOp == ELargeOp) {
        tile->accumNorm(tile->impProp, tile->proposed);
      }

      // Just a check in case something bad happen
      if (std::isnan(tile->impProp) || tile->impProp < 0) {
        SLog(EError, "Encountered a sample with luminance = %f, ignoring!", tile->impProp);
      }
    }

    bool classicalMCMC(AbstractTile &tile, Float a, Float rVal) {
      // Compute the acceptance or not
      // Using veach weights
      bool accept;
      Float currentWeight, proposedWeight;
      if (a > 0) {
        currentWeight = 1 - a;
        proposedWeight = a;
        accept = (a == 1) || (rVal < a);
      } else {
        currentWeight = 1;
        proposedWeight = 0;
        accept = false;
      }

      tile.cumulativeWeight += currentWeight;
      if (accept) {
        tile.accept(proposedWeight);
      } else {
        tile.reject(proposedWeight);
      }
      return accept;
    }

    void RE(AbstractTile &s1, AbstractTile &s2,
            Random *random, RenderingTechniqueTilde *pathSampler, bool horizontal, bool even) {
      if (s1.impCurr != 0.0 && s2.impCurr != 0.0) {
        s1.newSample();
        s2.newSample();
        // Prepare sampler to replay seq
        s1.sampler->replayPrevious();
        s2.sampler->replayPrevious();
        s1.sampler->setLargeStep(false);
        s2.sampler->setLargeStep(false);

        // Generate path by swapping
        auto s1_id_pix = pathSampler->Lpix(s1.proposed, s2.sampler, s1.pixel(0));
        s1.impProp = importance(s1.proposed, s1_id_pix.index);
        auto s2_id_pix = pathSampler->Lpix(s2.proposed, s1.sampler, s2.pixel(0));
        s2.impProp = importance(s2.proposed, s2_id_pix.index);

        Float r = std::min((Float) 1.0f, (s1.impProp * s2.impProp) / (s1.impCurr * s2.impCurr));
        Float rVal = random->nextFloat();

        bool acc = classicalMCMC(s1, r, rVal);
        classicalMCMC(s2, r, rVal);

        s1.REAttempt += 1;
        s2.REAttempt += 1;

        if (acc) {
          // Swap the random number
          std::swap(s1.sampler, s2.sampler);
          std::swap(s1.sampler->amcmc, s2.sampler->amcmc);
          s1.REAcc += 1;
          s2.REAcc += 1;
        } else {
          // Remake the sampler in the normal position
          // If the move was accepted, the time is already
          // incremented back
          s1.sampler->fixTime();
          s2.sampler->fixTime();
        }
        s1.sampler->endReplay();
        s2.sampler->endReplay();
      } else if (s1.impCurr != 0.0 || s2.impCurr != 0.0) {
        AbstractTile *tInit = nullptr;
        AbstractTile *tNonInit = nullptr;
        if (s1.impCurr == 0.0) {
          tInit = &s2;
          tNonInit = &s1;
        } else {
          tInit = &s1;
          tNonInit = &s2;
        }

        // Steal the sampler
        tNonInit->sampler->copy_from(tInit->sampler);
        tNonInit->sampler->replayPrevious();
        tNonInit->sampler->setLargeStep(false);
        auto non_init_id_pix = pathSampler->Lpix(tNonInit->proposed, tNonInit->sampler, tNonInit->pixel(0));
        tNonInit->impProp = importance(tNonInit->proposed, non_init_id_pix.index);
        if (tNonInit->impProp != 0.0) {
          classicalMCMC(*tNonInit, 1.0, random->nextFloat());
          tNonInit->cumulativeWeight = 0.0; // Reinit the weight
          tNonInit->REInit = true;
          tNonInit->sampler->amcmc = tInit->sampler->amcmc;
        } else {
          // Make the time coherent with replay
          // Note that if the sampler was accepted
          // The time already get advanced.
          tNonInit->sampler->fixTime();
        }
        tNonInit->sampler->endReplay();

        // Do the sampling of tNonInit otherwise
        tInit->newSample();
        mutate(tInit, random, pathSampler);
        Float a = tInit->impCurr == 0.0 ? 1.0 : std::min((Float) 1.0f, tInit->impProp / tInit->impCurr);
        bool acc = classicalMCMC(*tInit, a, random->nextFloat());
        if (!tInit->sampler->isLargeStep()) {
          tInit->nbSmallMut++;
          if (acc) {
            tInit->nbSmallMutAcc++;
          }
        }
      } else {
        // The two are not init
        // Only do the init step
        if(config.useUniformInit) {
          {
            mutate(&s1, random, pathSampler);
            Float a = s1.impCurr == 0.0 ? 1.0 : std::min((Float) 1.0f, s1.impProp / s1.impCurr);
            bool acc = classicalMCMC(s1, a, random->nextFloat());
          }

          {
            mutate(&s2, random, pathSampler);
            Float a = s2.impCurr == 0.0 ? 1.0 : std::min((Float) 1.0f, s2.impProp / s2.impCurr);
            bool acc = classicalMCMC(s2, a, random->nextFloat());
          }
        }
      }
    }

protected:
    SPSSMLTConfiguration &config;
    size_t nCores;
    ref_vector <RenderingTechniqueTilde> pathSamplers;
    Vector2i cropSize;
};

MTS_NAMESPACE_END

#endif //MITSUBA_ALGO_H
