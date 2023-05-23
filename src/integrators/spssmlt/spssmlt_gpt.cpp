/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob and others.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <omp.h>

#include <mitsuba/bidir/util.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/core/statistics.h>
#include <fstream>  

#include "spssmlt.h"
#include "tile.h"
#include "spssmlt_sampler.h"

#include "rendering.h"
#include "../reconstruction.h"

MTS_NAMESPACE_BEGIN

StatsCounter acceptanceRate("Primary sample space MLT",
                            "Overall acceptance rate", EPercentage);

/*!\plugin{pssmlt}{Primary Sample Space Metropolis Light Transport}
 * \order{9}
 * \parameters{
 *	   \parameter{bidirectional}{\Boolean}{
 *	   PSSMLT works in conjunction with another rendering
 *	   technique that is endowed with Markov Chain-based sample generation.
 *	   Two choices are available (Default: \code{true}):
 *	    \begin{itemize}
 *	    \item \code{true}: Operate on top of a fully-fleged bidirectional
 *	      path tracer with multiple importance sampling.
 *	    \item \code{false}: Rely on a unidirectional
 *	    volumetric path tracer (i.e. \pluginref{volpath})
 *	    \vspace{-4mm}
 *	    \end{itemize}
 *	   }
 *     \parameter{maxDepth}{\Integer}{Specifies the longest path depth
 *         in the generated output image (where \code{-1} corresponds to $\infty$).
 *	       A value of \code{1} will only render directly visible light sources.
 *	       \code{2} will lead to single-bounce (direct-only) illumination,
 *	       and so on. \default{\code{-1}}
 *	   }
 *	   \parameter{directSamples}{\Integer}{
 *	       By default, this plugin renders the direct illumination component
 *	       separately using an optimized direct illumination sampling strategy
 *	       that uses low-discrepancy number sequences for superior performance
 *	       (in other words, it is \emph{not} rendered by PSSMLT). This
 *	       parameter specifies the number of samples allocated to that method. To
 *	       force PSSMLT to be responsible for the direct illumination
 *	       component as well, set this parameter to \code{-1}. \default{16}
 *	   }
 *	   \parameter{rrDepth}{\Integer}{Specifies the minimum path depth, after
 *	      which the implementation will start to use the ``russian roulette''
 *	      path termination criterion. \default{\code{5}}
 *	   }
 *	   \parameter{luminanceSamples}{\Integer}{
 *	      MLT-type algorithms create output images that are only
 *	      \emph{relative}. The algorithm can e.g. determine that a certain pixel
 *	      is approximately twice as bright as another one, but the absolute
 *	      scale is unknown. To recover it, this plugin computes
 *	      the average luminance arriving at the sensor by generating a
 *	      number of samples. \default{\code{100000} samples}
 *     }
 *     \parameter{twoStage}{\Boolean}{Use two-stage MLT?
 *       See below for details. \default{{\footnotesize\code{false}}}}
 *	   \parameter{pLarge}{\Float}{
 *	     Rate at which the implementation tries to replace the current path
 *	     with a completely new one. Usually, there is little need to change
 *	     this. \default{0.3}
 *	   }
 * }
 * Primary Sample Space Metropolis Light Transport (PSSMLT) is a rendering
 * technique developed by Kelemen et al. \cite{Kelemen2002Simple} which is
 * based on Markov Chain Monte Carlo (MCMC) integration.
 * \renderings{
 *    \vspace{-2mm}
 *    \includegraphics[width=11cm]{images/integrator_pssmlt_sketch.pdf}\hfill\,
 *    \vspace{-3mm}
 *    \caption{PSSMLT piggybacks on a rendering method that can turn points
 *    in the primary sample space (i.e. ``random numbers'') into paths. By
 *    performing small jumps in primary sample space, it can explore the neighborhood
 *    of a path\vspace{-5mm}}
 * }
 * In contrast to simple methods like path tracing that render
 * images by performing a na\"ive and memoryless random search for light paths,
 * PSSMLT actively searches for \emph{relevant} light paths (as is the case
 * for other MCMC methods). Once such a path is found, the algorithm tries to
 * explore neighboring paths to amortize the cost of the search. This can
 * significantly improve the convergence rate of difficult input.
 * Scenes that were already relatively easy to render usually don't benefit
 * much from PSSMLT, since the MCMC data management causes additional
 * computational overheads.
 *
 * An interesting aspect of PSSMLT is that it performs this exploration
 * of light paths by perturbing the ``random numbers'' that were initially
 * used to construct the path. Subsequent regeneration of the path using the
 * perturbed numbers yields a new path in a slightly different configuration, and
 * this process repeats over and over again.
 * The path regeneration step is fairly general and this is what makes
 * the method powerful: in particular, it is possible to use PSSMLT as a
 * layer on top of an existing method to create a new ``metropolized''
 * version of the rendering algorithm that is enhanced with a certain
 * degree of adaptiveness as described earlier.
 *
 * The PSSMLT implementation in Mitsuba can operate on top of either a simple
 * unidirectional volumetric path tracer or a fully-fledged bidirectional path
 * tracer with  multiple importance sampling, and this choice is controlled by the
 * \code{bidirectional} flag. The unidirectional path tracer is generally
 * much faster, but it produces lower-quality samples. Depending on the input, either may be preferable.
 * \vspace{-7mm}
 * \paragraph{Caveats:}
 * There are a few general caveats about MLT-type algorithms that are good
 * to know. The first one is that they only render ``relative'' output images,
 * meaning that there is a missing scale factor that must be applied to
 * obtain proper scene radiance values. The implementation in Mitsuba relies
 * on an additional Monte Carlo estimator to recover this scale factor. By
 * default, it uses 100K samples (controlled by the \code{luminanceSamples}
 * parameter), which should be adequate for most applications.
 *
 * The second caveat is that the amount of computational expense
 * associated with a pixel in the output image is roughly proportional to
 * its intensity. This means that when a bright object (e.g. the sun) is
 * visible in a rendering, most resources are committed to rendering the
 * sun disk at the cost of increased variance everywhere else. Since this is
 * usually not desired, the \code{twoStage} parameter can be used to
 * enable \emph{Two-stage MLT} in this case.
 *
 * In this mode of operation, the renderer first creates a low-resolution
 * version of the output image to determine the approximate distribution of
 * luminance values. The second stage then performs the actual rendering, while
 * using the previously collected information to ensure that
 * the amount of time spent rendering each pixel is uniform.
 *
 * The third caveat is that, while PSMLT can work with scenes that are extremely
 * difficult for other methods to handle, it is not particularly efficient
 * when rendering simple things such as direct illumination (which is more easily
 * handled by a brute-force type algorithm). By default, the
 * implementation in Mitsuba therefore delegates this to such a method
 * (with the desired quality being controlled by the \code{directSamples} parameter).
 * In very rare cases when direct illumination paths are very difficult to find,
 * it is preferable to disable this separation so that PSSMLT is responsible
 * for everything. This can be accomplished by setting
 * \code{directSamples=-1}.
 */

struct StatePSSMLTSimple {
    ref<SPSSMLTSampler> sampler = nullptr;

    // The current state
    Float cumulativeWeight = 0.f; // accumulate
    std::vector<Spectrum> current;
    Float impCurr = 0.f;
    Point2i pixelCurr;

    // The proposed state
    std::vector<Spectrum> proposed;
    Float impProp = 0.f;
    Point2i pixelProp;

    // Rendering informations
    ref<RenderingTechniqueTilde> pathSampler;
    ref<Sampler> indepSampler;

    // Buffer where we accumulate the samples
    ref<Bitmap> throughput;
    ref<Bitmap> dx;
    ref<Bitmap> dy;

    Vector2i cropSize;

    StatePSSMLTSimple() {}

    ~StatePSSMLTSimple() {
      if (cumulativeWeight != 0.f) {
        SLog(EError, "0 cumulative weights");
      }
    }

    void clear() {
      throughput->clear();
      dx->clear();
      dy->clear();
    }

    void accumulate(const Point2i &pixel,
                    const Float weight,
                    const std::vector<Spectrum> &values,
                    int threadID) {
      SAssert(std::isfinite(weight) && weight > 0);
      Spectrum *throughputPix = (Spectrum *) throughput->getData();
      Spectrum *dxPix = (Spectrum *) dx->getData();
      Spectrum *dyPix = (Spectrum *) dy->getData();

      // Assuming the cross order for our buffers
      size_t curr_pix = pixel.y * cropSize.x + pixel.x;
      throughputPix[curr_pix] += 0.2 * weight * values[ECurr];

      // Consolidate the throughput with shifted values
      if (pixel.x < cropSize.x - 1) {
        throughputPix[curr_pix + 1] += 0.2 * weight * values[EXP]; // X+
      }
      if (pixel.y < cropSize.y - 1) {
        throughputPix[curr_pix + cropSize.x] += 0.2 * weight * values[EYP]; // Y+
      }
      if (pixel.x > 0) {
        throughputPix[curr_pix - 1] += 0.2 * weight * values[EXN]; // X-
      }
      if (pixel.y > 0) {
        throughputPix[curr_pix - cropSize.x] += 0.2 * weight * values[EYN]; // Y-
      }

      // Updating the gradient buffers
      if (pixel.x < cropSize.x - 1) {
        dxPix[curr_pix] += 0.5 * weight * (values[EXP] - values[ECurr]);
      }
      if (pixel.y < cropSize.y - 1) {
        dyPix[curr_pix] += 0.5 * weight * (values[EYP] - values[ECurr]);
      }
      if (pixel.x > 0) {
        dxPix[curr_pix - 1] += 0.5 * weight * (values[ECurr] - values[EXN]);
      }
      if (pixel.y > 0) {
        dyPix[curr_pix - cropSize.x] += 0.5 * weight * (values[ECurr] - values[EYN]);
      }
    }

    // Accepting or rejecting a state
    void accept(Float proposedWeight, int threadID) {
      // Accumulate the results inside the tilde
      if (cumulativeWeight != 0) {
        accumulate(pixelCurr, cumulativeWeight / impCurr, current, threadID);
      }

      // Change the current state
      cumulativeWeight = proposedWeight;
      impCurr = impProp;
      pixelCurr = pixelProp;
      impProp = 0;

      current = proposed;
      proposed.clear();

      // Accept the move and compute some statistics
      sampler->accept();
    }

    void reject(Float proposedWeight, int threadID) {
      // Accumulate the results inside the tilde
      if (proposedWeight != 0) {
        accumulate(pixelProp, proposedWeight / impProp, proposed, threadID);
      }
      impProp = 0.f;
      proposed.clear();

      sampler->reject();
    }

    void flush(int threadID) {
      if (impCurr != 0.f) {
        accumulate(pixelCurr, cumulativeWeight / impCurr, current, threadID);
        cumulativeWeight = 0.f;
      } else {
        SAssert(impCurr != 0.0);
      }
    }

    void resetProposed() {
      impProp = 0.f;
      proposed = std::vector<Spectrum>(5, Spectrum(0.f));
    }

    void setLargeStep(bool v) {
      sampler->setLargeStep(v);
    }
};

class SPSSMLTGPT : public Integrator {
public:
    SPSSMLTGPT(const Properties &props) : Integrator(props) {
      m_config.parse(props);

      m_reconstructL1 = props.getBoolean("reconstructL1", true);
      m_reconstructL2 = props.getBoolean("reconstructL2", false);
      m_reconstructUni = props.getBoolean("reconstructUni", false);
    }

    /// Unserialize from a binary data stream
    SPSSMLTGPT(Stream *stream, InstanceManager *manager)
            : Integrator(stream, manager) {
      m_config = SPSSMLTConfiguration(stream);
      configure();
    }

    virtual ~SPSSMLTGPT() {}

    void serialize(Stream *stream, InstanceManager *manager) const {
      Integrator::serialize(stream, manager);
      m_config.serialize(stream);
    }

    bool preprocess(const Scene *scene, RenderQueue *queue,
                    const RenderJob *job, int sceneResID, int sensorResID,
                    int samplerResID) {
      Integrator::preprocess(scene, queue, job, sceneResID,
                             sensorResID, samplerResID);
      ref<const Sensor> sensor = scene->getSensor();
      if (scene->getSubsurfaceIntegrators().size() > 0)
        Log(EError, "Subsurface integrators are not supported by MLT!");
      if (sensor->getSampler()->getClass()->getName() != "IndependentSampler")
        Log(EError, "Metropolis light transport requires the independent sampler");
      return true;
    }

    void cancel() {
      ref<RenderJob> nested = m_nestedJob;
//      if (nested)
//        nested->cancel();
      Scheduler::getInstance()->cancel(m_process);
      m_running = false;
    }

    bool render(Scene *scene, RenderQueue *queue, const RenderJob *job,
                int sceneResID, int sensorResID, int samplerResID) {
      m_running = true;
      /************************************************************************
       * Data Access
       ************************************************************************/
      // Get all the data
      ref<Scheduler> scheduler = Scheduler::getInstance();
      ref<Sensor> sensor = scene->getSensor();
      ref<Sampler> sensorSampler = sensor->getSampler();
      Film *film = sensor->getFilm();
      size_t nCores = scheduler->getCoreCount();
      size_t sampleCount = sensorSampler->getSampleCount();

      //Update the sample count to reflect the tile size
      Thread::initializeOpenMP(nCores);

      // Creation of timer file
      std::string timeFilename = scene->getDestinationFile().string()
                                 + "_time.csv";
      std::ofstream timeFile(timeFilename.c_str());
      ref<Timer> renderingTimer = new Timer;

      // Information about the rendering size
      m_cropSize = film->getCropSize();
      Assert(m_cropSize.x > 0 && m_cropSize.y > 0);
      Log(EInfo, "Starting render job (%ix%i, "
              SIZE_T_FMT
              " %s, "
              SSE_STR
              ", approx. "
              SIZE_T_FMT
              " mutations/pixel) ..",
          m_cropSize.x, m_cropSize.y,
          nCores, nCores == 1 ? "core" : "cores", sampleCount);

      // Deduce the number of mutation and workunits
      size_t desiredMutationsPerWorkUnit = 200000;
      if (m_config.workUnits <= 0) {
        const size_t cropArea = (size_t) m_cropSize.x * m_cropSize.y;
        const size_t workUnits = ((desiredMutationsPerWorkUnit - 1) +
                                  (cropArea * sampleCount)) / desiredMutationsPerWorkUnit;
        Assert(workUnits <= (size_t) std::numeric_limits<int>::max());
        m_config.workUnits = (int) std::max(workUnits, (size_t) 1);
      }
      size_t luminanceSamples = m_config.luminanceSamples;
      if (luminanceSamples < (size_t) m_config.workUnits * 10) {
        luminanceSamples = (size_t) m_config.workUnits * 10;
        Log(EWarn, "Warning: increasing number of luminance samples to "
                SIZE_T_FMT,
            luminanceSamples);
      }
      m_config.dump();

      /************************************************************************
       * Precomputation for tile shape initialization
       ************************************************************************/

      // Create this sampler before to change the random number
      ref<ReplayableSampler> rplSampler = new ReplayableSampler();
      // Number of samples per chains
      sampleCount *= std::max(size_t((m_cropSize.x * m_cropSize.y) / nCores), size_t(1)); //  * 0.1

      /************************************************************************
       * Render Loop
       ************************************************************************/
      // Allocate state object
      // Each state object will store all thread depend data
      std::vector<StatePSSMLTSimple> states(nCores);
      {
        ref<Sampler> indepSampler = static_cast<Sampler *> (PluginManager::getInstance()->
                createObject(MTS_CLASS(Sampler), Properties("independent")));
        for (int idCore = 0; idCore < nCores; idCore++) {
          // Create buffer informations
          states[idCore].throughput = new Bitmap(Bitmap::ESpectrum,
                                                 Bitmap::EFloat,
                                                 m_cropSize);
          states[idCore].dx = new Bitmap(Bitmap::ESpectrum,
                                         Bitmap::EFloat,
                                         m_cropSize);
          states[idCore].dy = new Bitmap(Bitmap::ESpectrum,
                                         Bitmap::EFloat,
                                         m_cropSize);
          // Create other information usefull for the state
          states[idCore].pathSampler = new PathTracingTilde(scene, m_config, m_cropSize);
          states[idCore].indepSampler = indepSampler->clone();
          states[idCore].cropSize = m_cropSize;
        }
      }

      /***************
       * Precompute data
       */
      // Prepare the kernel map

      // Allocate global accumulate buffer
      ref<Bitmap> throughputBuffer = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat, m_cropSize);
      throughputBuffer->clear();
      ref<Bitmap> dxBuffer = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat, m_cropSize);
      dxBuffer->clear();
      ref<Bitmap> dyBuffer = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat, m_cropSize);
      dyBuffer->clear();


      int iteration = 1;
      while (m_running) {
        ///////////// Step 2
        // Compute the global normalization factor
        if(!m_config.refNormalization) {
            ref<PathSampler> pathSampler = new PathSampler(m_config.volume, PathSampler::EUnidirectional, scene,
                                                           rplSampler, rplSampler, rplSampler,
                                                           m_config.maxDepth, m_config.rrDepth,
                                                           false,
                                                           true,
                                                           false, // Light image (BDPT)
                                                           m_config.hideEmitter,
                                                           m_config.minDepth);

            // Compute normalization factor
            Float luminance = pathSampler->computeAverageLuminance(luminanceSamples);

            // Average the global normalization factor over the iterations
            m_config.luminance = (luminance + (iteration - 1) * m_config.luminance) / iteration;
            Log(EInfo, "Luminance global value: %f", m_config.luminance);
        }

        if (iteration == 1) {
          SLog(EInfo, "Initialization all the chains ....");
          // Create sampler for all chains
          if (iteration == 1) {
            ref<SPSSMLTSampler> mltSampler = new SPSSMLTSampler(m_config);
            for (int i = 0; i < nCores; i++) {
              ref<Sampler> clonedSampler = mltSampler->clone();
              states[i].sampler = static_cast<SPSSMLTSampler *>(clonedSampler.get());
            }
          }

          // Initialize the chain ...
          for (int stateID = 0; stateID < nCores; stateID++) {
            StatePSSMLTSimple &s = states[stateID];
            s.setLargeStep(true);

            // Retry until we found some thing ...
            int nbTry = 0;
            while (s.impProp == 0) {
              if (nbTry % 10000 == 0 && nbTry != 0) {
                SLog(EInfo, "Difficult to initialize the chain [%i]: %i", stateID, nbTry);
              }
              if (nbTry != 0) {
                s.sampler->accept();
              }

              // Compute tile contribution and importance
              s.resetProposed();
              Spectrum basePath(0.f);
              auto pix_res = s.pathSampler->Lpix(s.proposed, s.sampler);
              s.pixelProp = pix_res.coord;
              s.impProp = importance(s.proposed, pix_res.index);

              ++nbTry;
            }

            SAssert(s.impProp != 0);
            s.setLargeStep(false);

            // Accept to initialize the chain
            s.accept(0.0, stateID);
          }

          SLog(EInfo, "Initialization finished");
        }

        // Before rendering prepare tiles
        // for multi thread writting (clear data)
        for (int i = 0; i < nCores; i++) {
          states[i].clear();
        }

        // Launch the rendering process
        Log(EInfo, "Do the rendering of tiles...");
        BlockScheduler runChains(nCores, nCores, 1);
        BlockScheduler::ComputeBlockFunction runfunc = [this, &states, sampleCount]
                (int stateID, int threadID) {
            StatePSSMLTSimple &s = states[stateID];
            ref<Sampler> random = s.indepSampler;
            ref<RenderingTechniqueTilde> pathSampler = s.pathSampler;

            for (int mutationCtr = 0;
                 mutationCtr < sampleCount && m_running;) {

              /**********************
               * Mutation
               */
              // Large step or not
              bool largeStep = random->next1D() < m_config.pLarge;
              s.setLargeStep(largeStep);

              // Pick on tile
              {
                s.resetProposed();

                // Compute tile contribution and importance
                auto pix_res = pathSampler->Lpix(s.proposed, s.sampler);
                s.pixelProp = pix_res.coord;
                s.impProp = importance(s.proposed, pix_res.index);

                mutationCtr += 1;
              }

              // Just a check in case something bad happen
              if (std::isnan(s.impProp) || s.impProp < 0) {
                SLog(EError, "Encountered a sample with luminance = %f, ignoring!", s.impProp);
              }

              /**************************
               * Classical MCMC
               */
              // Compute prob to accept the move
              Float a = std::min((Float) 1.0f, s.impProp / s.impCurr);

              // Make a decision to accept or not the sample
              bool accept;
              Float currentWeight, proposedWeight;
              if (a > 0) {
                currentWeight = 1 - a;
                proposedWeight = a;
                accept = (a == 1) || (random->next1D() < a);
              } else {
                currentWeight = 1;
                proposedWeight = 0;
                accept = false;
              }

              // Accumulate the results.
              s.cumulativeWeight += currentWeight;
              acceptanceRate.incrementBase();
              if (accept) {
                s.accept(proposedWeight, threadID);
                ++acceptanceRate;
              } else {
                s.reject(proposedWeight, threadID);
              }
            }
            s.flush(threadID);
        };
        runChains.run(runfunc);

        /************************************************************************
        * Get all the tiles and show the result
        ************************************************************************/
        size_t pixelCount = throughputBuffer->getPixelCount();
        ref<Bitmap> accumCurrIterBuffer = throughputBuffer->clone();
        accumCurrIterBuffer->clear();
        ref<Bitmap> dxCurrIterBuffer = throughputBuffer->clone();
        dxCurrIterBuffer->clear();
        ref<Bitmap> dyCurrIterBuffer = throughputBuffer->clone();
        dyCurrIterBuffer->clear();

        Spectrum *accumCurrIterPix = (Spectrum *) accumCurrIterBuffer->getData();
        Spectrum *dxCurrIterPix = (Spectrum *) dxCurrIterBuffer->getData();
        Spectrum *dyCurrIterPix = (Spectrum *) dyCurrIterBuffer->getData();
        for (int i = 0; i < nCores; i++) {
          Spectrum *accumStatePix = (Spectrum *) states[i].throughput->getData();
          Spectrum *dxStatePix = (Spectrum *) states[i].dx->getData();
          Spectrum *dyStatePix = (Spectrum *) states[i].dy->getData();
          for (size_t k = 0; k < pixelCount; k++) {
            accumCurrIterPix[k] += accumStatePix[k] / nCores;
            dxCurrIterPix[k] += dxStatePix[k] / nCores;
            dyCurrIterPix[k] += dyStatePix[k] / nCores;
          }
        }
        accumCurrIterBuffer->scale((m_cropSize.x * m_cropSize.y) / (Float) sampleCount);
        dxCurrIterBuffer->scale((m_cropSize.x * m_cropSize.y) / (Float) sampleCount);
        dyCurrIterBuffer->scale((m_cropSize.x * m_cropSize.y) / (Float) sampleCount);

        /************************************************************************
        * Accumulate the buffer overtime
        ************************************************************************/
        // Accumulate the buffer to accumBuffer
        Spectrum *accumPix = (Spectrum *) throughputBuffer->getData();
        Spectrum *dxPix = (Spectrum *) dxBuffer->getData();
        Spectrum *dyPix = (Spectrum *) dyBuffer->getData();
        {
          size_t k = 0;
          for (int y = 0; y < m_cropSize.y; y++) {
            for (int x = 0; x < m_cropSize.x; x++, k++) {
              accumPix[k] = (accumPix[k] * (iteration - 1) + accumCurrIterPix[k]) / iteration;
              dxPix[k] = (dxPix[k] * (iteration - 1) + dxCurrIterPix[k]) / iteration;
              dyPix[k] = (dyPix[k] * (iteration - 1) + dyCurrIterPix[k]) / iteration;
            }
          }
        }

        /************************************************************************
        * Rescale all the results based on the input
        ************************************************************************/
        // Throughput (compute the scaling)
        ref<Bitmap> finalBuffer = throughputBuffer->clone();
        Spectrum *finalPix = (Spectrum *) finalBuffer->getData();
        Float scaling = scaleGlobally(pixelCount, finalPix);

        // DX
        ref<Bitmap> finalDXBuffer = dxBuffer->clone();
        finalDXBuffer->scale(scaling);
        {
          ref<Bitmap> finalAbsBuffer = dxBuffer->clone();
          Spectrum *finalTarget = (Spectrum *) finalDXBuffer->getData();
          Spectrum *outTarget = (Spectrum *) finalAbsBuffer->getData();
          for (size_t t = 0; t < m_cropSize.x * m_cropSize.y; t++) {
            outTarget[t] = finalTarget[t].abs();
          }
          develop(scene, film, finalAbsBuffer, iteration, "_dxAbs_");
        }

        // DY
        ref<Bitmap> finalDYBuffer = dyBuffer->clone();
        finalDYBuffer->scale(scaling);
        {
          ref<Bitmap> finalAbsBuffer = dyBuffer->clone();
          Spectrum *finalTarget = (Spectrum *) finalDYBuffer->getData();
          Spectrum *outTarget = (Spectrum *) finalAbsBuffer->getData();
          for (size_t t = 0; t < m_cropSize.x * m_cropSize.y; t++) {
            outTarget[t] = finalTarget[t].abs();
          }
          develop(scene, film, finalAbsBuffer, iteration, "_dyAbs_");
        }

        // Throughput
        develop(scene, film, finalBuffer, iteration, "_");
        queue->signalRefresh(job); // Display the last computed image

        // Do the reconstruction
        {
          Reconstruction rec{};
          rec.reconstructL1 = m_reconstructL1;
          rec.reconstructL2 = m_reconstructL2;
          rec.reconstructUni = m_reconstructUni;
          rec.alpha = (float) 0.2f;

          auto throughputVector = bitmap2vec(finalBuffer);
          auto dxVector = bitmap2vec(finalDXBuffer);
          auto dyVector = bitmap2vec(finalDYBuffer);
          auto subPixelCount = size_t(3 * m_cropSize.x * m_cropSize.y);
          auto directVector = std::vector<float>(subPixelCount, 0.f);
          Reconstruction::Variance variance = {};

          auto rec_results = rec.reconstruct(film->getCropSize(),
                                             throughputVector, dxVector, dyVector, directVector,
                                             variance,
                                             PostProcessOption{
                                                     forceBlackPixels: false,
                                                     clampingValues: true
                                             });
          if (rec_results.size() == 1) {
            develop(scene, film, rec_results[0].img, iteration, "_recons_");
          } else {
            for (auto &result: rec_results) {
              develop(scene, film, result.img, iteration, "_" + result.name + "_");
            }
          }
        }

        /// Time it
        unsigned int milliseconds = renderingTimer->getMilliseconds();
        timeFile << (milliseconds / 1000.f) << ",\n";
        timeFile.flush();
        Log(EInfo, "Rendering time: %i, %i", milliseconds / 1000,
            milliseconds % 1000);
        renderingTimer->reset();


        // Go to the next iteration
        iteration += 1;
      }
      return m_running;
    }

    Float scaleGlobally(size_t pixelCount, Spectrum *accum) {
      /* Compute the luminance correction factor */
      double avgLuminance = 0;
      for (size_t i = 0; i < pixelCount; ++i) {
        Float currLum = accum[i].getLuminance();
        if (!std::isfinite(currLum) || currLum < 0) {
          SLog(EWarn, "global scale: %i not finite: %f", i, currLum);
        } else {
          avgLuminance += currLum;
        }
      }

      avgLuminance /= (Float) pixelCount;

      SLog(EInfo, "Average lum: %f", avgLuminance);
      Float luminanceFactor = m_config.luminance / avgLuminance;

      SLog(EInfo, "Lum factor: %f", luminanceFactor);
      for (size_t i = 0; i < pixelCount; ++i) {
        // No Direct added
        Float correction = luminanceFactor;
        Spectrum value = accum[i] * correction;
        accum[i] = value;
      }

      return luminanceFactor;
    }

    Float importance(const std::vector<Spectrum> &res, size_t index) {
      if (m_config.imp == ELuminance) {
          auto lum = 0.0;
          for (size_t t = 0; t < res.size(); t++) { lum += res[t].getLuminance(); }
          return lum;
      } else if (m_config.imp == EGD) {
          Spectrum lum = res[ECurr] * m_config.alpha;
          for(size_t t = 1; t < res.size(); t++) {
              lum += (res[t]-res[ECurr]).abs() * 0.5;
          }
          return lum.getLuminance();
      } else {
        SLog(EError, "Impossible to compute the target function");
      }
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

    MTS_DECLARE_CLASS()
private:
    ref<ParallelProcess> m_process;
    ref<RenderJob> m_nestedJob;
    SPSSMLTConfiguration m_config;
    bool m_running;

    // For the image tile sampling
    Vector2i m_cropSize;

    // For the reconstruction
    bool m_reconstructL1;
    bool m_reconstructL2;
    bool m_reconstructUni;
};

MTS_IMPLEMENT_CLASS_S(SPSSMLTGPT,
                      false, Integrator)

MTS_EXPORT_PLUGIN(SPSSMLTGPT,
                  "PSSMLT + GPT");
MTS_NAMESPACE_END
