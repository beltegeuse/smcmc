#ifndef MITSUBA_ALGO_RE_H
#define MITSUBA_ALGO_RE_H

#include "algo.h"

MTS_NAMESPACE_BEGIN

/**
 * The RE strategy explained inside the paper.
 * This is the approach that works the best in practice.
 * It is not an optimized implementation as thread synchronisation have an impact
 * to the global performance. Implementing a better scheduling approach will resolve
 * this small overhead issue.
 */
class RenderingAggressiveRE : public Rendering {
public:
    RenderingAggressiveRE(SPSSMLTConfiguration &config_,
                          Scene *scene, const Vector2i &cropSize_) :
            Rendering(config_, scene, cropSize_) {
    }

    void compute(std::vector<AbstractTile *> &tildes,
                 std::vector<std::vector<AbstractTile *>> &&tasks,
                 size_t sampleCount) override {
      // Create the random number vector
      ref_vector<Random> rand_gen;
      rand_gen.reserve(nCores);
      for (int idCore = 0; idCore < nCores; idCore++) {
        rand_gen.push_back(tildes[idCore]->sampler->getRandom());
      }

      BlockScheduler::ComputeBlockFunction
              advancefunc = [this, &tasks, &rand_gen](int tileID, int threadID) {
          auto pathSampler = pathSamplers[threadID];
          auto current_tiles = tasks[tileID];
          auto random = rand_gen[threadID];
          independentChainExploration(current_tiles, 1, pathSampler.get(), random.get());
      };
      BlockScheduler runClassical(tasks.size(), nCores, 128);
      BlockScheduler runRE(tasks.size() / 2, nCores, 64);

      // This code alternate horizonal RE, MCMC step, vertical RE, MCMC step
      // Moreover, each time, we change the pair of chains which we perform RE
      // so each chain will replicate with its 4 direct neighbors
      bool horizontal = true;
      int shift = 0;
      // Control at each frequency we perform RE step
      const int NON_RE_SAMPLES = (config.REFrequency + 1);

      for (size_t idSample = 0; idSample < sampleCount; idSample++) {
        if ((idSample + 1) % NON_RE_SAMPLES != 0) {
          runClassical.run(advancefunc);
        } else {
          std::vector<std::tuple<AbstractTile *, AbstractTile *, size_t>> groups;
          if (horizontal) {
            // Horizontal changes
            {
              groups.clear();
              groups.reserve(tildes.size() / 2);
              for (int y = 0; y < cropSize.y; y += 1) {
                for (int x = 0; x < cropSize.x; x += 2) {
                  size_t idTile1 = y * cropSize.x + x + shift;
                  size_t idTile2 = idTile1 + 1;
                  idTile1 = idTile1 % (cropSize.x * cropSize.y);
                  idTile2 = idTile2 % (cropSize.x * cropSize.y);

                  groups.push_back(std::make_tuple(tildes[idTile1], tildes[idTile2], idTile1));
                }
              }
            }

            BlockScheduler::ComputeBlockFunction
                    replicatefunc = [&](int tileID, int threadID) {
                auto current_group = groups[tileID];
                auto pathSampler = pathSamplers[threadID];
                auto random = rand_gen[threadID];
                RE(*std::get<0>(current_group), *std::get<1>(current_group), random.get(), pathSampler.get(),
                   horizontal, shift % 2 == 0);
            };
            runRE.run(replicatefunc);
          } else {
            // Vertical changes
            {
              groups.clear();
              groups.reserve(tildes.size() / 2);
              for (int y = 0; y < cropSize.y; y += 2) {
                for (int x = 0; x < cropSize.x; x += 1) {
                  size_t idTile1 = y * cropSize.x + x + shift * cropSize.x;
                  size_t idTile2 = idTile1 + cropSize.x;
                  idTile1 = idTile1 % (cropSize.x * cropSize.y);
                  idTile2 = idTile2 % (cropSize.x * cropSize.y);

                  groups.push_back(std::make_tuple(tildes[idTile1], tildes[idTile2], idTile1));
                }
              }
            }

            BlockScheduler::ComputeBlockFunction
                    replicatefunc = [&](int tileID, int threadID) {
                auto current_group = groups[tileID];
                auto pathSampler = pathSamplers[threadID];
                auto random = rand_gen[threadID];

                // Fetch data
                RE(*std::get<0>(current_group), *std::get<1>(current_group), random.get(), pathSampler.get(),
                   horizontal, shift % 2 == 0);
            };
            runRE.run(replicatefunc);
            shift += 1; // Make sure to change the shift mode
          }
          horizontal = !horizontal;
        }
      }
    }

    void dumpInfo(Scene *scene, Film *film, int iteration) {
      // Nothing
    }
};

MTS_NAMESPACE_END

#endif //MITSUBA_ALGO_TILED_AGRESSIVE_RE_H
