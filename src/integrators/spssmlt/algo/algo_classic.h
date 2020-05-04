#ifndef MITSUBA_ALGO_TILED_CLASSIC_H
#define MITSUBA_ALGO_TILED_CLASSIC_H

#include "algo.h"

MTS_NAMESPACE_BEGIN

/**
 * Implementation of MH where no RE is performed
 */
class RenderingClassical : public Rendering {
public:
    RenderingClassical(SPSSMLTConfiguration &config_,
                       Scene *scene, const Vector2i &cropSize_) : Rendering(config_, scene, cropSize_) {}

    void compute(std::vector<AbstractTile *> &tildes,
                 std::vector<std::vector<AbstractTile *>> &&tasks,
                 size_t sampleCount) override {
      // Create the random number vector
      ref_vector<Random> rand_gen;
      rand_gen.reserve(nCores);
      for (int idCore = 0; idCore < nCores; idCore++) {
        rand_gen.push_back(tildes[idCore]->sampler->getRandom());
      }

      BlockScheduler runChains(tasks.size(), nCores, config.MCMC == ERE ? 1 : 128);
      BlockScheduler::ComputeBlockFunction
              runfunc = [this, &tasks, sampleCount, &rand_gen](int tileID, int threadID) {
          // Fetch data
          auto pathSampler = pathSamplers[threadID];
          auto current_tiles = tasks[tileID];
          auto random = rand_gen[threadID];
          independentChainExploration(current_tiles, sampleCount, pathSampler.get(), random.get());

          // Make sure that all states are flush
          for (int i = 0; i < current_tiles.size(); i++) {
            current_tiles[i]->flush();
          }
      };
      runChains.run(runfunc);
    }
};

MTS_NAMESPACE_END


#endif //MITSUBA_ALGO_TILED_CLASSIC_H
