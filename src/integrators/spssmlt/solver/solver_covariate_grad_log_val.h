#ifndef MITSUBA_SOLVER_COVARIATE_GRAD_LOG_VAL_H
#define MITSUBA_SOLVER_COVARIATE_GRAD_LOG_VAL_H

#include "solver.h"
#include "../../reconstruction.h"

MTS_NAMESPACE_BEGIN
/**
 * This code is very similar to covariate reconstruction with uniform weight in G-PT
 * The inner tile gradient are scaled by the local normalization factor (noisy).
 * Then, this scaled gradient and initial scaled image version of the primal-domain image
 * is reconstructed.
 */
inline Spectrum log_value_val(Spectrum v) {
  Spectrum res;
  for (int i = 0; i < 3; i++) {
    if (v[i] < 1e-8) {
      res[i] = log(1e-8);
    } else {
      res[i] = log(v[i]);
    }
  }
  return res;
}

inline Spectrum exp_val(Spectrum v) {
  Spectrum res;
  for (int i = 0; i < 3; i++) {
    res[i] = exp(v[i]);
  }
  return res;
}

class TileCovariateGradLogVal : public TileSolver {
public:
    TileCovariateGradLogVal(const SPSSMLTConfiguration &config, const Vector2i &imgSize,
                            std::vector<AbstractTile *> &tiles) :
            TileSolver(config, imgSize, tiles) {}


    std::vector<Spectrum> compute_robust() {
      // Do the splatting for the tiles
      Spectrum *accum = new Spectrum[m_imgSize.x * m_imgSize.y];
      int *sampleCounts = new int[m_imgSize.x * m_imgSize.y];
      for (size_t i = 0; i < m_imgSize.x * m_imgSize.y; ++i) {
        accum[i] = Spectrum(0.f);
        sampleCounts[i] = 0;
      }

      //Push each of the tiles to the buffer
      for (AbstractTile *tile : m_tiles) {
        Point2i pixelLocation = tile->pixel(0);
        int bufferIndex = pixelLocation.y * m_imgSize.x + pixelLocation.x;
        Spectrum normalizedValue = tile->pixels[0].values_MC;
        accum[bufferIndex] += normalizedValue;
        sampleCounts[bufferIndex] += tile->nbSamplesUni;

//        for (size_t pixelIndex = 0; pixelIndex < tile->size(); ++pixelIndex) {
//          Spectrum normalizedValue = tile->get(pixelIndex) * curr_norm;
//
//          Point2i pixelLocation = tile->pixel(pixelIndex);
//          if (pixelLocation.x < 0 || pixelLocation.x >= m_imgSize.x || pixelLocation.y < 0
//              || pixelLocation.y >= m_imgSize.y)
//            continue;
//
//          int bufferIndex = pixelLocation.y * m_imgSize.x + pixelLocation.x;
//          accum[bufferIndex] += normalizedValue * sampleCount;
//          sampleCounts[bufferIndex] += sampleCount;
//        }
      }

      //Scale the accumulated flux by the sample counts
      for (size_t i = 0; i < m_imgSize.x * m_imgSize.y; ++i) {
        if (sampleCounts[i] == 0.0) {
          accum[i] = Spectrum(0.f);
        } else {
          accum[i] /= (Float) sampleCounts[i];
        }
      }

      std::vector<Spectrum> th(m_imgSize.x * m_imgSize.y, Spectrum(0.f));
      for (size_t y = 0; y < m_imgSize.y; y++) {
        for (size_t x = 0; x < m_imgSize.x; x++) {
          size_t t = y * m_imgSize.x + x;
          th[t] = accum[t];
        }
      }

      delete[] accum;
      delete[] sampleCounts;
      return th;
    }

    ReconstructionOptions solve() override {
      auto tile_size = m_tiles[0]->size();
      if (tile_size != 3 && tile_size != 5) {
        SLog(EError, "Only support L tile shapes");
      }

      auto throughput = [&]() -> std::vector<Spectrum> {
          auto throughput_float = compute_robust();
          std::vector<Spectrum> tmp(m_imgSize.x * m_imgSize.y, Spectrum(0.f));
          for (size_t k = 0; k < m_imgSize.x * m_imgSize.y; k++) {
            tmp[k] = log_value_val(throughput_float[k]);
          }
          return tmp;
      }();
      std::vector<Spectrum> dx(m_imgSize.x * m_imgSize.y, Spectrum(0.f));
      std::vector<Spectrum> dy(m_imgSize.x * m_imgSize.y, Spectrum(0.f));

      for (size_t y = 0; y < m_imgSize.y; y++) {
        for (size_t x = 0; x < m_imgSize.x; x++) {
          size_t curr_id = y * m_imgSize.x + x;
          if (x != 0) {
            size_t next_id = curr_id - 1;
            dx[next_id] += log_value_val(m_tiles[curr_id]->get(0)) - log_value_val(m_tiles[curr_id]->get(4));
          }
          if (y != 0) {
            size_t next_id = curr_id - m_imgSize.x;
            dy[next_id] += log_value_val(m_tiles[curr_id]->get(0)) - log_value_val(m_tiles[curr_id]->get(1));
          }
          if(x != m_imgSize.x - 1 ) {
            size_t next_id = curr_id + 1;
            dx[curr_id] += log_value_val(m_tiles[curr_id]->get(2)) - log_value_val(m_tiles[curr_id]->get(0));
          }
          if (y != m_imgSize.y - 1) {
            size_t next_id = curr_id + m_imgSize.x;
            dy[curr_id] += log_value_val(m_tiles[curr_id]->get(3)) - log_value_val(m_tiles[curr_id]->get(0));
          }
        }
      }
      for (size_t y = 0; y < m_imgSize.y; y++) {
        for (size_t x = 0; x < m_imgSize.x; x++) {
          size_t curr_id = y * m_imgSize.x + x;
          dx[curr_id] *= 0.5;
          dy[curr_id] *= 0.5;
        }
      }

      // Here reuse gradient-domain reconstruction
      // We only perform L1 reconstruction here
      Reconstruction rec{};
      rec.reconstructL1 = true;
      rec.reconstructL2 = false;
      rec.reconstructUni = false;
      rec.alpha = (float) m_config.alpha;
      rec.nbIteration = m_config.iterations;

      auto spec2floatvec = [&](const std::vector<Spectrum> &s) -> std::vector<float> {
          std::vector<float> vec(m_imgSize.x * m_imgSize.y * 3);
          for (size_t i = 0; i < m_imgSize.x * m_imgSize.y; i++) {
            vec[i * 3 + 0] = s[i][0];
            vec[i * 3 + 1] = s[i][1];
            vec[i * 3 + 2] = s[i][2];
          }
          return vec;
      };

      auto throughputVector = spec2floatvec(throughput);
      auto dxVector = spec2floatvec(dx);
      auto dyVector = spec2floatvec(dy);
      auto subPixelCount = size_t(3 * m_imgSize.x * m_imgSize.y);
      auto directVector = std::vector<float>(subPixelCount, 0.f);
      Reconstruction::Variance variance = {};

      auto rec_results = rec.reconstruct(m_imgSize,
                                         throughputVector, dxVector, dyVector, directVector,
                                         variance,
                                         PostProcessOption{
                                                 forceBlackPixels: false,
                                                 clampingValues: false
                                         });

      ref<Bitmap> result_bitmap = rec_results[0].img->clone();
      {
        auto data = (const Spectrum*)rec_results[0].img->getFloatData();
        auto data_out = (Spectrum*)result_bitmap->getFloatData();
        for (size_t k = 0; k < m_imgSize.x * m_imgSize.y; k++) {
          data_out[k] = exp_val(data[k]);
        }
      }


      return ReconstructionOptions{.needGlobalRescale = false,
              .res = result_bitmap};
    }

};


MTS_NAMESPACE_END

#endif //MITSUBA_SOLVER_COVARIATE_GRAD_H
