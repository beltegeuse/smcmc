#ifndef MITSUBA_SOLVER_COVARIATE_GRAD_LOG_H
#define MITSUBA_SOLVER_COVARIATE_GRAD_LOG_H

#include "solver.h"

MTS_NAMESPACE_BEGIN
/**
 * This code is very similar to covariate reconstruction with uniform weight in G-PT
 * The inner tile gradient are scaled by the local normalization factor (noisy).
 * Then, this scaled gradient and initial scaled image version of the primal-domain image
 * is reconstructed.
 * 
 * This is a simpler implementation of uniform weighting approach
 * For L1 reconstruction, look at solver_covariate_grad_log_val.h
 * 
 */
#define LogFunc log10
#define ExpFunc exp10

struct LogValue {
  Float v;
  bool usable;
};

inline LogValue log_value(Float v, const Float eps) {
  if(v == 0.0 || !std::isfinite(v)) {
    if(eps == 0.0) {
      return LogValue{0.0, false};
    } else {
      return LogValue{LogFunc(eps), true};
    }
  }
  return LogValue{LogFunc(v), true};
}

inline LogValue combined(Float new_v, LogValue pred, const Float eps) {
  if(!pred.usable) {
    return log_value(new_v, eps);
  } else {
    auto v = log_value(new_v, eps);
    if(v.usable) {
        return LogValue{LogFunc((ExpFunc(v.v) + ExpFunc(pred.v))*0.5), true};
    } else {
      return pred;
    }
  }
}

class TileCovariateGradLog: public TileSolver {
public:
  TileCovariateGradLog(const SPSSMLTConfiguration &config, const Vector2i &imgSize, std::vector<AbstractTile *> &tiles) :
      TileSolver(config, imgSize, tiles) {}

  ReconstructionOptions solve() override {
    auto tile_size = m_tiles[0]->size();
    if(tile_size != 3 && tile_size != 5) {
      SLog(EError, "Only support L tile shapes");
    }

    const Float LOG_EPSILON = 0.0;

    auto throughput = [&]() -> std::vector<LogValue> {
      auto throughput_float = compute_throughput();
      std::vector<LogValue> tmp(m_imgSize.x * m_imgSize.y, {0.f, false});
      for(size_t k = 0; k < m_imgSize.x * m_imgSize.y; k++) {
        tmp[k] = log_value(throughput_float[k],LOG_EPSILON);
      }
      return tmp;
    }();
    std::vector<LogValue> dx(m_imgSize.x * m_imgSize.y, {0.f, false});
    std::vector<LogValue> dy(m_imgSize.x * m_imgSize.y, {0.f, false});

    if(tile_size == 3) {
      for (size_t k = 0; k < m_imgSize.x * m_imgSize.y; k++) {
        dx[k] = log_value((m_tiles[k]->lum(1)) / (m_tiles[k]->lum(0)), LOG_EPSILON); // +X
        dy[k] = log_value((m_tiles[k]->lum(2)) / (m_tiles[k]->lum(0)), LOG_EPSILON); // +Y
      }
    } else {
      for (size_t k = 0; k < m_imgSize.x * m_imgSize.y; k++) {
        dx[k] = log_value((m_tiles[k]->lum(2)) / (m_tiles[k]->lum(0)), LOG_EPSILON); // +X
        dy[k] = log_value((m_tiles[k]->lum(3)) / (m_tiles[k]->lum(0)), LOG_EPSILON); // +Y
      }

      for (size_t y = 0; y < m_imgSize.y; y++) {
        for (size_t x = 0; x < m_imgSize.x; x++) {
          size_t curr_id = y * m_imgSize.x + x;
          auto center_curr = m_tiles[curr_id]->lum(0);
          if(x > 0) {
            size_t next_id = curr_id - 1;
            dx[next_id] = combined((center_curr) / (m_tiles[curr_id]->lum(4)), dx[next_id], LOG_EPSILON); // -X
          }
          if(y > 0) {
            size_t next_id = curr_id - m_imgSize.x;
            dy[next_id] = combined((center_curr) / (m_tiles[curr_id]->lum(1)), dy[next_id], LOG_EPSILON); // -Y
          }
        }
      }
    }


    for (size_t t = 0; t < m_config.iterations; t++) {
      std::vector<Float> next_throughput(m_imgSize.x * m_imgSize.y, Float(0.f));
#pragma omp parallel for schedule(dynamic)
      for (size_t y = 0; y < m_imgSize.y; y++) {
        for (size_t x = 0; x < m_imgSize.x; x++) {
          size_t curr_id = y * m_imgSize.x + x;

          Float& next = next_throughput[curr_id];


          // Compute the control variate
          int nb_sum = 0;
          if(throughput[curr_id].usable) {
            next += throughput[curr_id].v;
            nb_sum += 1;
          }

          // Left
          if (x != 0) { ;
            size_t next_id = curr_id - 1;
            if(throughput[next_id].usable && dx[next_id].usable) {
              next += throughput[next_id].v + dx[next_id].v;
              nb_sum += 1;
            }
          }

          // Right
          if (x != m_imgSize.x - 1) {
            size_t next_id = curr_id + 1;
            if(throughput[next_id].usable && dx[curr_id].usable) {
              next += throughput[next_id].v - dx[curr_id].v;
              nb_sum += 1;
            }
          }

          // Top
          if (y != 0) {
            size_t next_id = curr_id - m_imgSize.x;
            if(throughput[next_id].usable && dy[next_id].usable) {
              next += throughput[next_id].v + dy[next_id].v;
              nb_sum += 1;
            }
          }

          // Bottom
          if (y != m_imgSize.y - 1) {
            size_t next_id = curr_id + m_imgSize.x;
            if(throughput[next_id].usable && dy[curr_id].usable) {
              next += throughput[next_id].v - dy[curr_id].v;
              nb_sum += 1;
            }
          }

          next /= nb_sum;
        }
      } // Finish to treat the image
      for (size_t k = 0; k < m_imgSize.x * m_imgSize.y; k++) {
        throughput[k].v = next_throughput[k];
      }
    }



    // Fill the output
    {
      for (size_t k = 0; k < m_imgSize.x * m_imgSize.y; k++) {
        Float target_lum = std::max(ExpFunc(throughput[k].v), Float(0.0));
        Float current_lum = m_tiles[k]->lum(0);
        if(current_lum != 0.f) {
          m_tiles[k]->applyScale(target_lum / current_lum);
        } else {
          m_tiles[k]->applyScale(0.f); // Make it black
        }
      }
    }


    ref<Bitmap> accumBuffer = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat, m_imgSize);
    accumBuffer->clear();
    Spectrum *accum = (Spectrum *) accumBuffer->getData();
    for (size_t k = 0; k < m_imgSize.x * m_imgSize.y; k++) {
      accum[k] = m_tiles[k]->get(0);
    }

    return ReconstructionOptions {.needGlobalRescale = false,
                                  .res = accumBuffer};
  }

};


MTS_NAMESPACE_END

#endif //MITSUBA_SOLVER_COVARIATE_GRAD_H
