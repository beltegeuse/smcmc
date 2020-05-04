#ifndef MITSUBA_ITERATIVE_REWEIGHTED_H
#define MITSUBA_ITERATIVE_REWEIGHTED_H

#include "solver.h"

MTS_NAMESPACE_BEGIN

#define SUM_POS 1
class TileIterativeReweighted : public TileSolver {
    int dumpID = 0;

    struct TileCache {
        Float values[5] = {0.f, 0.f, 0.f, 0.f, 0.f};
        Float sum = 0.0;
        Float avg_non_null = 0.0;
        Float sumMC = 0.0;

        void initialize(AbstractTile *t, int channel,
                        const std::vector<Spectrum> &mc_estimate, const Vector2i &imgSize) {
          int nb_non_null = 0;
          sum = 0;
          sumMC = 0;
          for (int i = 0; i < 5; i++) {
            values[i] = t->get(i)[channel];
            // For the MC estimate use the more accurate one
            // not the one inside the tile
            auto pixel_loc = t->pixel(i);

            Float valueMC = 0.0;
            if (pixel_loc.x < 0 || pixel_loc.x >= imgSize.x || pixel_loc.y < 0
                || pixel_loc.y >= imgSize.y) {
            } else {
              int bufferIndex = t->pixel(i).y * imgSize.x + t->pixel(i).x;
              valueMC = mc_estimate[bufferIndex][channel];
            }

            if (values[i] > 0) {
              sum += values[i];
              sumMC += valueMC;
            }
            // Do the sum
            if (values[i] > 0) {
              nb_non_null += 1;
            }
          }
          avg_non_null = sum / nb_non_null;
          if(nb_non_null != 0) {
            sum = sum / nb_non_null;
            sumMC = sumMC / nb_non_null;
          }
        }

        float operator[](size_t index) const { return values[index]; }
    };

public:
    TileIterativeReweighted(const SPSSMLTConfiguration &config, const Vector2i &imgSize, std::vector<AbstractTile *> &tiles)
            :
            TileSolver(config, imgSize, tiles) {}

    std::vector<Spectrum> MCEstimates() {
      std::vector<Spectrum> accum(m_imgSize.x * m_imgSize.y, Spectrum(0.0));
      int *sampleCounts = new int[m_imgSize.x * m_imgSize.y]();

      //Push each of the tiles to the buffer
      for (AbstractTile *tile : m_tiles) {
        int sampleCount = tile->nbSamplesUni;

        for (size_t pixelIndex = 0; pixelIndex < tile->size(); ++pixelIndex) {
          Spectrum normalizedValue = tile->pixels[pixelIndex].values_MC;

          Point2i pixelLocation = tile->pixel(pixelIndex);
          if (pixelLocation.x < 0 || pixelLocation.x >= m_imgSize.x || pixelLocation.y < 0
              || pixelLocation.y >= m_imgSize.y)
            continue;

          int bufferIndex = pixelLocation.y * m_imgSize.x + pixelLocation.x;
          accum[bufferIndex] += normalizedValue;
          sampleCounts[bufferIndex] += sampleCount;
        }

      }

      //Scale the accumulted flux by the sample counts
      for (size_t i = 0, yy = 0; yy < m_imgSize.y; ++yy)
        for (size_t xx = 0; xx < m_imgSize.x; ++xx, ++i) {
          if (sampleCounts[i] == 0.0) {
            accum[i] = Spectrum(0.f);
          } else {
            accum[i] /= (Float) sampleCounts[i];
          }
        }
      delete[] sampleCounts;
      return accum;
    }

    void solveScaling(std::vector<Float> &b, int channel,
                      const std::vector<Spectrum> &mc_estimate) {
        const bool DIAG_USE = true;//!m_config.REWeighting;
        const bool ONE_OVERLAP = true;//!m_config.REWeighting;
        const bool ATTACH_TILE = true;
        std::vector<Float> w(m_imgSize.x*m_imgSize.y, 1.0);

        std::vector<TileCache> lum_buff(m_imgSize.x * m_imgSize.y);
        for (size_t y = 0; y < m_imgSize.y; y++) {
            for (size_t x = 0; x < m_imgSize.x; x++) {
                size_t t = y * m_imgSize.x + x;
                AbstractTile *curr_tile = m_tiles[t];
                lum_buff[t].initialize(curr_tile, channel, mc_estimate, m_imgSize);

                b[t] = curr_tile->getNorm();
            }
        }
        std::vector<Float> b0 = b;

        for(auto iter = 0; iter < 20; iter++) {
            for (size_t t = 0; t < m_config.iterations; t++) {
                std::vector<Float> b_next(m_imgSize.x * m_imgSize.y, Float(0.f));
#pragma omp parallel for schedule(dynamic)
                for (size_t y = 0; y < m_imgSize.y; y++) {
                    for (size_t x = 0; x < m_imgSize.x; x++) {
                        // Compute the next step
                        size_t curr_id = y * m_imgSize.x + x;
                        const TileCache &curr_tile = lum_buff[curr_id];

                        struct ResultForce {
                            Float force = 0.f;
                            Float pos = 0.f;
                        };
                        ResultForce res = {};

                        auto apply_force = [this](ResultForce &r, Float b1, Float v1, Float b2, Float v2, Float w1,
                                              Float w2) -> void {
                            // Computes the weights
                            Float w = std::min(w1,w2);//std::max(w1, w2);
                            if(!m_config.errorWeights) {
                                w = 1.0;
                            }
                            if(m_config.customWeights) {
                                Float w_c = std::min(v2 / v1, v1 / v2);
                                w_c *= w_c;
                                w *= w_c;
                            }

                            // function f
                            Float f = 0.5 * (v1 * b1 - v2 * b2);
                            if (std::isfinite(f) && v1 != 0.0 && v2 != 0.0) {
                                r.force += w * f;
#if SUM_POS
                                r.pos += w * v1;
#else
                                r.pos += w;
#endif
                            }
                        };

                        if (curr_tile.sum == 0.0) {
                            b_next[curr_id] = b[curr_id];
                            continue;
                        }

                        if (ATTACH_TILE) {
                            Float force = b[curr_id] * lum_buff[curr_id].sum - lum_buff[curr_id].sumMC;
                            Float weight = m_config.alpha * w[curr_id];
                            res.force += weight * force;
#if SUM_POS
                            res.pos += weight * lum_buff[curr_id].sum;
#else
                            res.pos += weight * 5;
#endif
                        }

                        // Left
                        if (x != 0) {
                            size_t next_id = curr_id - 1;
                            const TileCache &next_tile = lum_buff[next_id];
                            apply_force(res, b[curr_id], curr_tile[ECurr],
                                        b[next_id], next_tile[EXP], w[curr_id], w[next_id]);
                            apply_force(res, b[curr_id], curr_tile[EXN],
                                        b[next_id], next_tile[ECurr], w[curr_id], w[next_id]);
                        }
                        // Right
                        if (x != m_imgSize.x - 1) {
                            size_t next_id = curr_id + 1;
                            const TileCache &next_tile = lum_buff[next_id];
                            apply_force(res, b[curr_id], curr_tile[ECurr],
                                        b[next_id], next_tile[EXN], w[curr_id], w[next_id]);
                            apply_force(res, b[curr_id], curr_tile[EXP],
                                        b[next_id], next_tile[ECurr], w[curr_id], w[next_id]);
                        }
                        // Top
                        if (y != 0) {
                            size_t next_id = curr_id - m_imgSize.x;
                            const TileCache &next_tile = lum_buff[next_id];
                            apply_force(res, b[curr_id], curr_tile[ECurr],
                                        b[next_id], next_tile[EYP], w[curr_id], w[next_id]);
                            apply_force(res, b[curr_id], curr_tile[EYN],
                                        b[next_id], next_tile[ECurr], w[curr_id], w[next_id]);
                        }
                        // Bottom
                        if (y != m_imgSize.y - 1) {
                            size_t next_id = curr_id + m_imgSize.x;
                            const TileCache &next_tile = lum_buff[next_id];
                            apply_force(res, b[curr_id], curr_tile[ECurr],
                                        b[next_id], next_tile[EYN], w[curr_id], w[next_id]);
                            apply_force(res, b[curr_id], curr_tile[EYP],
                                        b[next_id], next_tile[ECurr], w[curr_id], w[next_id]);
                        }

                        // More overlapping
                        if (DIAG_USE) {
                            if (x != 0 && y != 0) {
                                size_t next_id = curr_id - m_imgSize.x - 1;
                                const TileCache &next_tile = lum_buff[next_id];
                                apply_force(res, b[curr_id], curr_tile[EXN],
                                            b[next_id], next_tile[EYP], w[curr_id], w[next_id]);
                                apply_force(res, b[curr_id], curr_tile[EYN],
                                            b[next_id], next_tile[EXP], w[curr_id], w[next_id]);
                            }
                            if (x != m_imgSize.x - 1 && y != m_imgSize.y - 1) {
                                size_t next_id = curr_id + m_imgSize.x + 1;
                                const TileCache &next_tile = lum_buff[next_id];
                                apply_force(res, b[curr_id], curr_tile[EXP],
                                            b[next_id], next_tile[EYN], w[curr_id], w[next_id]);
                                apply_force(res, b[curr_id], curr_tile[EYP],
                                            b[next_id], next_tile[EXN], w[curr_id], w[next_id]);
                            }
                            if (x != 0 && y != m_imgSize.y - 1) {
                                size_t next_id = curr_id + m_imgSize.x - 1;
                                const TileCache &next_tile = lum_buff[next_id];
                                apply_force(res, b[curr_id], curr_tile[EXN],
                                            b[next_id], next_tile[EYN], w[curr_id], w[next_id]);
                                apply_force(res, b[curr_id], curr_tile[EYP],
                                            b[next_id], next_tile[EXP], w[curr_id], w[next_id]);
                            }
                            if (x != m_imgSize.x - 1 && y != 0) {
                                size_t next_id = curr_id - m_imgSize.x + 1;
                                const TileCache &next_tile = lum_buff[next_id];
                                apply_force(res, b[curr_id], curr_tile[EXP],
                                            b[next_id], next_tile[EYP], w[curr_id], w[next_id]);
                                apply_force(res, b[curr_id], curr_tile[EYN],
                                            b[next_id], next_tile[EXN], w[curr_id], w[next_id]);
                            }
                        }

                        if (ONE_OVERLAP) {
                            if (x > 1) {
                                size_t next_id = curr_id - 2;
                                const TileCache &next_tile = lum_buff[next_id];
                                apply_force(res, b[curr_id], curr_tile[EXN],
                                            b[next_id], next_tile[EXP], w[curr_id], w[next_id]);
                            }

                            // Right
                            if (x < m_imgSize.x - 2) {
                                size_t next_id = curr_id + 2;
                                const TileCache &next_tile = lum_buff[next_id];
                                apply_force(res, b[curr_id], curr_tile[EXP],
                                            b[next_id], next_tile[EXN], w[curr_id], w[next_id]);
                            }

                            // Top
                            if (y > 1) {
                                size_t next_id = curr_id - 2 * m_imgSize.x;
                                const TileCache &next_tile = lum_buff[next_id];
                                apply_force(res, b[curr_id], curr_tile[EYN],
                                            b[next_id], next_tile[EYP], w[curr_id], w[next_id]);
                            }

                            // Bottom
                            if (y < m_imgSize.y - 2) {
                                size_t next_id = curr_id + 2 * m_imgSize.x;
                                const TileCache &next_tile = lum_buff[next_id];
                                apply_force(res, b[curr_id], curr_tile[EYP],
                                            b[next_id], next_tile[EYN], w[curr_id], w[next_id]);
                            }
                        }

                        if (res.pos != 0.0) {
                            // average forces * 0.5 -> gives how much the tile move
                            // dividing by tile luminance average give back the normalization factor?
#if SUM_POS
#else
                            res.pos *= lum_buff[curr_id].avg_non_null;
#endif
                            Float new_value = b[curr_id] - (res.force / res.pos);
                            b_next[curr_id] = new_value;
                        } else {
                            b_next[curr_id] = b[curr_id];
                        }
                    }
                } // Finish to treat the image
                // Optimization from Rex for the coforce
                auto nbNegativeNormalization = 0;
                for (size_t k = 0; k < m_imgSize.x * m_imgSize.y; k++) {
                    if (b_next[k] <= 0.0) {
                        nbNegativeNormalization += 1;
                    }
                    b[k] = b_next[k];
                }

                if (nbNegativeNormalization > 0) {
//          SLog(EInfo, "%i: %i negative normalization factors (%f)", t, nbNegativeNormalization,
//               nbNegativeNormalization / Float(m_imgSize.x * m_imgSize.y));
                }
            }

            if(m_config.errorWeights){
                // Compute the error
                std::vector<Float> w2 = w;
                #pragma omp parallel for schedule(dynamic)
                for (size_t y = 0; y < m_imgSize.y; y++) {
                    for (size_t x = 0; x < m_imgSize.x; x++) {
                        // Compute the next step
                        size_t curr_id = y * m_imgSize.x + x;
                        const TileCache &curr_tile = lum_buff[curr_id];
                        Float error = 0.0;

                        if (ATTACH_TILE) {
                            Float force = b[curr_id] * lum_buff[curr_id].sum - lum_buff[curr_id].sumMC;
                            Float weight = m_config.alpha; //* w[curr_id];
                            error += weight * std::abs(force);
                        }

                        auto add_error = [this](Float &r, Float b1, Float v1, Float b2, Float v2, Float w1, Float w2) -> void {
                            Float w = 1.0;
                            if(m_config.customWeights) {
                                Float w_c = std::min(v2 / v1, v1 / v2);
                                w_c*= w_c;
                                w *= w_c;
                            }
                            Float f = 0.5 * (v1 * b1 - v2 * b2);
                            if (std::isfinite(f) && v1 != 0.0 && v2 != 0.0) {
                                r += std::abs(f) * w;
                            }
                        };

                        if (x != 0) {
                            size_t next_id = curr_id - 1;
                            const TileCache &next_tile = lum_buff[next_id];
                            add_error(error, b[curr_id], curr_tile[ECurr],
                                      b[next_id], next_tile[EXP], w[curr_id], w[next_id]);
                            add_error(error, b[curr_id], curr_tile[EXN],
                                      b[next_id], next_tile[ECurr], w[curr_id], w[next_id]);
                        }
                        // Right
                        if (x != m_imgSize.x - 1) {
                            size_t next_id = curr_id + 1;
                            const TileCache &next_tile = lum_buff[next_id];
                            add_error(error, b[curr_id], curr_tile[ECurr],
                                      b[next_id], next_tile[EXN], w[curr_id], w[next_id]);
                            add_error(error, b[curr_id], curr_tile[EXP],
                                      b[next_id], next_tile[ECurr], w[curr_id], w[next_id]);
                        }
                        // Top
                        if (y != 0) {
                            size_t next_id = curr_id - m_imgSize.x;
                            const TileCache &next_tile = lum_buff[next_id];
                            add_error(error, b[curr_id], curr_tile[ECurr],
                                      b[next_id], next_tile[EYP], w[curr_id], w[next_id]);
                            add_error(error, b[curr_id], curr_tile[EYN],
                                      b[next_id], next_tile[ECurr], w[curr_id], w[next_id]);
                        }
                        // Bottom
                        if (y != m_imgSize.y - 1) {
                            size_t next_id = curr_id + m_imgSize.x;
                            const TileCache &next_tile = lum_buff[next_id];
                            add_error(error, b[curr_id], curr_tile[ECurr],
                                      b[next_id], next_tile[EYN], w[curr_id], w[next_id]);
                            add_error(error, b[curr_id], curr_tile[EYP],
                                      b[next_id], next_tile[ECurr], w[curr_id], w[next_id]);
                        }

                        // More overlapping
                        if (DIAG_USE) {
                            if (x != 0 && y != 0) {
                                size_t next_id = curr_id - m_imgSize.x - 1;
                                const TileCache &next_tile = lum_buff[next_id];
                                add_error(error, b[curr_id], curr_tile[EXN],
                                          b[next_id], next_tile[EYP], w[curr_id], w[next_id]);
                                add_error(error, b[curr_id], curr_tile[EYN],
                                          b[next_id], next_tile[EXP], w[curr_id], w[next_id]);
                            }
                            if (x != m_imgSize.x - 1 && y != m_imgSize.y - 1) {
                                size_t next_id = curr_id + m_imgSize.x + 1;
                                const TileCache &next_tile = lum_buff[next_id];
                                add_error(error, b[curr_id], curr_tile[EXP],
                                          b[next_id], next_tile[EYN], w[curr_id], w[next_id]);
                                add_error(error, b[curr_id], curr_tile[EYP],
                                          b[next_id], next_tile[EXN], w[curr_id], w[next_id]);
                            }
                            if (x != 0 && y != m_imgSize.y - 1) {
                                size_t next_id = curr_id + m_imgSize.x - 1;
                                const TileCache &next_tile = lum_buff[next_id];
                                add_error(error, b[curr_id], curr_tile[EXN],
                                          b[next_id], next_tile[EYN], w[curr_id], w[next_id]);
                                add_error(error, b[curr_id], curr_tile[EYP],
                                          b[next_id], next_tile[EXP], w[curr_id], w[next_id]);
                            }
                            if (x != m_imgSize.x - 1 && y != 0) {
                                size_t next_id = curr_id - m_imgSize.x + 1;
                                const TileCache &next_tile = lum_buff[next_id];
                                add_error(error, b[curr_id], curr_tile[EXP],
                                          b[next_id], next_tile[EYP], w[curr_id], w[next_id]);
                                add_error(error, b[curr_id], curr_tile[EYN],
                                          b[next_id], next_tile[EXN], w[curr_id], w[next_id]);
                            }
                        }

                        if (ONE_OVERLAP) {
                            if (x > 1) {
                                size_t next_id = curr_id - 2;
                                const TileCache &next_tile = lum_buff[next_id];
                                add_error(error, b[curr_id], curr_tile[EXN],
                                          b[next_id], next_tile[EXP], w[curr_id], w[next_id]);
                            }

                            // Right
                            if (x < m_imgSize.x - 2) {
                                size_t next_id = curr_id + 2;
                                const TileCache &next_tile = lum_buff[next_id];
                                add_error(error, b[curr_id], curr_tile[EXP],
                                          b[next_id], next_tile[EXN], w[curr_id], w[next_id]);
                            }

                            // Top
                            if (y > 1) {
                                size_t next_id = curr_id - 2 * m_imgSize.x;
                                const TileCache &next_tile = lum_buff[next_id];
                                add_error(error, b[curr_id], curr_tile[EYN],
                                          b[next_id], next_tile[EYP], w[curr_id], w[next_id]);
                            }

                            // Bottom
                            if (y < m_imgSize.y - 2) {
                                size_t next_id = curr_id + 2 * m_imgSize.x;
                                const TileCache &next_tile = lum_buff[next_id];
                                add_error(error, b[curr_id], curr_tile[EYP],
                                          b[next_id], next_tile[EYN], w[curr_id], w[next_id]);
                            }
                        }

                        w2[curr_id] = 1.0 / (error + std::max(0.05*std::pow(0.5, iter), 0.0001));
                    }
                }

                Float sumW2 = 0.0;
                for(auto wc: w2) {
                    sumW2 += wc;
                }
                for(Float& wc: w2) {
                    wc *= (m_imgSize.x*m_imgSize.y) / sumW2;
                }
                w = w2; // Update
            }
        }
    }

    ReconstructionOptions solve() override {
      auto tile_size = m_tiles[0]->size();
      SAssert(tile_size == 5);

      // Get more robust MC estimates
      // By combining the different uniform estimates
      auto mc_estimate = MCEstimates();

      // Here we will align each tiles color independently.
      // we could also do only luminance based alignment procedure
      // by this pervent us to remove the color noise in the tile's estimates.
      std::vector<Float> bRed(m_imgSize.x * m_imgSize.y, 0.f);
      solveScaling(bRed, 0, mc_estimate);
      std::vector<Float> bGreen(m_imgSize.x * m_imgSize.y, 0.f);
      solveScaling(bGreen, 1, mc_estimate);
      std::vector<Float> bBlue(m_imgSize.x * m_imgSize.y, 0.f);
      solveScaling(bBlue, 2, mc_estimate);

      // Combine all scaling factor to produce the last image
      ref<Bitmap> accumBuffer = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat, m_imgSize);
      accumBuffer->clear();
      Spectrum *accum = (Spectrum *) accumBuffer->getData();
      int *sampleCounts = new int[m_imgSize.x * m_imgSize.y]();
      for (size_t k = 0; k < m_imgSize.x * m_imgSize.y; k++) {
        AbstractTile *tile = m_tiles[k];
        int sampleCount = tile->nbSamples;
        Float rgbScale[3] = {bRed[k], bGreen[k], bBlue[k]};

        for (size_t pixelIndex = 0; pixelIndex < tile->size(); ++pixelIndex) {
          Spectrum normalizedValue = tile->pixels[pixelIndex].values * Spectrum(rgbScale);

          Point2i pixelLocation = tile->pixel(pixelIndex);
          if (pixelLocation.x < 0 || pixelLocation.x >= m_imgSize.x || pixelLocation.y < 0
              || pixelLocation.y >= m_imgSize.y)
            continue;

          int bufferIndex = pixelLocation.y * m_imgSize.x + pixelLocation.x;
          accum[bufferIndex] += normalizedValue;
          sampleCounts[bufferIndex] += sampleCount;
        }

      }

      //Scale the accumulted flux by the sample counts
      for (size_t i = 0, yy = 0; yy < m_imgSize.y; ++yy) {
        for (size_t xx = 0; xx < m_imgSize.x; ++xx, ++i) {
          if (sampleCounts[i] == 0.0) {
            accum[i] = Spectrum(0.f);
          } else {
            accum[i] /= (Float) sampleCounts[i];
          }
        }
      }
      delete[] sampleCounts;


      return ReconstructionOptions{.needGlobalRescale = false,
              .res = accumBuffer};
    }

};


MTS_NAMESPACE_END

#endif //MITSUBA_SOLVER_COVARIATE_H
