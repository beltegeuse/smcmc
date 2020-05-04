#ifndef MITSUBA_SOLVER_H
#define MITSUBA_SOLVER_H

#include <iostream>

#include <mitsuba/render/scene.h>
#include <mitsuba/core/fstream.h>

#include "../spssmlt.h"
#include "../tile.h"

MTS_NAMESPACE_BEGIN

// Structure to return the reconstructed image
struct ReconstructionOptions {
  bool needGlobalRescale; //< If the image needs a global scaling
  ref<Bitmap> res;
};

class TileSolver
{
protected:
    const SPSSMLTConfiguration &m_config;
    const Vector2i m_imgSize;
    std::vector<AbstractTile*> &m_tiles;
public:
    ref<Bitmap> tileIntoImage(std::vector<AbstractTile *> &tiles,
                              const Vector2i &cropSize,
                              bool norm, bool splatCenter) {
        // Do the splatting
        ref<Bitmap> accumBuffer = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat, cropSize);
        accumBuffer->clear();
        Spectrum *accum = (Spectrum *) accumBuffer->getData();
        int *sampleCounts = new int[cropSize.x * cropSize.y]();

        //Push each of the tiles to the buffer
        for (AbstractTile *tile : tiles) {
            Float curr_norm = norm ? tile->getNorm() : 1.0;
            int sampleCount = norm ? tile->getNormSamples() : 1;

            if(splatCenter) {
                Spectrum normalizedValue = tile->get(0) * curr_norm;
                Point2i pixelLocation = tile->pixel(0);
                int bufferIndex = pixelLocation.y * cropSize.x + pixelLocation.x;
                accum[bufferIndex] += normalizedValue * sampleCount;
                sampleCounts[bufferIndex] += sampleCount;
            } else {
                for (size_t pixelIndex = 0; pixelIndex < tile->size(); ++pixelIndex) {
                    Spectrum normalizedValue = tile->get(pixelIndex) * curr_norm;

                    Point2i pixelLocation = tile->pixel(pixelIndex);
                    if (pixelLocation.x < 0 || pixelLocation.x >= cropSize.x || pixelLocation.y < 0
                        || pixelLocation.y >= cropSize.y)
                        continue;

                    int bufferIndex = pixelLocation.y * cropSize.x + pixelLocation.x;
                    accum[bufferIndex] += normalizedValue * sampleCount;
                    sampleCounts[bufferIndex] += sampleCount;
                }
            }
        }

        //Scale the accumulted flux by the sample counts
        for (size_t i = 0, yy = 0; yy < cropSize.y; ++yy)
            for (size_t xx = 0; xx < cropSize.x; ++xx, ++i) {
                if(sampleCounts[i] == 0.0) {
                    accum[i] = Spectrum(0.f);
                } else {
                    accum[i] /= (Float) sampleCounts[i];
                }
            }
        delete[] sampleCounts;

        return accumBuffer;
    }

    // Classical constructor
    TileSolver(const SPSSMLTConfiguration &config, const Vector2i &imgSize, std::vector<AbstractTile*> &tiles) :
            m_config(config), m_imgSize(imgSize), m_tiles(tiles) {}

    virtual ~TileSolver() {}

    // Solve the system using an approach
    virtual ReconstructionOptions solve() = 0;

protected:
  std::vector<Float> compute_throughput(bool lum = true) {
      std::vector<Float> throughput(m_imgSize.x * m_imgSize.y, Float(0.f));
      std::vector<int> nbSplat(m_imgSize.x * m_imgSize.y, int(0));
      auto tile_size = m_tiles[0]->size();
      for (size_t y = 0; y < m_imgSize.y; y++) {
          for (size_t x = 0; x < m_imgSize.x; x++) {
              size_t k = y * m_imgSize.x + x;

              Float scale = m_tiles[k]->getNorm();
              throughput[k] += scale * (lum ? m_tiles[k]->lum(0) : 1.0);
              nbSplat[k] += 1;
              if (tile_size == 3) {
                  if (x < m_imgSize.x - 1) {
                      throughput[k + 1] +=  scale * (lum ? m_tiles[k]->lum(1) : 1.0);
                      nbSplat[k + 1] += 1;
                  }
                  if(y < m_imgSize.y - 1) {
                      throughput[k + m_imgSize.x] +=  scale * (lum ? m_tiles[k]->lum(2) : 1.0);
                      nbSplat[k + m_imgSize.x] += 1;
                  }
              } else if(tile_size == 5) {
                  if (x < m_imgSize.x - 1) {
                      throughput[k + 1] +=  scale * (lum ? m_tiles[k]->lum(2) : 1.0);
                      nbSplat[k + 1] += 1;
                  }
                  if(y < m_imgSize.y - 1) {
                      throughput[k + m_imgSize.x] +=  scale * (lum ? m_tiles[k]->lum(3) : 1.0);
                      nbSplat[k + m_imgSize.x] += 1;
                  }
                  if (x > 0) {
                      throughput[k - 1] += scale * (lum ? m_tiles[k]->lum(4) : 1.0);
                      nbSplat[k - 1] += 1;
                  }
                  if(y > 0) {
                      throughput[k - m_imgSize.x] += scale * (lum ? m_tiles[k]->lum(1) : 1.0);
                      nbSplat[k - m_imgSize.x] += 1;
                  }
              } else {
                  SLog(EError, "No support to other tiles shapes");
              }
          }
      }

      for(size_t k = 0; k < m_imgSize.x*m_imgSize.y; k++) {
          throughput[k] /= nbSplat[k];
      }

      return throughput;
  }
};

// Tools for init solvers
inline Float sum(std::vector<AbstractTile*> &tildes, const Vector2i &imgSize,
                 int iStart, int iEnd, int jStart, int jEnd,
                 std::function<Float(AbstractTile &)> f, Float w = 1.f) {
    Float s = 0.0;
    for (int i = iStart; i < iEnd; i++) {
        for (int j = jStart; j < jEnd; j++) {
            s += f(getTilde(tildes, imgSize, Point2i(i, j))) * w;
        }
    }
    return s;
}

inline Float sumH(std::vector<AbstractTile*> &tildes, const Vector2i &imgSize,
                  int iStart, int iEnd, int j, std::function<Float(AbstractTile &)> f) {
    return sum(tildes, imgSize, iStart, iEnd, j, j + 1, f);
}

inline Float sumV(std::vector<AbstractTile*> &tildes, const Vector2i &imgSize,
                  int i, int jStart, int jEnd, std::function<Float(AbstractTile &)> f) {
    return sum(tildes, imgSize, i, i + 1, jStart, jEnd, f);
}


MTS_NAMESPACE_END

#endif //MITSUBA_CONJUGATE_H