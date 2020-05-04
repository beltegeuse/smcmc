#ifndef MITSUBA_SOLVER_REF_H
#define MITSUBA_SOLVER_REF_H

#include "../../reconstruction.h"
#include "solver.h"

#include <mitsuba/render/scene.h>
#include <mitsuba/core/fstream.h>

MTS_NAMESPACE_BEGIN

/**
 * This solver use a reference image to compute the normalization factor
 * This solver is only useful to debug other algorithm step.
 */
#define SUM_POS 0
class TileCovariateRef : public TileSolver {
    ref<Bitmap> m_ref;
public:
    TileCovariateRef(const SPSSMLTConfiguration &config, const Vector2i &imgSize, std::vector<AbstractTile *> &tiles)
            : TileSolver(config, imgSize, tiles) {
        // Load the refernce
        m_ref = new Bitmap(m_config.reference);
    }

    ReconstructionOptions solve() override {
      auto tile_size = m_tiles[0]->size();
      SAssert(tile_size == 5);

      // Compute the normalization factor from the reference
      std::vector<Float> b(m_imgSize.x * m_imgSize.y, 0.f);
      for (size_t k = 0; k < m_imgSize.x * m_imgSize.y; k++) {
        AbstractTile *tile = m_tiles[k];

        auto sum_scale = 0.0;
        auto nb_scales = 0;
        for (size_t pixelIndex = 0; pixelIndex < tile->size(); ++pixelIndex) {
          Spectrum pixelValue = tile->pixels[pixelIndex].values;

          Point2i pixelLocation = tile->pixel(pixelIndex);
          if (pixelLocation.x < 0 || pixelLocation.x >= m_imgSize.x || pixelLocation.y < 0
              || pixelLocation.y >= m_imgSize.y)
              continue;

          auto bufferIndex = pixelLocation.y * m_imgSize.x + pixelLocation.x;
          auto ref_lum = m_ref->getPixel(pixelLocation).getLuminance();
          if(pixelValue.getLuminance()  != 0.0) {
              sum_scale += ref_lum / pixelValue.getLuminance();
              nb_scales += 1;
          }
        }

        if(nb_scales != 0) {
            b[k] = sum_scale / nb_scales;
        }
      }


      ref<Bitmap> accumBuffer = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat, m_imgSize);
      accumBuffer->clear();
      Spectrum *accum = (Spectrum *) accumBuffer->getData();
      int *sampleCounts = new int[m_imgSize.x * m_imgSize.y]();
      // Push each of the tiles to the buffer
      for (size_t k = 0; k < m_imgSize.x * m_imgSize.y; k++) {
        AbstractTile *tile = m_tiles[k];
        int sampleCount = tile->nbSamples;
        Float rgbScale[3] = {b[k], b[k], b[k]};

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


      return ReconstructionOptions{.needGlobalRescale = false,.res = accumBuffer};
    }

};


MTS_NAMESPACE_END

#endif
