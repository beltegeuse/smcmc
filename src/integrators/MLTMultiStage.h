#pragma once

#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/plugin.h>
MTS_NAMESPACE_BEGIN

enum EMLTStageTechniques {
      ENoStage = 0,
      ETwoStage = 1,
      EMultiStage = 2,
      EUnknowStage = 3
    };

    /**
     * Set the stagedTechnique attritube
     * to the good value using the technique name
     * If the technique is unknown, stagedTechnique will have EUnknowStage
     */
inline EMLTStageTechniques MLTStageTechnique(const std::string& stage) {
      if(stage == "NoStage") {
        return ENoStage;
      } else if(stage == "TwoStage") {
        return ETwoStage;
      } else if(stage == "MultiStage") {
        return EMultiStage;
      } else {
        return EUnknowStage;
      }
    }

inline std::string MLTStageTechniquesName(EMLTStageTechniques stagedTechnique) {
  if(stagedTechnique == ENoStage) {
    return "No Stage";
  } else if(stagedTechnique == ETwoStage) {
    return "Two Stage";
  } else if(stagedTechnique == EMultiStage) {
    return "Multi Stage";
  } else {
    return "Unknow";
  }
}

struct MLTAccumulBuffer {
  ref<Bitmap> prev;
  ref<Bitmap> accumulation;
  Float prevWeight;
  Float currWeight;

  double currAccum;
  size_t currAccumSamples;


  inline MLTAccumulBuffer(const Vector2i& size) {
    prev = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat, size);
    accumulation = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat, size);
    prevWeight = 0.f;
    currWeight = 0.f;

    currAccum = 0.0f;
    currAccumSamples = 0;
  }

  inline void addNormalization(Float b, size_t nbSamples) {
    SLog(EInfo, "Normalization factor update: (old: %f, current: %f)", getNormalization(),b);
    currAccum += b*nbSamples;
    currAccumSamples += nbSamples;
    SLog(EInfo, "Normalization factor update: (new: %f)", getNormalization());

    // Update prev image to have same level
    Float avgLuminance = 0;
    Spectrum *accum = (Spectrum *) prev->getData();
    size_t pixelCount = prev->getPixelCount();
    for (size_t i=0; i<pixelCount; ++i)
      avgLuminance += accum[i].getLuminance();
    avgLuminance /= (Float) pixelCount;

    Float luminanceFactor = getNormalization() / avgLuminance;
    SLog(EInfo, "Prev Exposition correction: %f", luminanceFactor);
    prev->scale(luminanceFactor);
  }


  Float getNormalization() const {
    if(currAccumSamples == 0) {
      return 1.f;
    }
    return currAccum / currAccumSamples;
  }
  /**
   * Copy the accumulation inside prev one
   * And add the weight into the currentWeight attribut
   */
  inline void next(Float passW) {
    prev->copyFrom(accumulation.get());
    prevWeight += passW;
  }

  /**
   * Combine a bitmap (img) with a certain weight (w)
   * with the previous pass.
   * Use the following formula:
   *  accum = prev*prevWeight + img*w*currWeight / (prevWeight+w*currWeight)
   */
  inline void add(ref<Bitmap> img, Float w) {
    if(currWeight == 0.f) {
      SLog(EError, "Impossible to add with 0 current weight");
    }

    accumulation->clear();
    if(prevWeight != 0.f) {
      // Add previous results and scale it
      accumulation->accumulate(prev.get());
      accumulation->scale(prevWeight);

      // Pre-scale current results
      img->scale(w*currWeight);
    }

    // Add new img (may be pre-scaled)
    accumulation->accumulate(img.get());
    if(prevWeight != 0.f) {
      // If we need to blend all results
      // Normalize accumulate results
      accumulation->scale(1.f / (prevWeight+w*currWeight));

      // Undo the scaling may in the img
      img->scale(1.f/(w*currWeight));
    }

    //SLog(EInfo, "Accumulate: %f x %f (%f) ", prevWeight, currWeight*w, w);
    // Recorrect image exposition:
    Float avgLuminance = 0;
    Spectrum *accum = (Spectrum *) accumulation->getData();
    size_t pixelCount = accumulation->getPixelCount();
    for (size_t i=0; i<pixelCount; ++i)
      avgLuminance += accum[i].getLuminance();
    avgLuminance /= (Float) pixelCount;

    Float luminanceFactor = getNormalization() / avgLuminance;
    SLog(EInfo, "Exposition correction: %f", luminanceFactor);
    accumulation->scale(luminanceFactor);

  }
};

// This idea of this class
// is to scale the number of samples
// to aim a target rendering time
struct MLTMultiStage {
  std::vector<int> sequence;
  std::vector<int> budget;

  // This will split the rendering using N passes
  // each passes will be timed and the budget will be
  // updated automatically.
  void setLinearCount(const Film* film, size_t sampleCount, int nbPasses) {
    sequence.clear();
    budget.clear();
    int xSize = film->getCropSize().x;
    for(int k = 0; k < nbPasses; k++) {
      sequence.push_back(xSize);
      budget.push_back(sampleCount/nbPasses);
    }
  }

  inline MLTMultiStage(const Film* film, size_t sampleCount) {
       int totalPass1 = 0;
       int totalPass2 = 0;
       int xSize = film->getCropSize().x;

       // Here the sequence of image grow by two each iteration
       // until reaching 50% of the image size
       for(int k = 1; k <= xSize / 2; k = k * 2) {
         sequence.push_back(k);
         budget.push_back(k);
         totalPass1 += k;
       }
       // Then the budget scale more smoothly until reaching the image size desired
       for(int k = xSize / 2; k <= xSize; k+= (xSize-xSize/2) / 10) {
         sequence.push_back(k);
         budget.push_back(k);
       }

       // Normalization the budget (regarding the number of samples)
       int totalBudget = 0;
       for(auto b: budget) {
           totalBudget += b;
       }
       Float factor = sampleCount/(Float)totalBudget;
       for(int& b: budget) {
           b = std::max((int)round(b*factor), 1);
       }

       // Show all budgets (For debugging proposes)
       SLog(EInfo, "Sequence with budget:");
       for(size_t i = 0; i < budget.size(); i++) {
         SLog(EInfo, " - %i : %i (%i)", i, sequence[i], budget[i]);
       }
  }

  /**
   * Return the total number of samples count
   * s is the scaling factor to scale the budget
   * i is the iteration number which doesn't need to be scale
   */
  inline size_t scaleBudget(Float s, int i) {
      int j = 0;
      size_t totalNewBudget = 0;

      // Sum all previous budget
      for(;j <= i; j++) {
          totalNewBudget += budget[j];
      }

      // Now scale it
      for(; j < (int)budget.size(); j++) {
          int newB = roundf((int) budget[j] * s);
          SLog(EInfo, " * iteration %i: Scale budget (old: %i, new: %i)", j, budget[j], newB);
          budget[j] = newB;
          totalNewBudget += newB;
      }

      return totalNewBudget;
  }

  inline ref<Bitmap> updateLuminanceMap(const Film* film, int i) {
      // TODO: This is dangerous because wrong gaussian size can over blur the TF image.
      Vector2i origCropSize   = film->getCropSize();
      float ratio = ((float)origCropSize.y) / origCropSize.x;
      Vector2i reducedCropSize = Vector2i(
          std::max(1, sequence[i]),
          std::max(1, (int)(ratio*sequence[i])));

      // Reconstruction filter used to downsample and upsample the different images
      ref<ReconstructionFilter> rfilter = static_cast<ReconstructionFilter *> (
          PluginManager::getInstance()->createObject(
          MTS_CLASS(ReconstructionFilter), Properties("gaussian")));
      rfilter->configure();

      // This is the previous results we got
      ref<Bitmap> luminanceMap = new Bitmap(Bitmap::ELuminance,
              Bitmap::EFloat, origCropSize);
      film->develop(Point2i(0, 0), origCropSize,
          Point2i(0, 0), luminanceMap);

      // Create smaller image
      SLog(EInfo, "Average down to %ix%i", reducedCropSize.x, reducedCropSize.y);
      luminanceMap = luminanceMap->resample(rfilter,
                                            ReconstructionFilter::EClamp,
                                            ReconstructionFilter::EClamp,
                                            reducedCropSize,
                                            0.0f, std::numeric_limits<Float>::infinity());

      // Interpolate
      SLog(EInfo, "Interpolate up");
      luminanceMap = luminanceMap->resample(rfilter,
              ReconstructionFilter::EClamp,
              ReconstructionFilter::EClamp, origCropSize,
              0.0f, std::numeric_limits<Float>::infinity());

      return luminanceMap;
  }

};


MTS_NAMESPACE_END
