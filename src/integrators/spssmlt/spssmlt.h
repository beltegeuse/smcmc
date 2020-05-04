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

#if !defined(__PSSMLT_H)
#define __PSSMLT_H

#include "../blockthread.h"
#include <mitsuba/bidir/pathsampler.h>
#include <mitsuba/core/bitmap.h>
#include <mitsuba/render/scene.h>

/// Use Kelemen-style mutations in random number space?
#define KELEMEN_STYLE_MUTATIONS 1

MTS_NAMESPACE_BEGIN

/* ==================================================================== */
/*                         Configuration storage                        */
/* ==================================================================== */

enum EAlignAlgo {
    ENoAlign,
    EReference,
    ECovariateLog,
    ECovariateLogVal,
    EIterativeReweight,
};

enum EPosition {
    ECurr = 0,
    EYN = 1,
    EXP = 2,
    EYP = 3,
    EXN = 4
};


enum ERenderingAlgo {
    EClassical,
    ERE,
    EAggressiveRE,
    EGDNormalization,
};

enum ETargetFunction {
    ELuminance,
    EMax,
    EMaxAll,
    ESum,
    EGD,
};

enum EGlobalAlignOption {
    EDefault,
    EForceToAlign,
    EForceToNotAlign,
};

enum EInitAlgo {
    EInitBruteForce,
    EInitMCMC,
    EInitNaive,
};

/**
 * \brief Stores all configuration parameters used by
 * the MLT rendering implementation
 */
struct SPSSMLTConfiguration {
    // Sampling configuration
    int minDepth;
    int maxDepth;
    int rrDepth;

    // The global factor deduction
    bool refNormalization;
    Float luminance;
    int luminanceSamples;

    // The number of samples and nb of workunits
    int workUnits;

    // Options about the mutation
    Float pLarge;
    Float mutationSizeLow;
    Float mutationSizeHigh;

    // Compute & Use previous results
    bool noMIS;
    bool hideEmitter;
    int block; // Size of the block
    ERenderingAlgo MCMC;

    ETargetFunction imp;
    EInitAlgo initAlgo;
    EAlignAlgo alignAlgo;
    EGlobalAlignOption alignGlobally;

    // For reference
    std::string reference;

    //For cache
    bool useTileCache;
    bool writeToTileCache;
    std::string cacheDirectory;
    std::string cacheBaseFileName;
    std::string CACHE_FILE_EXTENSION;

    // Others
    bool strictNormals;
    Float shiftThreshold;

    // Current parameters
    bool useInitialization;
    std::string referencePath;
    int SPPInit;

    //Force reconstruction parameters
    int iterations;
    bool noShift;
    int randomSeed;
    bool dumpingStats;
    bool amcmc;
    Float aStar;
    int REFrequency;
    bool fluxPDF;
    Float percentageInit;
    int nbMCMC;
    Float alpha;
    bool useUniformInit;
    bool volume;

    // Reconstruction
    bool errorWeights;
    bool customWeights;

    inline SPSSMLTConfiguration() = default;

    void dump() const {
      SLog(EDebug, "SPSSMLT configuration:");
      SLog(EDebug, "   Maximum path depth          : %i", maxDepth);
      SLog(EDebug, "   Russian roulette depth      : %i", rrDepth);
      SLog(EDebug, "   Large step probability      : %f", pLarge);
      SLog(EDebug, "   Mutation size               : [%f, %f]",
           mutationSizeLow, mutationSizeHigh);
      SLog(EDebug, "   Overall MLT image luminance : %f (%i samples)",
           luminance, luminanceSamples);
      SLog(EDebug, "   Total number of work units  : %i", workUnits);
    }

    inline explicit SPSSMLTConfiguration(Stream *stream) {
      SLog(EError, "No serialization");
    }

    inline void serialize(Stream *stream) const {
      SLog(EError, "No serialization");
    }

    void parse(const Properties &props) {
        customWeights = props.getBoolean("customWeights", false);
        errorWeights = props.getBoolean("errorWeights", true);

      noShift = props.getBoolean("noShift", false);
      useInitialization = props.getBoolean("useInitialization", false); // DO NOT
      referencePath = props.getString("referencePath", "");
      randomSeed = props.getInteger("randomSeed", -1);
      dumpingStats = !props.getBoolean("useCache", true);
      if(props.getBoolean("dumpingStats", false)) {
        dumpingStats = true;
      }
      amcmc = props.getBoolean("amcmc", false);
      aStar = props.getFloat("aStar", 0.5);
      REFrequency = props.getInteger("REFrequency", 1);
      fluxPDF = props.getBoolean("fluxPDF", true);
      percentageInit = props.getFloat("percentageInit", 0.3);
      nbMCMC = props.getInteger("nbMCMC", 0);
      alpha = props.getFloat("alpha", 0.05);
      useUniformInit = props.getBoolean("useUniformInit", true);
      volume = props.getBoolean("volume", false);

      // For the ref normalization
      refNormalization = props.getBoolean("refNormalization", false);
      if(refNormalization) {
          luminance = props.getFloat("normalization");
      }

      /* Longest visualized path length (<tt>-1</tt>=infinite).
         A value of <tt>1</tt> will visualize only directly visible light
         sources. <tt>2</tt> will lead to single-bounce (direct-only)
         illumination, and so on. */
      maxDepth = props.getInteger("maxDepth", -1);
      minDepth = props.getInteger("minDepth", 0);

      /* Depth to begin using russian roulette (set to -1 to disable) */
      rrDepth = props.getInteger("rrDepth", 5);

      /* Number of samples used to estimate the total luminance
         received by the sensor's sensor */
      luminanceSamples = props.getInteger("luminanceSamples", 100000);

      /* Probability of creating large mutations in the [Kelemen et. al]
         MLT variant. The default is 0.3. */
      pLarge = props.getFloat("pLarge", 0.3f);

      /* This parameter can be used to specify the samples per pixel used to
         render the direct component. Should be a power of two (otherwise, it will
         be rounded to the next one). When set to zero or less, the
         direct illumination component will be hidden, which is useful
         for analyzing the component rendered by MLT. When set to -1,
         PSSMLT will handle direct illumination as well */
      hideEmitter = props.getBoolean("hideEmitter", true);

      /* Recommended mutation sizes in primary sample space */
      mutationSizeLow = props.getFloat("mutationSizeLow", 1.0f / 1024.0f);
      mutationSizeHigh = props.getFloat("mutationSizeHigh", 1.0f / 64.0f);

      SAssert(mutationSizeLow > 0 && mutationSizeHigh > 0 &&
              mutationSizeLow < 1 && mutationSizeHigh < 1 &&
              mutationSizeLow < mutationSizeHigh);

      /* Specifies the number of parallel work units required for
         multithreaded and network rendering. When set to <tt>-1</tt>, the
         amount will default to four times the number of cores. Note that
         every additional work unit entails a significant amount of
         communication overhead (a full-sized floating put image must be
         transmitted), hence it is important to set this value as low as
         possible, while ensuring that there are enough units to keep all
         workers busy. */
      workUnits = props.getInteger("workUnits", -1);

      initAlgo = [&]() -> EInitAlgo {
          std::string init = props.getString("init", "mcmc");
          std::transform(init.begin(), init.end(),
                         init.begin(), ::tolower);
          if (init == "brute") {
            return EInitBruteForce;
          } else if (init == "mcmc") {
            return EInitMCMC;
          } else if (init == "naive") {
            return EInitNaive;
          } else {
            SLog(EError, "Impossible to found a init algo for: %s", init.c_str());
          }
      }();
      SPPInit = props.getInteger("SPPInit", 0);

      alignGlobally = [&]() -> EGlobalAlignOption {
          std::string globalAlign = props.getString("globalAlign", "default");
          std::transform(globalAlign.begin(), globalAlign.end(),
                         globalAlign.begin(), ::tolower);
          if (globalAlign == "default") {
            return EDefault;
          } else if (globalAlign == "align") {
            return EGlobalAlignOption::EForceToAlign;
          } else if (globalAlign == "noalign") {
            return EGlobalAlignOption::EForceToNotAlign;
          } else {
            SLog(EError, "Impossible to found a global align option for: %s", globalAlign.c_str());
          }
      }();

      noMIS = props.getBoolean("noMIS", false);

      // Get reference image path
      reference = props.getString("reference", ""); // Only used if align proc. is "reference"

      //Directory of the cached tile data
      useTileCache = props.getBoolean("useCache", false);
      writeToTileCache = props.getBoolean("writeToCache",
                                          true);//If it makes sense to disable writing to cache this is available
      cacheDirectory = props.getString("cacheDirectory", "");//
      cacheBaseFileName = props.getString("cacheBaseFileName", "set_the_base_file_name_in_the_xml_file");//
      CACHE_FILE_EXTENSION = ".tiles";

      // Other
      strictNormals = false;
      shiftThreshold = Float(0.001); // FIXME: Default value for now

      // Importance function
      imp = [&]() -> ETargetFunction {
          auto imp = props.getString("imp", "max");
          std::transform(imp.begin(), imp.end(),
                         imp.begin(), ::tolower);
          if (imp == "luminance") {
            return ELuminance;
          } else if (imp == "max") {
            return EMax;
          } else if (imp == "maxall") {
            return EMaxAll;
          } else if (imp == "sum") {
            return ESum;
          } else if(imp == "gd") {
            return EGD;
          } else {
            SLog(EError, "Impossible to found correct target function");
          }
      }();
      block = props.getInteger("block", 32); // This parameter only is important for "re" in MCMC
      MCMC = [&]() -> ERenderingAlgo {
          std::string algo = props.getString("MCMC", "REHV"); // only valid values: "re" and "classical" or "REHV"
          std::transform(algo.begin(), algo.end(),
                         algo.begin(), ::tolower);
          if (algo == "re") {
            return ERE;
          } else if (algo == "classical") {
            return EClassical;
          } else if (algo == "rehv") {
            return EAggressiveRE;
          } else if (algo == "gdnorm") {
            return EGDNormalization;
          } else {
            SLog(EError, "Impossible to found match algo");
          }
      }();

      alignAlgo = [&]() -> EAlignAlgo {
          std::string algo = props.getString("alignAlgo", "noalign");
          std::transform(algo.begin(), algo.end(),
                         algo.begin(), ::tolower);
          if (algo == "noalign") {
            return ENoAlign;
          } else if (algo == "iterative") {
            return EIterativeReweight;
          } else if (algo == "covariate_gd_log") {
              return ECovariateLog;
          } else if(algo == "covariate_gd_log_val") {
              return ECovariateLogVal;
          } else if (algo == "reference") {
              return EReference;
          } else {
            SLog(EError, "Bad option for align algorithm: %s", algo.c_str());
          }
      }();

      //Handle force reconstruction properties
      iterations = props.getInteger("iterations", 50);
    }
};

MTS_NAMESPACE_END

#endif /* __PSSMLT_H */
