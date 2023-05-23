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

#include <mitsuba/bidir/util.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/core/statistics.h>
#include <fstream>  

#include "spssmlt.h"
#include "tile.h"
#include "rendering.h"

#include "solver/solver_covariate_grad_log_val.h"
#include "solver/solver_covariate_grad_log.h"
#include "solver/solver_iterative_reweighted.h"
#include "solver/solver_covariate_reference.h"

#include "algo/algo.h"
#include "algo/algo_re.h"
#include "algo/algo_classic.h"

MTS_NAMESPACE_BEGIN

/*!\plugin{pssmlt}{Stratified Primary Sample Space Metropolis Light Transport}
 */

class SPSSMLT : public Integrator {
public:
    explicit SPSSMLT(const Properties &props) : Integrator(props) {
      m_config.parse(props);
      m_running = false;
    }

    /// Unserialize from a binary data stream
    SPSSMLT(Stream *stream, InstanceManager *manager)
            : Integrator(stream, manager) {
      m_config = SPSSMLTConfiguration(stream);
      configure();
      m_running = false;
    }

    ~SPSSMLT() override {}

    void serialize(Stream *stream, InstanceManager *manager) const override {
      Integrator::serialize(stream, manager);
      m_config.serialize(stream);
    }

    bool preprocess(const Scene *scene, RenderQueue *queue,
                    const RenderJob *job, int sceneResID, int sensorResID,
                    int samplerResID) override {
      Integrator::preprocess(scene, queue, job, sceneResID,
                             sensorResID, samplerResID);

      if (m_config.fluxPDF) {
        const_cast<Scene *>(scene)->weightEmitterFlux();
      }

      ref<const Sensor> sensor = scene->getSensor();

      if (!scene->getSubsurfaceIntegrators().empty())
        Log(EError, "Subsurface integrators are not supported by MLT!");

      if (sensor->getSampler()->getClass()->getName() != "IndependentSampler")
        Log(EError, "Metropolis light transport requires the independent sampler");

      return true;
    }

    void cancel() override {
      Scheduler::getInstance()->cancel(m_process);
      m_running = false;
    }

    bool render(Scene *scene,
                RenderQueue *queue,
                const RenderJob *job,
                int sceneResID,
                int sensorResID,
                int samplerResID) override {
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
      Thread::initializeOpenMP(nCores);

      /************************************************************************
       * Consts
       ************************************************************************/
      //Base filepath for tile cache files
      std::string tileCacheBaseFileName = scene->getDestinationFile().filename().string();
      std::string xmlBasePath = scene->getSourceFile().parent_path().string();

      // Creation of timer file
      std::string timeFilename = scene->getDestinationFile().string()
                                 + "_time.csv";
      std::ofstream timeFile(timeFilename.c_str(), std::ofstream::out);
      ref<Timer> renderingTimer = new Timer;

      // Information about the rendering size
      Vector2i cropSize = film->getCropSize();
      Assert(cropSize.x > 0 && cropSize.y > 0);
      Log(EInfo, "Starting render job (%ix%i, "
              SIZE_T_FMT
              " %s, "
              SSE_STR
              ", approx. "
              SIZE_T_FMT
              " mutations/pixel) ..",
          cropSize.x, cropSize.y,
          nCores, nCores == 1 ? "core" : "cores", sampleCount);

      // Deduce the number of mutation and workunits
      size_t desiredMutationsPerWorkUnit = 200000;
      if (m_config.workUnits <= 0) {
        const size_t cropArea = (size_t) cropSize.x * cropSize.y;
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
       * Select the good rendering algorithm
       ************************************************************************/
      std::unique_ptr<Rendering> renderingAlgo = std::unique_ptr<Rendering>([&]() -> Rendering * {
          if (m_config.MCMC == EClassical || m_config.MCMC == ERE) {
            return new RenderingClassical(m_config, scene, cropSize);
          } else if (m_config.MCMC == EAggressiveRE) {
            return new RenderingAggressiveRE(m_config, scene, cropSize);
          } else {
            SLog(EError, "Impossible to found constructor for rendering algo");
            return nullptr;
          }
      }());

      /************************************************************************
       * Precomputation for tile shape initialization
       ************************************************************************/
      std::vector<AbstractTile *> tildes;
      renderingAlgo->initTiles(tildes);

      /************************************************************************
       * Initialize Tiles and states
       ************************************************************************/
      if (!m_config.useTileCache)
        renderingAlgo->initializeChains(tildes, 0);

      // Create this sampler before to change the random number
      ref<ReplayableSampler> rplSampler = new ReplayableSampler();
      int iteration = 1;

      /************************************************************************
       * Render Loop
       ************************************************************************/
      // Launch the rendering process
      while (m_running) {

        if (!m_config.useTileCache) {
          ///////////// Step 2
          // Compute the global normalization factor
          if(!m_config.refNormalization) {
              ref<PathSampler> pathSampler = new PathSampler(m_config.volume,
                                                             PathSampler::EUnidirectional, scene,
                                                             rplSampler, rplSampler, rplSampler,
                                                             m_config.maxDepth, m_config.rrDepth,
                                                             false,
                                                             true,
                                                             false, // Light image (BDPT)
                                                             m_config.hideEmitter);

              // Compute normalization factor
              Float luminance = pathSampler->computeAverageLuminance(luminanceSamples);

              // Average the global normalization factor over the iterations
              m_config.luminance = (luminance + (iteration - 1) * m_config.luminance) / iteration;
              Log(EInfo, "Luminance global value: %f", m_config.luminance);
          }

          /************************************************************************
           * Create the initlal task set
           ************************************************************************/
          // Create the tasks (std::vector<Tilde*>)
          // For now just make one tasks at a time
          std::vector<std::vector<AbstractTile *>> estimatedTasks = renderingAlgo->generateTasks(iteration, tildes);

          /************************************************************************
           * Tile Main Loop (Render)
           ************************************************************************/
          // Launch the rendering process
          {

            // FIXME: Add the states somewhere which will be more easy
            // FIXME: and it will avoid multiple initialization
            std::vector<std::vector<AbstractTile *>> tasks;
            for (std::vector<AbstractTile *> &group : estimatedTasks)
              if (!group.empty())
                tasks.push_back(group);
            if (iteration != 1) {
              renderingAlgo->initializeChains(tildes, iteration);
            }
            Log(EInfo, "Do the rendering of tiles : %i x %i", tasks.size(), sampleCount);
            renderingAlgo->compute(tildes, std::move(tasks), sampleCount);
          }


          /************************************************************************
           * Write the time values
           ************************************************************************/
          renderingAlgo->dumpInfo(scene, film, iteration);

          /// Time it
          unsigned int milliseconds = renderingTimer->getMilliseconds();
          timeFile << (milliseconds / 1000.f) << ",\n";
          timeFile.flush();
          Log(EInfo, "Rendering time: %i, %i", milliseconds / 1000,
              milliseconds % 1000);

          /************************************************************************
           * Scale and Align the tiles
           ************************************************************************/
          // Print out non aligned output
          if (m_config.writeToTileCache) {
            if (m_config.cacheBaseFileName != "") {
              // Replace the name of the file by overwrite it.
              tileCacheBaseFileName = m_config.cacheBaseFileName;
            }
            //Write out the tiles
            //The filename is of the form /path/xmlconfig.xml_tile_iteration#.tiles
            std::string cacheFile = createTileCacheFilePath(xmlBasePath, tileCacheBaseFileName, iteration);
            Log(EInfo, "Start to write a cache file: %s", cacheFile.c_str());

            ref<FileStream> tileStream = new FileStream(cacheFile, FileStream::EFileMode::ETruncWrite);
            serializeTiles(tileStream, tildes);
            tileStream->flush();
            tileStream->close();
          }

        } else {//End if(!useTileCache)

          //Load tile cache file into the current set of tiles
          //Cosntruct the file name based on the iteration and cacheDirectory and cacheBaseFilePath parameters
          std::string cacheFile = createTileCacheFilePath(xmlBasePath, m_config.cacheBaseFileName, iteration);
          ref<FileStream> tileStream = new FileStream(cacheFile);
          bool deserializationSucceeded = deserializeTiles(tileStream, tildes);
          tileStream->close();

          if (!deserializationSucceeded) {
            SLog(EError, "Tile deserialization failed, canceling job.");
            m_running = false;
            break;
          }
        }

        {
          int minNbSamples = 1000000;
          int maxNbSamples = 0;
          Float avgNbSamples = 0;
          for (auto *tile : tildes) {
            avgNbSamples += tile->nbSamples;
            minNbSamples = std::min(minNbSamples, tile->nbSamples);
            maxNbSamples = std::max(maxNbSamples, tile->nbSamples);
            tile->resetScale();
            tile->scaleNbSamples();
          }
          avgNbSamples /= tildes.size();
          SLog(EInfo, "NbSamples statistics: [%i, %i], avg: %f", minNbSamples, maxNbSamples, avgNbSamples);
          ref<Bitmap> accumBuffer = tileIntoImage(tildes, cropSize, true, false);
          develop(scene, film, accumBuffer, iteration, "_noAlign_");
        }

        // Do the reconstruction
        auto recons_options = renderingAlgo->reconstruct(tildes, sampleCount * iteration);
        {
          ref<Bitmap> accumBuffer = recons_options.res;
          auto do_align = [&]() -> bool {
              if (m_config.alignGlobally == EDefault) {
                return recons_options.needGlobalRescale;
              } else if (m_config.alignGlobally == EForceToNotAlign) {
                return false;
              } else if (m_config.alignGlobally == EForceToAlign) {
                return true;
              }
          }();
          if (do_align) {
            scaleGlobally(accumBuffer->getPixelCount(), (Spectrum *) accumBuffer->getData());
          }
          develop(scene, film, accumBuffer, iteration, "_");
          queue->signalRefresh(job); // Display the last computed image

          // For debugging
          if (m_config.dumpingStats) {
            accumBuffer = dataIntoImage(tildes, cropSize, [cropSize](AbstractTile *tile) -> std::tuple<Spectrum, int> {
                Point2i pixelLocation = tile->pixel(0);
                int bufferIndex = pixelLocation.y * cropSize.x + pixelLocation.x;
                auto normInv = (Float) tile->nbSamples;
                Float rgb[3] = {tile->scale, normInv * tile->scale, tile->getNorm()};
                return std::make_tuple(Spectrum(rgb), bufferIndex);
            });
            develop(scene, film, accumBuffer, iteration, "_scale_");

            accumBuffer = dataIntoImage(tildes, cropSize, [cropSize](AbstractTile *tile) -> std::tuple<Spectrum, int> {
                Point2i pixelLocation = tile->pixel(0);
                int bufferIndex = pixelLocation.y * cropSize.x + pixelLocation.x;
                Float rgb[3] = {tile->REAttempt == 0 ? 0.0 : tile->REAcc / (Float) tile->REAttempt,
                                tile->nbSmallMut == 0 ? 0.0 : tile->nbSmallMutAcc / (Float) tile->nbSmallMut,
                                tile->OriInit ? 1.0 : 0.0};
                return std::make_tuple(Spectrum(rgb), bufferIndex);
            });
            develop(scene, film, accumBuffer, iteration, "_sampling_");


            accumBuffer = tileIntoImageMC(tildes, cropSize, false);
            develop(scene, film, accumBuffer, iteration, "_MCEstimates_");
          }
        }
        /************************************************************************
         * Reset Timer
         ************************************************************************/
        renderingTimer->reset();

        // Go to the next iteration
        iteration += 1;
      }
#if 0
      // All constant
      for(int i = 0; i < cropSize.x; i++) {
          for(int j = 0; j < cropSize.y; j++) {
              Tilde& t = getTilde(tildes, cropSize, Point2i(i,j));
              int id = getIMpos(cropSize, Point2i(i,j)) + 1;
              t.values[0] = Spectrum(id);
              t.values[1] = Spectrum(id);
              t.values[2] = Spectrum(id);
          }
      }
#endif
      return m_running;
    }

    std::string createTileCacheFilePath(std::string basePath, std::string baseFileName, int iteration) {
      std::string cacheFile =
              basePath + "/" + m_config.cacheDirectory + baseFileName + "_" + std::to_string(iteration) +
              m_config.CACHE_FILE_EXTENSION;
      return cacheFile;
    }

    void serializeTiles(FileStream *stream, std::vector<AbstractTile *> &sourceVector) {
      //For all tiles in the vector, serialize them to the stream
      stream->writeFloat(m_config.luminance);
      stream->writeLong(sourceVector.size());

      SLog(EInfo, "Serialize tiles");
      for (AbstractTile *tile : sourceVector) {
        tile->serialize(stream);
      }
    }

    bool deserializeTiles(FileStream *stream, std::vector<AbstractTile *> &destinationVector) {
      //Expects the vector to be pre-populated with the exact number of tiles required
      //  Using cache values from a run with different tile settings than the current config is not supported

      //Read in the tile count
      //Verify that it matches the current size
      //For the count, deserialize into the tiles
      //Return true if succeeded, false otherwise
      m_config.luminance = stream->readFloat();
      long cacheLen = stream->readLong();
      if (cacheLen != destinationVector.size()) {
        std::cerr << "ERROR: Encountered tile count missmatch while deserializing:  cacheSize[" << cacheLen
                  << "] != activeSize[" << destinationVector.size() << "]." << std::endl;
        std::cerr << "Please verify that the current settings match the settings used to generate the cache.";
        return false;
      }

      for (int i = 0; i < cacheLen; ++i) {
        destinationVector[i]->deserialize(stream);
      }
      return true;
    }

    ref<Bitmap> dataIntoImage(std::vector<AbstractTile *> &tiles, const Vector2i &cropSize,
                              std::function<std::tuple<Spectrum, int>(AbstractTile *)> func) {
      // Do the splatting
      ref<Bitmap> accumBuffer = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat, cropSize);
      accumBuffer->clear();
      Spectrum *accum = (Spectrum *) accumBuffer->getData();

      //Push each of the tiles to the buffer
      for (AbstractTile *tile : tiles) {
        auto res = func(tile);
        accum[std::get<1>(res)] = std::get<0>(res);
      }

      return accumBuffer;
    }

    ref<Bitmap> tileIntoImageMC(std::vector<AbstractTile *> &tiles,
                              const Vector2i &cropSize,
                              bool splatCenter) {
      // Do the splatting
      ref<Bitmap> accumBuffer = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat, cropSize);
      accumBuffer->clear();
      Spectrum *accum = (Spectrum *) accumBuffer->getData();
      int *sampleCounts = new int[cropSize.x * cropSize.y]();

      //Push each of the tiles to the buffer
      for (AbstractTile *tile : tiles) {
        if (splatCenter) {
          Spectrum normalizedValue = tile->pixels[0].values_MC ;
          Point2i pixelLocation = tile->pixel(0);
          int bufferIndex = pixelLocation.y * cropSize.x + pixelLocation.x;
          accum[bufferIndex] += normalizedValue;
          sampleCounts[bufferIndex] += tile->nbSamplesUni;
        } else {
          for (size_t pixelIndex = 0; pixelIndex < tile->size(); ++pixelIndex) {
            Point2i pixelLocation = tile->pixel(pixelIndex);
            if (pixelLocation.x < 0 || pixelLocation.x >= cropSize.x || pixelLocation.y < 0
                || pixelLocation.y >= cropSize.y)
              continue;

            int bufferIndex = pixelLocation.y * cropSize.x + pixelLocation.x;
            accum[bufferIndex] +=  tile->pixels[pixelIndex].values_MC ;
            sampleCounts[bufferIndex] += tile->nbSamplesUni;
          }
        }
      }

      //Scale the accumulted flux by the sample counts
      for (size_t i = 0, yy = 0; yy < cropSize.y; ++yy)
        for (size_t xx = 0; xx < cropSize.x; ++xx, ++i) {
          if (sampleCounts[i] == 0.0) {
            accum[i] = Spectrum(0.f);
          } else {
            accum[i] /= (Float) sampleCounts[i];
          }
        }
      delete[] sampleCounts;

      return accumBuffer;
    }

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

        if (splatCenter) {
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
          if (sampleCounts[i] == 0.0) {
            accum[i] = Spectrum(0.f);
          } else {
            accum[i] /= (Float) sampleCounts[i];
          }
        }
      delete[] sampleCounts;

      return accumBuffer;
    }

    void scaleGlobally(size_t pixelCount, Spectrum *accum) {
      /* Compute the luminance correction factor */
      double avgLuminance = 0;
      for (size_t i = 0; i < pixelCount; ++i) {
        Float currLum = accum[i].getLuminance();
        if (!std::isfinite(currLum)) {
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
    }

    void output(Scene *scene, Film *film, int currentIteration, const std::string &suffix,
                std::function<Spectrum(size_t)> &f) {
      auto cropSize = film->getCropSize();
      ref<Bitmap> bitmap = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat, cropSize);
      auto bitmap_ptr = (Spectrum *) bitmap->getData();
      for (size_t t = 0; t < cropSize.x * cropSize.y; t++) {
        bitmap_ptr[t] = f(t);
      }
      develop(scene, film, bitmap, currentIteration, suffix);
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
    SPSSMLTConfiguration m_config;
    bool m_running;
};

MTS_IMPLEMENT_CLASS_S(SPSSMLT, false, Integrator)

MTS_EXPORT_PLUGIN(SPSSMLT, "Primary Sample Space MLT");
MTS_NAMESPACE_END
