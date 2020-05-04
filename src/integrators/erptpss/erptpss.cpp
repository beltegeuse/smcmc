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
#include <mitsuba/core/statistics.h>
#include "erptpss_sampler.h"
#include "../blockthread.h"

MTS_NAMESPACE_BEGIN

static StatsCounter statsAccepted("Energy redistribution path tracing",
                                  "Accepted mutations", EPercentage);
static StatsCounter statsChainsPerPixel("Energy redistribution path tracing",
                                        "Chains started per pixel", EAverage);

/*!\plugin{erptpss}{Energy Redistribution Primary Sample Space Metropolis Light Transport}
 */

class ERPTPSS : public Integrator {
public:
    ERPTPSS(const Properties &props) : Integrator(props) {
        /* Development parameter. */
        m_config.useKelemenMutation = props.getBoolean("useKelemenMutation", true);

        /* Longest visualized path length (<tt>-1</tt>=infinite).
           A value of <tt>1</tt> will visualize only directly visible light
           sources. <tt>2</tt> will lead to single-bounce (direct-only)
           illumination, and so on. */
        m_config.maxDepth = props.getInteger("maxDepth", -1);

        /* Depth to begin using russian roulette (set to -1 to disable) */
        m_config.rrDepth = props.getInteger("rrDepth", 5);

        /* If set to <tt>true</tt>, the MLT algorithm runs on top of a
           bidirectional path tracer with multiple importance sampling.
           Otherwise, the implementation reverts to a basic path tracer.
           Generally, the bidirectinal path tracer should be noticably
           better, so it's best to this setting at its default. */
        m_config.technique = props.getBoolean("bidirectional", false) ?
                             PathSampler::EBidirectional : PathSampler::EUnidirectional;

        /* When running two-stage MLT, this parameter determines the size
           of the downsampled image created in the first pass (i.e. setting this
           to 16 means that the horizontal/vertical resolution will be 16 times
           lower). Usually, it's fine to leave this parameter unchanged. When
           the two-stage process introduces noisy halos around very bright image
           regions, it can be set to a lower value */
        m_config.firstStageSizeReduction = props.getInteger(
                "firstStageSizeReduction", 16);

        /* Used internally to let the nested rendering process of a
           two-stage MLT approach know that it is running the first stage */
        m_config.firstStage = props.getBoolean("firstStage", false);

        /* Number of samples used to estimate the total luminance
           received by the sensor's sensor */
        m_config.refNormalization = props.getBoolean("refNormalization", false);
        if(m_config.refNormalization) {
            m_config.luminance = props.getFloat("normalization");
        }
        m_config.luminanceSamples = props.getInteger("luminanceSamples", 100000);

        /* Probability of creating large mutations in the [Kelemen et. al]
           MLT variant. The default is 0.3. */
        m_config.pLarge = props.getFloat("pLarge", 0.3f);

        /* This parameter can be used to specify the samples per pixel used to
           render the direct component. Should be a power of two (otherwise, it will
           be rounded to the next one). When set to zero or less, the
           direct illumination component will be hidden, which is useful
           for analyzing the component rendered by MLT. When set to -1,
           PSSMLT will handle direct illumination as well */
        m_config.directSamples = props.getInteger("directSamples", 16);
        m_config.separateDirect = m_config.directSamples >= 0;

        /* Should an optimized direct illumination sampling strategy be used
           for s=1 paths? (as opposed to plain emission sampling). Usually
           a good idea. Note that this setting only applies when the
           bidirectional path tracer is used internally. The optimization
           affects all paths, not just the ones contributing direct illumination,
           hence it is completely unrelated to the <tt>separateDirect</tt>
           parameter. */
        m_config.directSampling = props.getBoolean(
                "directSampling", true);

        /* Recommended mutation sizes in primary sample space */
        m_config.mutationSizeLow = props.getFloat("mutationSizeLow", 1.0f / 1024.0f);
        m_config.mutationSizeHigh = props.getFloat("mutationSizeHigh", 1.0f / 64.0f);
        Assert(m_config.mutationSizeLow > 0 && m_config.mutationSizeHigh > 0 &&
               m_config.mutationSizeLow < 1 && m_config.mutationSizeHigh < 1 &&
               m_config.mutationSizeLow < m_config.mutationSizeHigh);

        /* Stop MLT after X seconds -- useful for equal-time comparisons */
        m_config.timeout = props.getInteger("timeout", 0);

        // If we want to correct the pdf of sampling emitters
        fluxPDF = props.getBoolean("fluxPDF", true);

        /* Special parameters for ERPT */
        m_config.maxChains = props.getInteger("maxChains", 0);
        m_config.chainLength = props.getInteger("chainLength", 100);
        m_config.numChains = props.getInteger("numChains", 1);
        m_config.blockSize = props.getInteger("blockSize", 0);
    }

    /// Unserialize from a binary data stream
    ERPTPSS(Stream *stream, InstanceManager *manager)
            : Integrator(stream, manager) {
        m_config = ERPTPSSConfiguration(stream);
        configure();
    }

    virtual ~ERPTPSS() {}

    void serialize(Stream *stream, InstanceManager *manager) const {
        Integrator::serialize(stream, manager);
        m_config.serialize(stream);
    }

    bool preprocess(const Scene *scene, RenderQueue *queue,
                    const RenderJob *job, int sceneResID, int sensorResID,
                    int samplerResID) {
        Integrator::preprocess(scene, queue, job, sceneResID,
                               sensorResID, samplerResID);
        if (fluxPDF) {
            const_cast<Scene *>(scene)->weightEmitterFlux();
        }
        ref<const Sensor> sensor = scene->getSensor();
        if (scene->getSubsurfaceIntegrators().size() > 0)
            Log(EError, "Subsurface integrators are not supported by MLT!");

        if (sensor->getSampler()->getClass()->getName() != "IndependentSampler")
            Log(EError, "Metropolis light transport requires the independent sampler");

        m_mltSampler = new ERPTPSSSampler(m_config);
        m_mltSampler->setSampleCount(sensor->getSampler()->getSampleCount());

        return true;
    }

    void cancel() {
        ref<RenderJob> nested = m_nestedJob;
        Scheduler::getInstance()->cancel(m_process);
    }

    bool render(Scene *scene, RenderQueue *queue, const RenderJob *job,
                int sceneResID, int sensorResID, int samplerResID) {
        ref<Scheduler> scheduler = Scheduler::getInstance();
        ref<Sensor> sensor = scene->getSensor();
        ref<Sampler> sampler = sensor->getSampler();
        Film *film = sensor->getFilm();
        size_t sampleCount = sampler->getSampleCount();
        size_t nCores = scheduler->getCoreCount();

        // Get number of samples approximate sample count
        if(sampleCount % nCores != 0) {
            sampleCount = ((sampleCount / nCores) + 1)*nCores;
            SLog(EWarn, "Rescale the samples count to: "
            SIZE_T_FMT, sampleCount);
        }
        size_t sampleCountPerCore = sampleCount / nCores;

        // This code do not support importance maps.
        m_config.importanceMap = NULL;
        ref<PathSampler> pathSampler = new PathSampler(false, m_config.technique, scene,
                                                       sampler, sampler, sampler, m_config.maxDepth,
                                                       m_config.rrDepth,
                                                       m_config.separateDirect, m_config.directSampling);
        Vector2i cropSize = film->getCropSize();
        Assert(cropSize.x > 0 && cropSize.y > 0);
        Log(EInfo, "Starting render job (%ix%i, "
                SIZE_T_FMT
                " %s, "
                SSE_STR
                ", approx. "
                SIZE_T_FMT
                " mutations/pixel) ..", cropSize.x, cropSize.y,
            nCores, nCores == 1 ? "core" : "cores", sampleCount);

        ref<Bitmap> directImage;
        if (m_config.separateDirect && m_config.directSamples > 0) {
            directImage = BidirectionalUtils::renderDirectComponent(scene,
                                                                    sceneResID, sensorResID, queue, job,
                                                                    m_config.directSamples);
            if (directImage == NULL)
                return false;
        }

        // Compute the luminance only if the reference luminance is not provided
        m_config.luminance = pathSampler->computeAverageLuminance(m_config.luminanceSamples);

        // Multi-thread computations
        // We need to create the data used to for parallel processing loop
        ref<Bitmap> result = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat, cropSize);
        result->clear();
        struct ThreadData {
            ref<Sampler> sampler;
            ref<Bitmap> image;
        };
        std::vector<ThreadData> threadData;
        for(auto i = 0; i < nCores; i++) {
            threadData.emplace_back(ThreadData {
                m_mltSampler->clone(),
                result->clone(),
            });
        }

        // The parallel strategy is to split the number of samples
        // per cores. Meaning that all cores will render a full image.
        // These images are combined at the end to produce the final rendering.
        BlockScheduler runPool(nCores, nCores, 1);
        runPool.run([&](int tileID, int threadID) {
            ThreadData& data = threadData[threadID];
            auto mltsampler = (ERPTPSSSampler*) data.sampler.get();
            ref<PathSampler> localpathsampler = new PathSampler(false, PathSampler::EUnidirectional, scene,
                                                                data.sampler, data.sampler, data.sampler, m_config.maxDepth,
                                                                m_config.rrDepth,
                                                                m_config.separateDirect, m_config.directSampling);

            // Accumlate lambda
            auto accumulate = [&](const Point2& pixel, const Spectrum& value) {
                Spectrum *throughputPix = (Spectrum *) data.image->getData();
                size_t curr_pix = ((int)pixel.y) * cropSize.x + ((int)pixel.x);
                throughputPix[curr_pix] += value;
            };

            // Lambda that applies ERPT update rule. This function will be called
            // inside PathSampler::samplePathsPT
            auto erpt_callback = [&](const Point2& samplePos, Spectrum contrib) {
                auto weight = contrib.getLuminance();
                if (std::isnan(weight) || std::isinf(weight) || weight < 0)
                    Log(EWarn, "Invalid path weight: %f, ignoring path!", weight);

                auto random = mltsampler->getRandom();

                // Compute the expected number of chain we want to launch
                Float meanChains = m_config.numChains * weight
                                   / (m_config.luminance * mltsampler->getSampleCount());

                /* Optional: do not launch too many chains if this is desired by the user */
                if (m_config.maxChains > 0 && meanChains > m_config.maxChains)
                    meanChains = std::min(meanChains, (Float) m_config.maxChains);

                /* Decide the actual number of chains that will be launched, as well
                   as their deposition energy */
                int numChains = (int) std::floor(random->nextFloat() + meanChains);
                if (numChains == 0) {
                    mltsampler->accept();
                    return;
                }

                // Need to accept to make that the sampler
                // have the PSS representation of the path before mutating it
                mltsampler->accept();

                Float depositionEnergy = weight / (sampleCount
                                                   * meanChains * m_config.chainLength);
                size_t mutations = 0;

                // This is similar to "Reversible jump":
                // This procedure corrects the random number used in image space
                // So the chain can mutate and explore the complete image
                mltsampler->correctSensorRandom(*sensor, samplePos);

                // Save the state to run several chains
                auto stateSampler = mltsampler->getState();
                mltsampler->setLargeStep(false);

                // The number of chains that we want to start
                for (int chain = 0; chain < numChains; ++chain) {
                    Point2 currentPosition = samplePos;
                    Float accumulatedWeight = 0;
                    ++statsChainsPerPixel;

                    // As we start a chain, we will
                    mltsampler->setState(stateSampler);

                    for (size_t it = 0; it < m_config.chainLength; ++it) {
                        /* Sample a mutated path */
                        SplatList proposed;
                        localpathsampler->sampleSplats(Point2i(-1), proposed);
                        Float proposed_weight = proposed.splats[0].second.getLuminance();

                        Float a = std::min((Float) 1.0f, proposed_weight / weight);
                        accumulatedWeight += 1 - a;
                        /* Accept with probability 'a' */
                        if (a == 1 || random->nextFloat() < a) {
                            if (accumulatedWeight > 0) {
                                Spectrum value = (contrib / weight) * (accumulatedWeight * depositionEnergy);
                                accumulate(currentPosition, value);
                            }

                            /* The mutation was accepted */
                            // --- Update the internal values of the current state
                            contrib = proposed.splats[0].second;
                            currentPosition = proposed.splats[0].first;
                            weight = proposed_weight;
                            // --- Accept the move
                            mltsampler->accept();
                            accumulatedWeight = a;
                            // --- Update statistics
                            ++statsAccepted;
                            ++mutations;
                        } else {
                            if (a > 0) {
                                Spectrum value = (proposed.splats[0].second / proposed_weight) * (a * depositionEnergy);
                                accumulate(proposed.splats[0].first, value);
                            }
                            mltsampler->reject();
                        }
                        statsAccepted.incrementBase();
                    }

                    // Flush the remaining energy
                    if (accumulatedWeight > 0) {
                        Spectrum value = (contrib / weight) * (accumulatedWeight * depositionEnergy);
                        accumulate(currentPosition, value);
                    }
                }
            };

            // Each thread render a full image
            // Force to use independent sampling here.
            // each time a path is generated, the erpt_callback is called.
            for(auto x = 0; x < cropSize.x; x++) {
                for(auto y = 0; y < cropSize.y; y++) {
                    Point2i offset = Point2i(x, y);
                    for(auto s = 0; s < sampleCountPerCore; s++) {
                        mltsampler->setLargeStep(true); // Uniform sampling
                        localpathsampler->samplePathsPT(offset, erpt_callback);
                        mltsampler->advance();
                    }
                }
            }
        });

        // Merge all images togather
        for(auto res: threadData) {
            result->accumulate(res.image);
        }
        film->setBitmap(result);

        return true;
    }

    MTS_DECLARE_CLASS()
private:
    ref<ParallelProcess> m_process;
    ref<RenderJob> m_nestedJob;
    ref<ERPTPSSSampler> m_mltSampler;
    ERPTPSSConfiguration m_config;
    bool fluxPDF;
};

MTS_IMPLEMENT_CLASS_S(ERPTPSS, false, Integrator)

MTS_EXPORT_PLUGIN(ERPTPSS, "ERPT + Primary Sample Space MLT");
MTS_NAMESPACE_END
