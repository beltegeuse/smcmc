#include "rendering.h"

// Force to use coherent (local) coordinate system
// This is based on the Pixar's paper.
// Note that it is important to keep the coordinate system coherent
// to keep random number replay performance high.
#define COHERENT_FRAME 1

MTS_NAMESPACE_BEGIN

//////////////////////////////////
// Classical Path tracing
/////////////////////////////////
Spectrum PathTracingTilde::L(const TPoint2<int> &pos,
                                     SPSSMLTSampler *sampler, bool hideEmitter) const {
    // Trace a ray from the camera
    // TODO: Make the sampling of these values
    TPoint2<Float> apertureSample(0.5f);
    double timeSample = 0.5f;

    // Generate the ray from the pixelPos
    sampler->force_uniform = true;
    TPoint2<Float> samplePos = sampler->next2D();
    sampler->force_uniform = false;
    samplePos.x += pos.x;
    samplePos.y += pos.y;
    RayDifferential ray;
    Spectrum value = m_scene->getSensor()->sampleRayDifferential(ray, samplePos, apertureSample, timeSample);
    ray.hasDifferentials = false; // BDPT issue

    // Check the first intersection
    // The validity of it will be checked at the beginning of the loop
    Intersection its;
    m_scene->rayIntersect(ray, its);

    // TODO: have removed hide emitter and stric normals
    // Initialize the variable for the path tracing
    Spectrum Li(0.0f);
    Spectrum throughput(1.0f);
    double eta = 1.0f;
    int depth = 1;
    bool scattered = false;

    while (depth <= m_config.maxDepth || m_config.maxDepth < 0) {
        if (!its.isValid()) {
            if ((!hideEmitter && scattered)) {
                // Just evaluate the sky
                Li += throughput * m_scene->evalEnvironment(ray);
            }
            //TODO: Fix me: not scattering support.
            break;
        }

        /* Possibly include emitted radiance if requested */
        if (its.isEmitter() && (!hideEmitter && scattered))
            Li += throughput * its.Le(-ray.d);

        /* Include radiance from a subsurface scattering model if requested */
        if (its.hasSubsurface()) {
            SLog(EError, "No support of subsurface scattering");
            //Li += throughput * its.LoSub(m_scene, sampler, -ray.d, depth);
        }
        if ((depth >= m_config.maxDepth && m_config.maxDepth > 0)) {
            /* Only continue if:
               1. The current path length is below the specifed maximum
               2. If 'strictNormals'=true, when the geometric and shading
                  normals classify the incident direction to the same side */
            break;
        }

        /* ==================================================================== */
        /*                     Direct illumination sampling                     */
        /* ==================================================================== */

        const BSDF *bsdf = its.getBSDF(ray);
#if COHERENT_FRAME
        // Change to coherent frame
    its.wi = its.toWorld(its.wi);
    {
      its.shFrame = Frame(its.shFrame.n);
      its.wi = its.toLocal(its.wi);
    }
#endif

        /* Estimate the direct illumination if this is requested */
        DirectSamplingRecord dRec(its);

        // Do only if it is not a specular BSDF
        if (bsdf->getType() & BSDF::ESmooth) {
            Spectrum value = m_scene->sampleEmitterDirect(dRec, sampler->next2D());
            if (!value.isZero()) {
                const Emitter *emitter = static_cast<const Emitter *>(dRec.object);

                /* Allocate a record for querying the BSDF */
                BSDFSamplingRecord bRec(its, its.toLocal(dRec.d), ERadiance);

                /* Evaluate BSDF * cos(theta) */
                const Spectrum bsdfVal = bsdf->eval(bRec);

                /* Prevent light leaks due to the use of shading normals */
                if (!bsdfVal.isZero()) {

                    /* Calculate prob. of having generated that direction
                       using BSDF sampling */
                    double bsdfPdf = (emitter->isOnSurface() && dRec.measure == ESolidAngle)
                                     ? bsdf->pdf(bRec) : 0;

                    /* Weight using the power heuristic */
                    double weight = miWeight(dRec.pdf, bsdfPdf);
                    Li += throughput * value * bsdfVal * weight;
                }
            }
        }

        /* ==================================================================== */
        /*                            BSDF sampling                             */
        /* ==================================================================== */

        /* Sample BSDF * cos(theta) */
        double bsdfPdf;
        BSDFSamplingRecord bRec(its, sampler, ERadiance);
        Spectrum bsdfWeight = bsdf->sample(bRec, bsdfPdf, sampler->next2D());
        if (bsdfWeight.isZero())
            break;

        scattered |= bRec.sampledType != BSDF::ENull;

        /* Prevent light leaks due to the use of shading normals */
        const Vector wo = its.toWorld(bRec.wo);

        bool hitEmitter = false;
        Spectrum value(0.f);

        /* Trace a ray in this direction */
        ray = Ray(its.p, wo, ray.time);
        if (m_scene->rayIntersect(ray, its)) {
            /* Intersected something - check if it was a luminaire */
            if (its.isEmitter()) {
                value = its.Le(-ray.d);
                dRec.setQuery(ray, its);
                hitEmitter = true;
            }
        } else {
            /* Intersected nothing -- perhaps there is an environment map? */
            const Emitter *env = m_scene->getEnvironmentEmitter();

            if (env) {
                if (hideEmitter && !scattered)
                    break;

                value = env->evalEnvironment(ray);
                if (!env->fillDirectSamplingRecord(dRec, ray))
                    break;
                hitEmitter = true;
            } else {
                break;
            }
        }

        /* Keep track of the throughput and relative
           refractive index along the path */
        throughput *= bsdfWeight;
        eta *= bRec.eta;

        /* If a luminaire was hit, estimate the local illumination and
           weight using the power heuristic */
        if (hitEmitter) {
            /* Compute the prob. of generating that direction using the
               implemented direct illumination sampling technique */
            const double lumPdf = (!(bRec.sampledType & BSDF::EDelta)) ?
                                  m_scene->pdfEmitterDirect(dRec) : 0;
            Li += throughput * value * miWeight(bsdfPdf, lumPdf);
        }

        /* ==================================================================== */
        /*                         Indirect illumination                        */
        /* ==================================================================== */
        if (!its.isValid()) {
            break;
        }

        if (depth++ >= m_config.rrDepth) {
            /* Russian roulette: try to keep path weights equal to one,
               while accounting for the solid angle compression at refractive
               index boundaries. Stop with at least some probability to avoid
               getting stuck (e.g. due to total internal reflection) */

            double q = std::min(throughput.max() * eta * eta, (double) 0.95f);
            if (sampler->next1D() >= q)
                break;
            throughput /= q;
        }
    }

    return Li;
}

/* Some aliases and local variables */
Spectrum PathTracingTilde::LVol(const TPoint2<int> &pos, const Medium* medium,
                                SPSSMLTSampler *sampler, bool hideEmitter) const {
    MediumSamplingRecord mRec;
    Spectrum Li(0.0f);
    Float eta = 1.0f;

    // Trace a ray from the camera
    // TODO: Make the sampling of these values
    TPoint2<Float> apertureSample(0.5f);
    double timeSample = 0.5f;

    // Generate the ray from the pixelPos
    sampler->force_uniform = true;
    TPoint2<Float> samplePos = sampler->next2D();
    sampler->force_uniform = false;
    samplePos.x += pos.x;
    samplePos.y += pos.y;
    RayDifferential ray;
    Spectrum value = m_scene->getSensor()->sampleRayDifferential(ray, samplePos, apertureSample, timeSample);
    ray.hasDifferentials = false; // BDPT issue

    // Check the first intersection
    // The validity of it will be checked at the beginning of the loop
    Intersection its;
    m_scene->rayIntersect(ray, its);

    Spectrum throughput(1.0f);
    bool scattered = false;
    int depth = 1;

    while (depth <= m_config.maxDepth || m_config.maxDepth < 0) {
        /* ==================================================================== */
        /*                 Radiative Transfer Equation sampling                 */
        /* ==================================================================== */
        if (medium && medium->sampleDistance(Ray(ray, 0, its.t), mRec, sampler)) {
            /* Sample the integral
               \int_x^y tau(x, x') [ \sigma_s \int_{S^2} \rho(\omega,\omega') L(x,\omega') d\omega' ] dx'
            */
            const PhaseFunction *phase = mRec.getPhaseFunction();

            if (depth >= m_config.maxDepth  && m_config.maxDepth  != -1) // No more scattering events allowed
                break;

            throughput *= mRec.sigmaS * mRec.transmittance / mRec.pdfSuccess;

            /* ==================================================================== */
            /*                          Luminaire sampling                          */
            /* ==================================================================== */

            /* Estimate the single scattering component if this is requested */
            DirectSamplingRecord dRec(mRec.p, mRec.time);

            if (true) {
                int interactions = m_config.maxDepth  - depth - 1;

                Spectrum value = m_scene->sampleAttenuatedEmitterDirect(
                        dRec, medium, interactions,
                        sampler->next2D(), sampler);

                if (!value.isZero()) {
                    const Emitter *emitter = static_cast<const Emitter *>(dRec.object);

                    /* Evaluate the phase function */
                    PhaseFunctionSamplingRecord pRec(mRec, -ray.d, dRec.d);
                    Float phaseVal = phase->eval(pRec);

                    if (phaseVal != 0) {
                        /* Calculate prob. of having sampled that direction using
                           phase function sampling */
                        Float phasePdf = (emitter->isOnSurface() && dRec.measure == ESolidAngle)
                                         ? phase->pdf(pRec) : (Float) 0.0f;

                        /* Weight using the power heuristic */
                        const Float weight = miWeight(dRec.pdf, phasePdf);
                        Li += throughput * value * phaseVal * weight;
                    }
                }
            }

/* ==================================================================== */
/*                         Phase function sampling                      */
/* ==================================================================== */

            Float phasePdf;
            PhaseFunctionSamplingRecord pRec(mRec, -ray.d);
            Float phaseVal = phase->sample(pRec, phasePdf, sampler);
            if (phaseVal == 0)
                break;
            throughput *= phaseVal;

            /* Trace a ray in this direction */
            ray = Ray(mRec.p, pRec.wo, ray.time);
            ray.mint = 0;

            Spectrum value(0.0f);
            rayIntersectAndLookForEmitter(m_scene.get(), sampler, medium,
                                          m_config.maxDepth  - depth - 1, ray, its, dRec, value);

            /* If a luminaire was hit, estimate the local illumination and
           weight using the power heuristic */
            if (!value.isZero() && (/*rRec.type & RadianceQueryRecord::EDirectMediumRadiance*/ true)) {
                const Float emitterPdf = m_scene->pdfEmitterDirect(dRec);
                Li += throughput * value * miWeight(phasePdf, emitterPdf);
            }

            /* ==================================================================== */
            /*                         Multiple scattering                          */
            /* ==================================================================== */

            /* Stop if multiple scattering was not requested */
            //      if (!(rRec.type & RadianceQueryRecord::EIndirectMediumRadiance))
            //        break;
            //rRec.type = RadianceQueryRecord::ERadianceNoEmission;
        } else {
            /* Sample
              tau(x, y) (Surface integral). This happens with probability mRec.pdfFailure
              Account for this and multiply by the proper per-color-channel transmittance.
            */
            if (medium)
                throughput *= mRec.transmittance / mRec.pdfFailure;

            if (!its.isValid()) {
                /* If no intersection could be found, possibly return
                   attenuated radiance from a background luminaire */
                if (//true // rRec.type & RadianceQueryRecord::EEmittedRadiance &&
                        (!hideEmitter || scattered)) {
                    Spectrum value = throughput * m_scene->evalEnvironment(ray);
                    if (medium)
                        value *= medium->evalTransmittance(ray, sampler);
                    Li += value;
                }

                break;
            }

            /* Possibly include emitted radiance if requested */
            if (its.isEmitter() // && (rRec.type & RadianceQueryRecord::EEmittedRadiance)
                && (!hideEmitter || scattered))
                Li += throughput * its.Le(-ray.d);

            /* Include radiance from a subsurface integrator if requested */
            if (its.hasSubsurface()) //&& (rRec.type & RadianceQueryRecord::ESubsurfaceRadiance))
                Li += throughput * its.LoSub(m_scene.get(), sampler, -ray.d, depth);

            if (depth >= m_config.maxDepth && m_config.maxDepth != -1)
                break;

            /* Prevent light leaks due to the use of shading normals */
            Float wiDotGeoN = -dot(its.geoFrame.n, ray.d),
                    wiDotShN = Frame::cosTheta(its.wi);
            //      if (wiDotGeoN * wiDotShN < 0 && m_strictNormals)
            //        break;

            /* ==================================================================== */
            /*                          Luminaire sampling                          */
            /* ==================================================================== */

            const BSDF *bsdf = its.getBSDF(ray);
            DirectSamplingRecord dRec(its);

            /* Estimate the direct illumination if this is requested */
            if (//(rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance) &&
                    (bsdf->getType() & BSDF::ESmooth)) {
                int interactions = m_config.maxDepth - depth - 1;

                Spectrum value = m_scene->sampleAttenuatedEmitterDirect(
                        dRec, its, medium, interactions,
                        sampler->next2D(), sampler);

                if (!value.isZero()) {
                    const Emitter *emitter = static_cast<const Emitter *>(dRec.object);

                    /* Evaluate BSDF * cos(theta) */
                    BSDFSamplingRecord bRec(its, its.toLocal(dRec.d));
                    const Spectrum bsdfVal = bsdf->eval(bRec);

                    Float woDotGeoN = dot(its.geoFrame.n, dRec.d);

                    /* Prevent light leaks due to the use of shading normals */
                    if (!bsdfVal.isZero()) { // !m_strictNormals || woDotGeoN * Frame::cosTheta(bRec.wo) > 0)
                        /* Calculate prob. of having generated that direction
                           using BSDF sampling */
                        Float bsdfPdf = (emitter->isOnSurface()
                                         && dRec.measure == ESolidAngle)
                                        ? bsdf->pdf(bRec) : (Float) 0.0f;

                        /* Weight using the power heuristic */
                        const Float weight = miWeight(dRec.pdf, bsdfPdf);
                        Li += throughput * value * bsdfVal * weight;
                    }
                }
            }


            /* ==================================================================== */
            /*                            BSDF sampling                             */
            /* ==================================================================== */

            /* Sample BSDF * cos(theta) */
            BSDFSamplingRecord bRec(its, sampler, ERadiance);
            Float bsdfPdf;
            Spectrum bsdfWeight = bsdf->sample(bRec, bsdfPdf, sampler->next2D());
            if (bsdfWeight.isZero())
                break;

            /* Prevent light leaks due to the use of shading normals */
            const Vector wo = its.toWorld(bRec.wo);
            Float woDotGeoN = dot(its.geoFrame.n, wo);
//      if (woDotGeoN * Frame::cosTheta(bRec.wo) <= 0 && m_strictNormals)
//        break;

            /* Trace a ray in this direction */
            ray = Ray(its.p, wo, ray.time);

            /* Keep track of the throughput, medium, and relative
               refractive index along the path */
            throughput *= bsdfWeight;
            eta *= bRec.eta;
            if (its.isMediumTransition())
                medium = its.getTargetMedium(ray.d);

            /* Handle index-matched medium transitions specially */
            if (bRec.sampledType == BSDF::ENull) {
//        if (!(rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance))
//          break;
//        rRec.type = scattered ? RadianceQueryRecord::ERadianceNoEmission
//                              : RadianceQueryRecord::ERadiance;
                m_scene->rayIntersect(ray, its);
                depth++;
                continue;
            }

            Spectrum value(0.0f);
            rayIntersectAndLookForEmitter(m_scene.get(), sampler, medium,
                                          m_config.maxDepth - depth - 1, ray, its, dRec, value);

            /* If a luminaire was hit, estimate the local illumination and
               weight using the power heuristic */
            if (!value.isZero()) { // (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance)
                const Float emitterPdf = (!(bRec.sampledType & BSDF::EDelta)) ?
                                         m_scene->pdfEmitterDirect(dRec) : 0;
                Li += throughput * value * miWeight(bsdfPdf, emitterPdf);
            }

            /* ==================================================================== */
            /*                         Indirect illumination                        */
            /* ==================================================================== */

            /* Stop if indirect illumination was not requested */
//      if (!(rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance))
//        break;
//
//      rRec.type = RadianceQueryRecord::ERadianceNoEmission;
        }

        depth++;

//    if (rRec.depth++ >= m_rrDepth) {
///* Russian roulette: try to keep path weights equal to one,
//   while accounting for the solid angle compression at refractive
//   index boundaries. Stop with at least some probability to avoid
//   getting stuck (e.g. due to total internal reflection) */
//
//      Float q = std::min(throughput.max() * eta * eta, (Float) 0.95f);
//      if (rRec.nextSample1D() >= q)
//        break;
//      throughput /= q;
//    }

        scattered = true;
    }
//  avgPathLength.incrementBase();
//  avgPathLength += rRec.depth;
    return Li;
}

/**
 * This function is called by the recursive ray tracing above after
 * having sampled a direction from a BSDF/phase function. Due to the
 * way in which this integrator deals with index-matched boundaries,
 * it is necessarily a bit complicated (though the improved performance
 * easily pays for the extra effort).
 *
 * This function
 *
 * 1. Intersects 'ray' against the scene geometry and returns the
 *    *first* intersection via the '_its' argument.
 *
 * 2. It checks whether the intersected shape was an emitter, or if
 *    the ray intersects nothing and there is an environment emitter.
 *    In this case, it returns the attenuated emittance, as well as
 *    a DirectSamplingRecord that can be used to query the hypothetical
 *    sampling density at the emitter.
 *
 * 3. If current shape is an index-matched medium transition, the
 *    integrator keeps on looking on whether a light source eventually
 *    follows after a potential chain of index-matched medium transitions,
 *    while respecting the specified 'maxDepth' limits. It then returns
 *    the attenuated emittance of this light source, while accounting for
 *    all attenuation that occurs on the wya.
 */
void PathTracingTilde::rayIntersectAndLookForEmitter(const Scene *scene, Sampler *sampler,
                                                     const Medium *medium, int maxInteractions, Ray ray, Intersection &_its,
                                                     DirectSamplingRecord &dRec, Spectrum &value) const {
    Intersection its2, *its = &_its;
    Spectrum transmittance(1.0f);
    bool surface = false;
    int interactions = 0;

    while (true) {
        surface = scene->rayIntersect(ray, *its);

        if (medium)
            transmittance *= medium->evalTransmittance(Ray(ray, 0, its->t), sampler);

        if (surface && (interactions == maxInteractions ||
                        !(its->getBSDF()->getType() & BSDF::ENull) ||
                        its->isEmitter())) {
            /* Encountered an occluder / light source */
            break;
        }

        if (!surface)
            break;

        if (transmittance.isZero())
            return;

        if (its->isMediumTransition())
            medium = its->getTargetMedium(ray.d);

        Vector wo = its->shFrame.toLocal(ray.d);
        BSDFSamplingRecord bRec(*its, -wo, wo, ERadiance);
        bRec.typeMask = BSDF::ENull;
        transmittance *= its->getBSDF()->eval(bRec, EDiscrete);

        ray.o = ray(its->t);
        ray.mint = Epsilon;
        its = &its2;

        if (++interactions > 100) { /// Just a precaution..
            SLog(EWarn, "rayIntersectAndLookForEmitter(): round-off error issues?");
            return;
        }
    }

    if (surface) {
        /* Intersected something - check if it was a luminaire */
        if (its->isEmitter()) {
            dRec.setQuery(ray, *its);
            value = transmittance * its->Le(-ray.d);
        }
    } else {
        /* Intersected nothing -- perhaps there is an environment map? */
        const Emitter *env = scene->getEnvironmentEmitter();

        if (env && env->fillDirectSamplingRecord(dRec, ray))
            value = transmittance * env->evalEnvironment(RayDifferential(ray));
    }
}

MTS_IMPLEMENT_CLASS(RenderingTechniqueTilde, false, Object)
MTS_NAMESPACE_END