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

#include <mitsuba/render/scene.h>

MTS_NAMESPACE_BEGIN

/*! \plugin{direct}{Direct illumination integrator}
 * \order{1}
 * \parameters{
 *     \parameter{shadingSamples}{\Integer}{This convenience parameter can be
 *         used to set both \code{emitterSamples} and \code{bsdfSamples} at
 *         the same time.
 *     }
 *     \parameter{emitterSamples}{\Integer}{Optional more fine-grained
 *        parameter: specifies the number of samples that should be generated
 *        using the direct illumination strategies implemented by the scene's
 *        emitters\default{set to the value of \code{shadingSamples}}
 *     }
 *     \parameter{bsdfSamples}{\Integer}{Optional more fine-grained
 *        parameter: specifies the number of samples that should be generated
 *        using the BSDF sampling strategies implemented by the scene's
 *        surfaces\default{set to the value of \code{shadingSamples}}
 *     }
 *     \parameter{strictNormals}{\Boolean}{Be strict about potential
 *        inconsistencies involving shading normals? See
 *        page~\pageref{sec:strictnormals} for details.
 *        \default{no, i.e. \code{false}}
 *     }
 *     \parameter{hideEmitters}{\Boolean}{Hide directly visible emitters?
 *        See page~\pageref{sec:hideemitters} for details.
 *        \default{no, i.e. \code{false}}
 *     }
 * }
 * \vspace{-1mm}
 * \renderings{
 *     \medrendering{Only BSDF sampling}{integrator_direct_bsdf}
 *     \medrendering{Only emitter sampling}{integrator_direct_lum}
 *     \medrendering{BSDF and emitter sampling}{integrator_direct_both}
 *     \caption{
 *         \label{fig:integrator-direct}
 *         This plugin implements two different strategies for computing the
 *         direct illumination on surfaces. Both of them are dynamically
 *         combined then obtain a robust rendering algorithm.
 *     }
 * }
 *
 * This integrator implements a direct illumination technique that makes use
 * of \emph{multiple importance sampling}: for each pixel sample, the
 * integrator generates a user-specifiable number of BSDF and emitter
 * samples and combines them using the power heuristic. Usually, the BSDF
 * sampling technique works very well on glossy objects but does badly
 * everywhere else (\subfigref{integrator-direct}{a}), while the opposite
 * is true for the emitter sampling technique
 * (\subfigref{integrator-direct}{b}). By combining these approaches, one
 * can obtain a rendering technique that works well in both cases
 * (\subfigref{integrator-direct}{c}).
 *
 * The number of samples spent on either technique is configurable, hence
 * it is also possible to turn this plugin into an emitter sampling-only
 * or BSDF sampling-only integrator.
 *
 * For best results, combine the direct illumination integrator with the
 * low-discrepancy sample generator (\code{ldsampler}). Generally, the number
 * of pixel samples of the sample generator can be kept relatively
 * low (e.g. \code{sampleCount=4}), whereas the \code{shadingSamples}
 * parameter of this integrator should be increased until the variance in
 * the output renderings is acceptable.
 *
 * \remarks{
 *    \item This integrator does not handle participating media or
 *          indirect illumination.
 * }
 */

class MIDirectIntegrator : public SamplingIntegrator {
public:
	MIDirectIntegrator(const Properties &props) : SamplingIntegrator(props) {

	}

	/// Unserialize from a binary data stream
	MIDirectIntegrator(Stream *stream, InstanceManager *manager)
	 : SamplingIntegrator(stream, manager) {
		configure();
	}

	void serialize(Stream *stream, InstanceManager *manager) const {
		SamplingIntegrator::serialize(stream, manager);
	}

	void configure() {
		SamplingIntegrator::configure();
	}

	void configureSampler(const Scene *scene, Sampler *sampler) {
		SamplingIntegrator::configureSampler(scene, sampler);
	}

	Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
		/* Some aliases and local variables */
		const Scene *scene = rRec.scene;
		Intersection &its = rRec.its;
		RayDifferential ray(r);
		Spectrum Li(0.0f);
		Point2 sample;

		/* Perform the first ray intersection (or ignore if the
		   intersection has already been provided). */
		if (!rRec.rayIntersect(ray)) {
			return Spectrum(0.0f);
		} else {
			Float values[3];
			values[0] = 0.5*its.shFrame.t.x+0.5;
			values[1] = 0.5*its.shFrame.t.y+0.5;
			values[2] = 0.5*its.shFrame.t.z+0.5;
			return Spectrum(values);
		}
	}

	inline Float miWeight(Float pdfA, Float pdfB) const {
		pdfA *= pdfA; pdfB *= pdfB;
		return pdfA / (pdfA + pdfB);
	}

	std::string toString() const {
		std::ostringstream oss;
		oss << "MIDirectIntegrator[" << endl
			<< "]";
		return oss.str();
	}

	MTS_DECLARE_CLASS()
private:
};

MTS_IMPLEMENT_CLASS_S(MIDirectIntegrator, false, SamplingIntegrator)
MTS_EXPORT_PLUGIN(MIDirectIntegrator, "Direct illumination integrator");
MTS_NAMESPACE_END
