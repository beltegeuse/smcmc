// This file is part of the reference implementation for the paper 
//   Bayesian Collaborative Denoising for Monte-Carlo Rendering
//   Malik Boughida and Tamy Boubekeur.
//   Computer Graphics Forum (Proc. EGSR 2017), vol. 36, no. 4, p. 137-153, 2017
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE.txt file.

#ifndef SPIKE_REMOVAL_FILTER_H
#define SPIKE_REMOVAL_FILTER_H

#include <vector>

namespace bcd
{

	template<typename T> class DeepImage;

	class SpikeRemovalFilter
	{
	public:
		static void filter(DeepImage<float>& io_rInputColorImage,
				DeepImage<float>& io_rInputNbOfSamplesImage,
				DeepImage<float>& io_rInputHistogramImage,
				DeepImage<float>& io_rInputCovImage,
				float i_thresholdStDevFactor = 2.f);

	private:
		static void computeAverageAndStandardDeviation(
				float& o_rAverage,
				float& o_rStandardDeviation,
				const std::vector<float>& i_rData);

		/// @brief Simple and expensive (quadratic) implementation of multi-dimensional median as the minimizer of L1 distance among elements
		static int compute3DMedianIndex(
				const std::vector<float>& i_rDataR,
				const std::vector<float>& i_rDataG,
				const std::vector<float>& i_rDataB);



	};

} // namespace bcd


#endif
