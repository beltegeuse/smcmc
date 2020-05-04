// This file is part of the reference implementation for the paper 
//   Bayesian Collaborative Denoising for Monte-Carlo Rendering
//   Malik Boughida and Tamy Boubekeur.
//   Computer Graphics Forum (Proc. EGSR 2017), vol. 36, no. 4, p. 137-153, 2017
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE.txt file.

#include "MultiscaleDenoiser.h"

#include "Denoiser.h"

#include "DeepImage.h"

#include <sstream>

#include <iostream>

#include <cassert>

//#define SAVE_ADDITIONAL_OUTPUTS
#ifdef SAVE_ADDITIONAL_OUTPUTS
#define ADDITIONAL_OUTPUTS_PREFIX "outputs/multiscale/"
#endif

using namespace std;

namespace bcd
{

	bool MultiscaleDenoiser::denoise()
	{
		{
			Denoiser denoiser;
			denoiser.setInputs(m_inputs);
			denoiser.setParameters(m_parameters);
			denoiser.setOutputs(m_outputs);
			if(!denoiser.inputsOutputsAreOk())
				return false;
		}
		vector< unique_ptr< Deepimf > > additionalColorImages = generateDownscaledAverageImages(*m_inputs.m_pColors, m_nbOfScales - 1);
		vector< unique_ptr< Deepimf > > additionalNbOfSamplesImages = generateDownscaledSumImages(*m_inputs.m_pNbOfSamples, m_nbOfScales - 1);
		vector< unique_ptr< Deepimf > > additionalHistogramImages = generateDownscaledSumImages(*m_inputs.m_pHistograms, m_nbOfScales - 1);
		vector< unique_ptr< Deepimf > > additionalSampleCovarianceImages = generateDownscaledSampleCovarianceSumImages(
				*m_inputs.m_pSampleCovariances,
				*m_inputs.m_pNbOfSamples,
				additionalNbOfSamplesImages,
				m_nbOfScales - 1);

		vector< unique_ptr< Deepimf > > additionalOutputImages = generateDownscaledEmptyImages(*m_outputs.m_pDenoisedColors, m_nbOfScales - 1);

		unique_ptr< Deepimf > uScale0TmpImage = unique_ptr< Deepimf >(new Deepimf(*m_outputs.m_pDenoisedColors));
		vector< unique_ptr< Deepimf > > additionalTmpImages = generateDownscaledEmptyImages(*m_outputs.m_pDenoisedColors, m_nbOfScales - 1);

		vector< DenoiserInputs > inputsArray(m_nbOfScales);
		vector< DenoiserOutputs > outputsArray(m_nbOfScales);
		vector< Deepimf* > tmpImagesArray(m_nbOfScales);

		inputsArray[0] = m_inputs;
		outputsArray[0] = m_outputs;
		tmpImagesArray[0] = uScale0TmpImage.get();
		for(int scale = 1; scale < m_nbOfScales; ++scale)
		{
			inputsArray[scale].m_pColors = additionalColorImages[scale - 1].get();
			inputsArray[scale].m_pNbOfSamples = additionalNbOfSamplesImages[scale - 1].get();
			inputsArray[scale].m_pHistograms = additionalHistogramImages[scale - 1].get();
			inputsArray[scale].m_pSampleCovariances = additionalSampleCovarianceImages[scale - 1].get();
			outputsArray[scale].m_pDenoisedColors = additionalOutputImages[scale - 1].get();
			tmpImagesArray[scale] = additionalTmpImages[scale - 1].get();
#ifdef SAVE_ADDITIONAL_OUTPUTS
			{ // TEMPORARY
				ostringstream oss;
				oss << ADDITIONAL_OUTPUTS_PREFIX << "inputScale" << scale << ".exr";
				ImageIO::writeEXR(*inputsArray[scale].m_pColors, oss.str().c_str());
			}
#endif
		}

		{
			cout << "Computing scale " << m_nbOfScales - 1 << "..." << endl;
			Denoiser denoiser;
			denoiser.setInputs(inputsArray[m_nbOfScales - 1]);
			denoiser.setOutputs(outputsArray[m_nbOfScales - 1]);
			denoiser.setParameters(m_parameters);
			denoiser.setProgressCallback([this](float i_progress){ m_progressCallback(i_progress / float(((1 << (2 * m_nbOfScales)) - 1) / 3)); });
			denoiser.denoise();
#ifdef SAVE_ADDITIONAL_OUTPUTS
			{ // TEMPORARY
				ostringstream oss;
				oss << ADDITIONAL_OUTPUTS_PREFIX << "outputScale" << m_nbOfScales - 1 << ".exr";
				ImageIO::writeEXR(*outputsArray[m_nbOfScales - 1].m_pDenoisedColors, oss.str().c_str());
			}
#endif

		}
		for(int scale = m_nbOfScales - 2; scale >= 0; --scale)
		{
			cout << "Computing scale " << scale << "..." << endl;
			Denoiser denoiser;
			denoiser.setInputs(inputsArray[scale]);
			denoiser.setOutputs(outputsArray[scale]);
			denoiser.setParameters(m_parameters);
			denoiser.setProgressCallback([this, scale](float i_progress)
					{
						int s = m_nbOfScales - 1 - scale;
						// next higher definition scale is 4 times slower
						// 1 + 4 + 4^2 + ... 4^s = (4^(s+1) - 1) / (4 - 1) = (2^(2*(s+1)) - 1) / 3 = ((1 << (2*(s+1))) - 1) / 3
						float factor = 1.f / float(((1 << (2 * m_nbOfScales)) - 1) / 3);
						float minValue = factor * float(((1 << (2 * s)) - 1) / 3);
						float maxValue = factor * float(((1 << (2 * (s + 1))) - 1) / 3);
						m_progressCallback(minValue + i_progress * (maxValue - minValue));
					});
			denoiser.denoise();
#ifdef SAVE_ADDITIONAL_OUTPUTS
			{ // TEMPORARY
				ostringstream oss;
				oss << ADDITIONAL_OUTPUTS_PREFIX << "outputScale" << scale << ".exr";
				ImageIO::writeEXR(*outputsArray[scale].m_pDenoisedColors, oss.str().c_str());
			}
#endif
			mergeOutputs(
					*outputsArray[scale].m_pDenoisedColors,
					*tmpImagesArray[scale],
					*tmpImagesArray[scale + 1],
					*outputsArray[scale + 1].m_pDenoisedColors,
					*outputsArray[scale].m_pDenoisedColors);
#ifdef SAVE_ADDITIONAL_OUTPUTS
			{ // TEMPORARY
				ostringstream oss;
				oss << ADDITIONAL_OUTPUTS_PREFIX << "outputScale" << scale << "_merged.exr";
				ImageIO::writeEXR(*outputsArray[scale].m_pDenoisedColors, oss.str().c_str());
			}
#endif
		}
		return true;
	}
	vector< unique_ptr< Deepimf > > MultiscaleDenoiser::generateDownscaledEmptyImages(
			const Deepimf& i_rScale0Image,
			int i_nbOfScalesToGenerate)
	{
		vector< unique_ptr< Deepimf > > emptyImages(i_nbOfScalesToGenerate);

		int width = i_rScale0Image.getWidth();
		int height = i_rScale0Image.getHeight();
		const int depth = i_rScale0Image.getDepth();
		for(int scale = 0; scale < i_nbOfScalesToGenerate; ++scale)
		{
			width /= 2;
			height /= 2;
			emptyImages[scale] = unique_ptr< Deepimf >(new Deepimf(width, height, depth));
		}

		return emptyImages;
	}


	vector< unique_ptr< Deepimf > > MultiscaleDenoiser::generateDownscaledSumImages(
			const Deepimf& i_rScale0Image,
			int i_nbOfScalesToGenerate)
	{
		vector< unique_ptr< Deepimf > > downscaledImages(i_nbOfScalesToGenerate);
		const Deepimf* pPreviousImage = &i_rScale0Image;
		for(int scale = 0; scale < i_nbOfScalesToGenerate; ++scale)
		{
			downscaledImages[scale] = downscaleSum(*pPreviousImage);
			pPreviousImage = downscaledImages[scale].get();
		}
		return downscaledImages;
	}

	vector< unique_ptr< Deepimf > > MultiscaleDenoiser::generateDownscaledAverageImages(
			const Deepimf& i_rScale0Image,
			int i_nbOfScalesToGenerate)
	{
		vector< unique_ptr< Deepimf > > downscaledImages(i_nbOfScalesToGenerate);
		const Deepimf* pPreviousImage = &i_rScale0Image;
		for(int scale = 0; scale < i_nbOfScalesToGenerate; ++scale)
		{
			downscaledImages[scale] = downscaleAverage(*pPreviousImage);
			pPreviousImage = downscaledImages[scale].get();
		}
		return downscaledImages;
	}

	vector< unique_ptr< Deepimf > > MultiscaleDenoiser::generateDownscaledSampleCovarianceSumImages(
			const Deepimf& i_rScale0SampleCovarianceImage,
			const Deepimf& i_rScale0NbOfSamplesImage,
			const vector< unique_ptr< Deepimf > >& i_rDownscaledNbOfSamplesImages,
			int i_nbOfScalesToGenerate)
	{
		vector< unique_ptr< Deepimf > > downscaledImages(i_nbOfScalesToGenerate);
		const Deepimf* pPreviousImage = &i_rScale0SampleCovarianceImage;
		const Deepimf* pPreviousNbOfSamplesImage = &i_rScale0NbOfSamplesImage;
		for(int scale = 0; scale < i_nbOfScalesToGenerate; ++scale)
		{
			downscaledImages[scale] = downscaleSampleCovarianceSum(*pPreviousImage, *pPreviousNbOfSamplesImage);
			pPreviousImage = downscaledImages[scale].get();
			pPreviousNbOfSamplesImage = i_rDownscaledNbOfSamplesImages[scale].get();
		}
		return downscaledImages;
	}

#if 0
	vector< unique_ptr< Deepimf > > MultiscaleDenoiser::generateDownscaledWeightedSumImages(
			const Deepimf& i_rScale0Image,
			const Deepimf& i_rScale0WeightImage,
			const vector< unique_ptr< Deepimf > >& i_rDownscaledWeightImages,
			int i_nbOfScalesToGenerate)
	{
		vector< unique_ptr< Deepimf > > downscaledImages(i_nbOfScalesToGenerate);
		const Deepimf* pPreviousImage = &i_rScale0Image;
		const Deepimf* pPreviousWeightImage = &i_rScale0WeightImage;
		for(int scale = 0; scale < i_nbOfScalesToGenerate; ++scale)
		{
			downscaledImages[scale] = downscaleWeightedSum(*pPreviousImage, *pPreviousWeightImage);
			pPreviousImage = downscaledImages[scale].get();
			pPreviousWeightImage = i_rDownscaledWeightImages[scale].get();
		}
		return downscaledImages;
	}

	vector< unique_ptr< Deepimf > > MultiscaleDenoiser::generateDownscaledSquaredWeightedSumImages(
			const Deepimf& i_rScale0Image,
			const Deepimf& i_rScale0WeightImage,
			const vector< unique_ptr< Deepimf > >& i_rDownscaledWeightImages,
			int i_nbOfScalesToGenerate)
	{
		vector< unique_ptr< Deepimf > > downscaledImages(i_nbOfScalesToGenerate, nullptr);
		const Deepimf* pPreviousImage = &i_rScale0Image;
		const Deepimf* pPreviousWeightImage = &i_rScale0WeightImage;
		for(int scale = 0; scale < i_nbOfScalesToGenerate; ++scale)
		{
			downscaledImages[scale] = downscaleSquaredWeightedSum(*pPreviousImage, *pPreviousWeightImage);
			pPreviousImage = downscaledImages[scale].get();
			pPreviousWeightImage = i_rDownscaledWeightImages[scale].get();
		}
		return downscaledImages;
	}
#endif



	unique_ptr< Deepimf > MultiscaleDenoiser::downscaleSum(const Deepimf& i_rImage)
	{
		const int width = i_rImage.getWidth();
		const int height = i_rImage.getHeight();
		const int depth = i_rImage.getDepth();
		const int downscaledWidth = width / 2;
		const int downscaledHeight = height / 2;

		int line, col, z;
		PixelPosition p1, p2, p3, p4;

		unique_ptr< Deepimf > uImage(new Deepimf(downscaledWidth, downscaledHeight, depth));

		for(line = 0; line < downscaledHeight; ++line)
			for(col = 0; col < downscaledWidth; ++col)
			{
				p1 = PixelPosition(2 * line, 2 * col);
				p2 = i_rImage.clamp(p1 + PixelVector(1,0));
				p3 = i_rImage.clamp(p1 + PixelVector(0,1));
				p4 = i_rImage.clamp(p1 + PixelVector(1,1));
				for(z = 0; z < depth; ++z)
					uImage->set(line, col, z,
							i_rImage.get(p1, z) + i_rImage.get(p2, z) + i_rImage.get(p3, z) + i_rImage.get(p4, z));
			}
		return move(uImage);
	}

	unique_ptr< Deepimf > MultiscaleDenoiser::downscaleAverage(const Deepimf& i_rImage)
	{
		const int width = i_rImage.getWidth();
		const int height = i_rImage.getHeight();
		const int depth = i_rImage.getDepth();
		const int downscaledWidth = width / 2;
		const int downscaledHeight = height / 2;

		int line, col, z;
		PixelPosition p1, p2, p3, p4;

		unique_ptr< Deepimf > uImage(new Deepimf(downscaledWidth, downscaledHeight, depth));

		for(line = 0; line < downscaledHeight; ++line)
			for(col = 0; col < downscaledWidth; ++col)
			{
				p1 = PixelPosition(2 * line, 2 * col);
				p2 = i_rImage.clamp(p1 + PixelVector(1,0));
				p3 = i_rImage.clamp(p1 + PixelVector(0,1));
				p4 = i_rImage.clamp(p1 + PixelVector(1,1));
				for(z = 0; z < depth; ++z)
					uImage->set(line, col, z,
							0.25f * (i_rImage.get(p1, z) + i_rImage.get(p2, z) + i_rImage.get(p3, z) + i_rImage.get(p4, z)));
			}
		return uImage;
	}

	unique_ptr< Deepimf > MultiscaleDenoiser::downscaleSampleCovarianceSum(const Deepimf& i_rSampleCovarianceImage, const Deepimf& i_rNbOfSamplesImage)
	{
		const int width = i_rSampleCovarianceImage.getWidth();
		const int height = i_rSampleCovarianceImage.getHeight();
		const int depth = i_rSampleCovarianceImage.getDepth();
		const int downscaledWidth = width / 2;
		const int downscaledHeight = height / 2;

		int line, col, z;
		PixelPosition p1, p2, p3, p4;
		float n1, n2, n3, n4, nSum, w1, w2, w3, w4;

		unique_ptr< Deepimf > uImage(new Deepimf(downscaledWidth, downscaledHeight, depth));

		const float squaredWeight = (1.f / 4.f) * (1.f / 4.f);

		for(line = 0; line < downscaledHeight; ++line)
			for(col = 0; col < downscaledWidth; ++col)
			{
				p1 = PixelPosition(2 * line, 2 * col);
				p2 = i_rSampleCovarianceImage.clamp(p1 + PixelVector(1,0));
				p3 = i_rSampleCovarianceImage.clamp(p1 + PixelVector(0,1));
				p4 = i_rSampleCovarianceImage.clamp(p1 + PixelVector(1,1));
				n1 = i_rNbOfSamplesImage.get(p1, 0);
				n2 = i_rNbOfSamplesImage.get(p2, 0);
				n3 = i_rNbOfSamplesImage.get(p3, 0);
				n4 = i_rNbOfSamplesImage.get(p4, 0);
				nSum = n1 + n2 + n3 + n4;
				w1 = squaredWeight * nSum / n1;
				w2 = squaredWeight * nSum / n2;
				w3 = squaredWeight * nSum / n3;
				w4 = squaredWeight * nSum / n4;
				for(z = 0; z < depth; ++z)
					uImage->set(line, col, z,
							w1 * i_rSampleCovarianceImage.get(p1, z) + w2 * i_rSampleCovarianceImage.get(p2, z) + w3 * i_rSampleCovarianceImage.get(p3, z) + w4 * i_rSampleCovarianceImage.get(p4, z));
			}
		return uImage;
	}

#if 0

	unique_ptr< Deepimf > MultiscaleDenoiser::downscaleWeightedSum(const Deepimf& i_rImage, const Deepimf& i_rWeightImage) const
	{
		const int width = i_rImage.getWidth();
		const int height = i_rImage.getHeight();
		const int depth = i_rImage.getDepth();
		const int downscaledWidth = width / 2;
		const int downscaledHeight = height / 2;

		int line, col, z;
		PixelPosition p1, p2, p3, p4;
		float w1, w2, w3, w4, wInvSum;

		unique_ptr< Deepimf > uImage(new Deepimf(downscaledWidth, downscaledHeight, depth));

		for(line = 0; line < downscaledHeight; ++line)
			for(col = 0; col < downscaledWidth; ++col)
			{
				p1 = PixelPosition(2 * line, 2 * col);
				p2 = i_rSampleCovarianceImage.clamp(p1 + PixelVector(1,0));
				p3 = i_rSampleCovarianceImage.clamp(p1 + PixelVector(0,1));
				p4 = i_rSampleCovarianceImage.clamp(p1 + PixelVector(1,1));
				w1 = i_rWeightImage.get(p1, 0);
				w2 = i_rWeightImage.get(p2, 0);
				w3 = i_rWeightImage.get(p3, 0);
				w4 = i_rWeightImage.get(p4, 0);
				wInvSum = 1.f / (w1 + w2 + w3 + w4);
				w1 *= wInvSum;
				w2 *= wInvSum;
				w3 *= wInvSum;
				w4 *= wInvSum;
				for(z = 0; z < depth; ++z)
					uImage->set(line, col, z,
							w1 * i_rImage.get(p1, z) + w2 * i_rImage.get(p2, z) + w3 * i_rImage.get(p3, z) + w4 * i_rImage.get(p4, z));
			}
		return move(uImage);
	}

	unique_ptr< Deepimf > MultiscaleDenoiser::downscaleSquaredWeightedSum(const Deepimf& i_rImage, const Deepimf& i_rWeightImage) const
	{
		const int width = i_rImage.getWidth();
		const int height = i_rImage.getHeight();
		const int depth = i_rImage.getDepth();
		const int downscaledWidth = width / 2;
		const int downscaledHeight = height / 2;

		int line, col, z;
		PixelPosition p1, p2, p3, p4;
		float w1, w2, w3, w4, wInvSum;

		unique_ptr< Deepimf > uImage(new Deepimf(downscaledWidth, downscaledHeight, depth));

		for(line = 0; line < downscaledHeight; ++line)
			for(col = 0; col < downscaledWidth; ++col)
			{
				p1 = PixelPosition(2 * line, 2 * col);
				p2 = i_rWeightImage.clamp(p1 + PixelVector(1,0));
				p3 = i_rWeightImage.clamp(p1 + PixelVector(0,1));
				p4 = i_rWeightImage.clamp(p1 + PixelVector(1,1));
				w1 = i_rWeightImage.get(p1, 0);
				w2 = i_rWeightImage.get(p2, 0);
				w3 = i_rWeightImage.get(p3, 0);
				w4 = i_rWeightImage.get(p4, 0);
				wInvSum = 1.f / (w1 + w2 + w3 + w4);
				w1 *= wInvSum; w1 *= w1;
				w2 *= wInvSum; w2 *= w2;
				w3 *= wInvSum; w3 *= w3;
				w4 *= wInvSum; w4 *= w4;
				for(z = 0; z < depth; ++z)
					uImage->set(line, col, z,
							i_rImage.get(p1, z) + i_rImage.get(p2, z) + i_rImage.get(p3, z) + i_rImage.get(p4, z));
			}
		return uImage;
	}
#endif

	void MultiscaleDenoiser::mergeOutputsNoInterpolation(Deepimf& o_rMergedImage, const Deepimf& i_rLowFrequencyImage, const Deepimf& i_rHighFrequencyImage)
	{
		const int width = o_rMergedImage.getWidth();
		const int height = o_rMergedImage.getHeight();
		const int depth = o_rMergedImage.getDepth();
		const int downscaledWidth = width / 2;
		const int downscaledHeight = height / 2;

		assert(width == i_rHighFrequencyImage.getWidth());
		assert(height == i_rHighFrequencyImage.getHeight());
		assert(downscaledWidth == i_rLowFrequencyImage.getWidth());
		assert(downscaledHeight == i_rLowFrequencyImage.getHeight());

		int line, col, z;
		PixelPosition p1, p2, p3, p4;
		float v, v1, v2, v3, v4, dv;

		for(line = 0; line < downscaledHeight; ++line)
			for(col = 0; col < downscaledWidth; ++col)
			{
				p1 = PixelPosition(2 * line, 2 * col);
				p2 = p1 + PixelVector(1,0);
				p3 = p1 + PixelVector(0,1);
				p4 = p1 + PixelVector(1,1);
				for(z = 0; z < depth; ++z)
				{
					v = i_rLowFrequencyImage.get(line, col, z);
					v1 = i_rHighFrequencyImage.get(p1, z);
					v2 = i_rHighFrequencyImage.get(p2, z);
					v3 = i_rHighFrequencyImage.get(p3, z);
					v4 = i_rHighFrequencyImage.get(p4, z);
					dv = v - 0.25f * (v1 + v2 + v3 + v4);
					o_rMergedImage.set(p1, z, v1 + dv);
					o_rMergedImage.set(p2, z, v2 + dv);
					o_rMergedImage.set(p3, z, v3 + dv);
					o_rMergedImage.set(p4, z, v4 + dv);
				}
			}
	}

	void MultiscaleDenoiser::mergeOutputs(
			Deepimf& o_rMergedHighResImage,
			Deepimf& o_rTmpHighResImage,
			Deepimf& o_rTmpLowResImage,
			const Deepimf& i_rLowResImage,
			const Deepimf& i_rHighResImage)
	{
		if(&o_rMergedHighResImage != &i_rHighResImage)
			o_rMergedHighResImage = i_rHighResImage;
		lowPass(o_rTmpHighResImage, o_rTmpLowResImage, i_rHighResImage);
		o_rMergedHighResImage -= o_rTmpHighResImage;
		interpolate(o_rTmpHighResImage, i_rLowResImage);
		o_rMergedHighResImage += o_rTmpHighResImage;
	}

	inline int clampPositiveInteger(int i_value, int i_maxValuePlusOne)
	{
		return (i_value <= 0 ? 0 : (i_value >= i_maxValuePlusOne ? i_maxValuePlusOne - 1 : i_value));
	}

	void MultiscaleDenoiser::interpolate(Deepimf& o_rInterpolatedImage, const Deepimf& i_rImage)
	{
		const int width = i_rImage.getWidth();
		const int height = i_rImage.getHeight();
		const int depth = i_rImage.getDepth();
		const int upscaledWidth = o_rInterpolatedImage.getWidth();
		const int upscaledHeight = o_rInterpolatedImage.getHeight();

		assert(width == upscaledWidth / 2);
		assert(height == upscaledHeight / 2);
		assert(depth == o_rInterpolatedImage.getDepth());

		const float mainPixelWeight = 9.f / 16.f;
		const float adjacentPixelWeight = 3.f / 16.f;
		const float diagonalPixelWeight = 1.f / 16.f;

		int upscaledLine, upscaledCol, z, line, col, adjacentLine, adjacentCol;

		for(upscaledLine = 0; upscaledLine < upscaledHeight; ++upscaledLine)
			for(upscaledCol = 0; upscaledCol < upscaledWidth; ++upscaledCol)
			{
				line = upscaledLine / 2;
				col = upscaledCol / 2;
				adjacentLine = clampPositiveInteger(line + ((upscaledLine % 2) * 2 - 1), height);
				adjacentCol = clampPositiveInteger(col + ((upscaledCol % 2) * 2 - 1), width);
				for(z = 0; z < depth; ++z)
				{
					const PixelPosition p1 = i_rImage.clamp(PixelPosition(line, col));
					const PixelPosition p2 = i_rImage.clamp(PixelPosition(line, adjacentCol));
					const PixelPosition p3 = i_rImage.clamp(PixelPosition(adjacentLine, col));
					const PixelPosition p4 = i_rImage.clamp(PixelPosition(adjacentLine, adjacentCol));

					o_rInterpolatedImage.set(upscaledLine, upscaledCol, z,
							mainPixelWeight * i_rImage.get(p1, z) +
							adjacentPixelWeight * (i_rImage.get(p2, z) + i_rImage.get(p3, z)) +
							diagonalPixelWeight * i_rImage.get(p4, z));
				}
			}

	}

	void MultiscaleDenoiser::downscale(Deepimf& o_rDownscaledImage, const Deepimf& i_rImage)
	{
		const int width = i_rImage.getWidth();
		const int height = i_rImage.getHeight();
		const int depth = i_rImage.getDepth();
		const int downscaledWidth = o_rDownscaledImage.getWidth();
		const int downscaledHeight = o_rDownscaledImage.getHeight();

		assert(downscaledWidth == width / 2);
		assert(downscaledHeight == height / 2);

		int line, col, z;
		PixelPosition p1, p2, p3, p4;

		for(line = 0; line < downscaledHeight; ++line)
			for(col = 0; col < downscaledWidth; ++col)
			{
				p1 = PixelPosition(2 * line, 2 * col);
				p2 = i_rImage.clamp(p1 + PixelVector(1,0));
				p3 = i_rImage.clamp(p1 + PixelVector(0,1));
				p4 = i_rImage.clamp(p1 + PixelVector(1,1));
				for(z = 0; z < depth; ++z)
					o_rDownscaledImage.set(line, col, z,
							0.25f * (i_rImage.get(p1, z) + i_rImage.get(p2, z) + i_rImage.get(p3, z) + i_rImage.get(p4, z)));
			}
	}

	void MultiscaleDenoiser::lowPass(
			Deepimf& o_rFilteredImage,
			Deepimf& o_rTmpLowResImage,
			const Deepimf& i_rImage)
	{
		downscale(o_rTmpLowResImage, i_rImage);
		interpolate(o_rFilteredImage, o_rTmpLowResImage);
	}

	void MultiscaleDenoiser::interpolateThenDownscale(Deepimf& o_rFilteredImage, const Deepimf& i_rImage)
	{
		const int width = i_rImage.getWidth();
		const int height = i_rImage.getHeight();
		const int depth = i_rImage.getDepth();

		assert(width == o_rFilteredImage.getWidth());
		assert(height == o_rFilteredImage.getHeight());
		assert(depth == o_rFilteredImage.getDepth());

		const float mainPixelWeight = 36.f / 64.f;
		const float adjacentPixelWeight = 6.f / 64.f;
		const float diagonalPixelWeight = 1.f / 64.f;

		int line, col, z, previousLine, previousCol, nextLine, nextCol;

		for(line = 0; line < height; ++line)
			for(col = 0; col < width; ++col)
			{
				previousLine = clampPositiveInteger(line - 1, height);
				nextLine = clampPositiveInteger(line + 1, height);
				previousCol = clampPositiveInteger(col - 1, width);
				nextCol = clampPositiveInteger(col + 1, width);
				for(z = 0; z < depth; ++z)
					o_rFilteredImage.set(line, col, z,
							mainPixelWeight * i_rImage.get(line, col, z) +
							adjacentPixelWeight * (i_rImage.get(line, previousCol, z) + i_rImage.get(line, nextCol, z) + i_rImage.get(previousLine, col, z) + i_rImage.get(nextLine, col, z)) +
							diagonalPixelWeight * (i_rImage.get(previousLine, previousCol, z) + i_rImage.get(previousLine, nextCol, z) + i_rImage.get(nextLine, previousCol, z) + i_rImage.get(nextLine, nextCol, z)));
			}

	}

} // namespace bcd
