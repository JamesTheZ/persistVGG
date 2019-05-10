#ifndef CNN_CUBLAS_FUNCTION_H
#define CNN_CUBLAS_FUNCTION_H

#include "cublas_v2.h"
#include "function.h"

__global__ void transformImageCublas(float *input, const float *raw_input, const int width, const int nChannels);
__global__ void transformFCCublas(float *input, const float *raw_input, const int width, const int nChannels);
__global__ void transformCublas(float *input, const float *raw_input, const int width, const int nChannels);
__global__ void maxpoolingCublas(float *output, const float *input, const int width, const int nChannels);

class CNNCublasFunction : public CNNFunction
{
	private:
		cublasHandle_t cubHandle;
		const float alpha = 1.0f;
		const float beta = 0.0f;

	public:
		virtual void init() override;
		virtual void convolution(int width, int nChannels, int nFilters, int layerId) override;
		virtual void fullyConnected(int width, int nChannels, int nFilters, int layerId) override;
		virtual void maxpool(int width, int nChannels) override;

};

#endif
