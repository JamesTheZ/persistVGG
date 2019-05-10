#ifndef CNN_CUDA_FUNCTION_H
#define CNN_CUDA_FUNCTION_H

#include "cudnn.h"
#include "cublas_v2.h"
#include "function.h"

class CNNCudaFunction : public CNNFunction
{
	private:
		cudnnHandle_t cudnnHandle;
		cublasHandle_t cublasHandle;
		cudnnTensorDescriptor_t cudnnIDesc;
		cudnnFilterDescriptor_t cudnnFDesc;
		cudnnTensorDescriptor_t cudnnODesc;
		cudnnTensorDescriptor_t cudnnBiasDesc;
		cudnnConvolutionDescriptor_t cudnnConvDesc;
		cudnnActivationDescriptor_t cudnnActDesc;
		cudnnPoolingDescriptor_t cudnnPoolDesc;

		const float alpha = 1.0f;
		const float beta = 0.0f;

	public:
		virtual void init() override;
		virtual void convolution(int width, int nChannels, int nFilters, int layerId) override;
		virtual void fullyConnected(int width, int nChannels, int nFilters, int layerId) override;
		virtual void maxpool(int width, int nChannels) override;

	private:
		void relu(float* dArray, int size);

};

#endif
