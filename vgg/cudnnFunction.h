#ifndef CNN_CUDNN_FUNCTION_H
#define CNN_CUDNN_FUNCTION_H

#include "cudnn.h"
#include "cublas_v2.h"
#include "function.h"

class CNNCudnnFunction : public CNNFunction
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
		virtual void convolution(int width, int channels, int num_filters, int layerId) override;
		virtual void fullyConnected(int width, int channels, int num_filters, int layerId) override;
		virtual void maxpool(int width, int channels) override;

};

#endif
