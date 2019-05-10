#include "cudnnFunction.h"

void printDArray(float *dArray, int size)
{
	float *hArray = new float[size];

	checkCudaErrors(cudaMemcpy(hArray, dArray, size * sizeof(float), cudaMemcpyDefault));

	for(int i=0; i<size; i++)
	{
		printf("%f\n", hArray[i]);
	}
	fflush(NULL);
}

void CNNCudnnFunction::init()
{
	CNNFunction::init();

    checkCudaErrors(cudnnCreate(&cudnnHandle));
    checkCudaErrors(cublasCreate(&cublasHandle));
	checkCudaErrors(cudnnCreateTensorDescriptor(&cudnnIDesc));
	checkCudaErrors(cudnnCreateFilterDescriptor(&cudnnFDesc));
	checkCudaErrors(cudnnCreateTensorDescriptor(&cudnnODesc));
	checkCudaErrors(cudnnCreateTensorDescriptor(&cudnnBiasDesc));

	checkCudaErrors(cudnnCreateConvolutionDescriptor(&cudnnConvDesc));
	
	// all activations in VGGNET are the same.
	checkCudaErrors(cudnnCreateActivationDescriptor(&cudnnActDesc));
	checkCudaErrors(cudnnSetActivationDescriptor(cudnnActDesc,
				CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));

	// all poolings in VGGNET are the same.
	checkCudaErrors(cudnnCreatePoolingDescriptor(&cudnnPoolDesc));
	const int poolDim = 2;
	int windowDim[poolDim] = {2, 2}; 
	int padding[poolDim] = {0, 0}; 
	int stride[poolDim] = {2, 2}; 
	checkCudaErrors(cudnnSetPoolingNdDescriptor(cudnnPoolDesc,
				CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
				poolDim, windowDim, padding, stride));
}

void CNNCudnnFunction::fullyConnected(int width, int nChannels, int nFilters, int layerId)
{
	int filterSize = width * width * nChannels;
	float *featureIn = nullptr;
	checkCudaErrors(cudaMalloc(&featureIn, filterSize * sizeof(float)));
	checkCudaErrors(cudaMemcpy(featureIn, featureOut, filterSize * sizeof(float), cudaMemcpyDefault));

	// CUBLAS is column major, which needs extra transform
	checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
				nFilters, 1, filterSize, &alpha, weights[layerId], filterSize, 
				featureIn, filterSize, &beta, featureOut, filterSize));

	// add bias
	checkCudaErrors(cudnnSetTensor4dDescriptor(cudnnODesc,
				CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, nFilters, 1, 1));
	checkCudaErrors(cudnnSetTensor4dDescriptor(cudnnBiasDesc,
				CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, nFilters, 1, 1));
	checkCudaErrors(cudnnAddTensor(cudnnHandle, 
				&alpha, cudnnBiasDesc, bias[layerId], 
				&alpha, cudnnODesc, featureOut));

	// activation
	checkCudaErrors(cudnnActivationForward(cudnnHandle, cudnnActDesc, 
				&alpha, cudnnODesc, featureOut, &beta, cudnnODesc, featureOut));

	checkCudaErrors(cudaFree(featureIn));
}

void CNNCudnnFunction::maxpool(int width, int nChannels)
{
	cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
	cudnnDataType_t type = CUDNN_DATA_FLOAT;
	checkCudaErrors(cudnnSetTensor4dDescriptor(cudnnIDesc,
				format, type, 1, nChannels, width, width));
	checkCudaErrors(cudnnSetTensor4dDescriptor(cudnnODesc,
				format, type, 1, nChannels, width / 2, width / 2));

	float* featureIn = nullptr;
	int featureSize = width * width * nChannels;
	checkCudaErrors(cudaMalloc(&featureIn, featureSize * sizeof(float)));
	checkCudaErrors(cudaMemcpy(featureIn, featureOut, featureSize * sizeof(float), cudaMemcpyDefault));

	checkCudaErrors(cudnnPoolingForward(cudnnHandle, cudnnPoolDesc,
				&alpha, cudnnIDesc, featureIn, &beta, cudnnODesc, featureOut));

	checkCudaErrors(cudaFree(featureIn));
}

void CNNCudnnFunction::convolution(int width, int nChannels, int nFilters, int layerId)
{
	cudnnDataType_t type = CUDNN_DATA_FLOAT;
	cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;

	checkCudaErrors(cudnnSetTensor4dDescriptor(cudnnIDesc,
				format, type, 1, nChannels, width, width));
	checkCudaErrors(cudnnSetFilter4dDescriptor(cudnnFDesc,
				type, format, nFilters, nChannels, 3, 3));
	checkCudaErrors(cudnnSetTensor4dDescriptor(cudnnODesc,
				format, type, 1, nFilters, width, width));

	checkCudaErrors(cudnnSetConvolution2dDescriptor(cudnnConvDesc,
				1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

	cudnnConvolutionFwdAlgo_t cudnnConvFwdAlgo;
	checkCudaErrors(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
				cudnnIDesc, cudnnFDesc, cudnnConvDesc, cudnnODesc, 
				CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &cudnnConvFwdAlgo));

	std::size_t workspaceSize = 0;
	checkCudaErrors(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
				cudnnIDesc, cudnnFDesc, cudnnConvDesc, cudnnODesc,
				cudnnConvFwdAlgo, &workspaceSize));

	float* dWorkspace = nullptr;
	checkCudaErrors(cudaMalloc(&dWorkspace, workspaceSize));

	std::size_t inputSize = width * width * nChannels * sizeof(float);
	float* dInput= nullptr;
    checkCudaErrors(cudaMalloc(&dInput, inputSize));
	checkCudaErrors(cudaMemcpy(dInput, featureOut, inputSize, cudaMemcpyDefault));

    float *dFilter = weights[layerId];

	checkCudaErrors(cudnnConvolutionForward(cudnnHandle,
				&alpha, cudnnIDesc, dInput, cudnnFDesc, dFilter, 
				cudnnConvDesc, cudnnConvFwdAlgo, dWorkspace, workspaceSize,
				&beta, cudnnODesc, featureOut));
	
	// add bias
	checkCudaErrors(cudnnSetTensor4dDescriptor(cudnnBiasDesc,
				format, type, 1, nFilters, 1, 1));
	checkCudaErrors(cudnnAddTensor(cudnnHandle, 
				&alpha, cudnnBiasDesc, bias[layerId], 
				&alpha, cudnnODesc, featureOut));

	// activation
	checkCudaErrors(cudnnActivationForward(cudnnHandle, cudnnActDesc, 
				&alpha, cudnnODesc, featureOut, &beta, cudnnODesc, featureOut));

	checkCudaErrors(cudaFree(dInput));
	checkCudaErrors(cudaFree(dWorkspace));
}

