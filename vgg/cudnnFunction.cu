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

void CNNCudnnFunction::fullyConnected(int width, int numChannels, int numFilters, int layerId)
{
	int filterSize = width * width * numChannels;
	float *featureIn = nullptr;
	checkCudaErrors(cudaMalloc(&featureIn, filterSize * sizeof(float)));
	checkCudaErrors(cudaMemcpy(featureIn, featureOut, filterSize * sizeof(float), cudaMemcpyDefault));

	// output = filter * featureMap
	// CUBLAS is column major, which needs extra transform
	checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
				numFilters, 1, filterSize, &alpha, weights[layerId], filterSize, 
				featureIn, filterSize, &beta, featureOut, filterSize));

	// add bias
	checkCudaErrors(cudnnSetTensor4dDescriptor(cudnnODesc,
				CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, numFilters, 1, 1));
	checkCudaErrors(cudnnSetTensor4dDescriptor(cudnnBiasDesc,
				CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, numFilters, 1, 1));
	checkCudaErrors(cudnnAddTensor(cudnnHandle, 
				&alpha, cudnnBiasDesc, bias[layerId], 
				&alpha, cudnnODesc, featureOut));

	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(featureIn));

	//int num_weights = (width * width * numChannels + 1) * numFilters;
	//int filter_size = width * width * numChannels;
	//float *d_weights = parameters[layerId];

	//float *d_input;
    //size_t input_size = (width * width * numChannels + 1) * sizeof(float);
    //checkCudaErrors(cudaMalloc(&d_input, input_size));

	//if(width == 1)
	//{
	//	checkCudaErrors(cudaMemcpy(d_input, featureOut, numChannels*sizeof(float), cudaMemcpyDefault));
	//	float val = 1.0f;
	//	checkCudaErrors(cudaMemcpy(d_input + numChannels, &val, sizeof(float), cudaMemcpyDefault));
	//}
	//else
	//{
	//	transformFCCudnn<<< 1, numChannels >>>(d_input, featureOut, width, numChannels);
	//}

	//checkCudaErrors(cudnnSgemm(cubHandle, CUDNN_OP_N, 
	//			CUDNN_OP_N, 1, numFilters, filter_size+1,
	//			&alpha, d_input, 1, d_weights, filter_size+1,
	//			&beta, featureOut, 1));

	//checkCudaErrors(cudaFree(d_input));
}

void CNNCudnnFunction::maxpool(int width, int numChannels)
{
	cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
	cudnnDataType_t type = CUDNN_DATA_FLOAT;
	checkCudaErrors(cudnnSetTensor4dDescriptor(cudnnIDesc,
				format, type, 1, numChannels, width, width));
	checkCudaErrors(cudnnSetTensor4dDescriptor(cudnnODesc,
				format, type, 1, numChannels, width / 2, width / 2));

	float* featureIn = nullptr;
	int featureSize = width * width * numChannels;
	checkCudaErrors(cudaMalloc(&featureIn, featureSize * sizeof(float)));
	checkCudaErrors(cudaMemcpy(featureIn, featureOut, featureSize, cudaMemcpyDefault));

	checkCudaErrors(cudnnPoolingForward(cudnnHandle, cudnnPoolDesc,
				&alpha, cudnnIDesc, featureIn, &beta, cudnnODesc, featureOut));

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaFree(featureIn));
	
    //float *d_temp;
    //size_t mem_size = width * width * numChannels * sizeof(float);
    //checkCudaErrors(cudaMalloc(&d_temp, mem_size));
    //checkCudaErrors(cudaMemcpy(d_temp, featureOut, mem_size, cudaMemcpyDefault));
    //maxpoolingCudnn <<< width / 2, width / 2 >>> (featureOut, d_temp, width, numChannels);
    //cudaFree(d_temp);
}

void CNNCudnnFunction::convolution(int width, int numChannels, int numFilters, int layerId)
{
	cudnnDataType_t type = CUDNN_DATA_FLOAT;
	cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;

	checkCudaErrors(cudnnSetTensor4dDescriptor(cudnnIDesc,
				format, type, 1, numChannels, width, width));
	checkCudaErrors(cudnnSetFilter4dDescriptor(cudnnFDesc,
				type, format, numFilters, numChannels, 3, 3));
	checkCudaErrors(cudnnSetTensor4dDescriptor(cudnnODesc,
				format, type, 1, numFilters, width, width));

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

	std::size_t inputSize = width * width * numChannels * sizeof(float);
	float* dInput= nullptr;
    checkCudaErrors(cudaMalloc(&dInput, inputSize));

	// memcpy and activation: relu
	if(layerId != 0)
	{
		checkCudaErrors(cudnnActivationForward(cudnnHandle, cudnnActDesc, 
					&alpha, cudnnIDesc, featureOut, &beta, cudnnIDesc, dInput));
	}
	else
	{
		checkCudaErrors(cudaMemcpy(dInput, featureOut, inputSize, cudaMemcpyDefault));
	}

    float *dFilter = weights[layerId];

	//printDArray(dInput, width * width * numChannels);

	//exit(0);

	checkCudaErrors(cudnnConvolutionForward(cudnnHandle,
				&alpha, cudnnIDesc, dInput, cudnnFDesc, dFilter, 
				cudnnConvDesc, cudnnConvFwdAlgo, dWorkspace, workspaceSize,
				&beta, cudnnODesc, featureOut));
	
	// add bias
	checkCudaErrors(cudnnSetTensor4dDescriptor(cudnnBiasDesc,
				format, type, 1, numFilters, 1, 1));
	checkCudaErrors(cudnnAddTensor(cudnnHandle, 
				&alpha, cudnnBiasDesc, bias[layerId], 
				&alpha, cudnnODesc, featureOut));

	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(dInput));
	checkCudaErrors(cudaFree(dWorkspace));

	//checkCudaErrors(cudnnCreateTensorDescriptor(&cudnnIDesc));
	//checkCudaErrors(cudnnCreateFilterDescriptor(&cudnnFDesc));
	//checkCudaErrors(cudnnCreateTensorDescriptor(&cudnnODesc));

    //int num_weights = (3 * 3 * numChannels + 1) * numFilters;
    //int output_size = width * width * numFilters;
    //int filter_size = 3 * 3 * numChannels;
    //int hidden_width = 3 * 3 * numChannels + 1;

    //float *d_raw_input;
    //float *d_input;
    //size_t input_size = width * width * hidden_width * sizeof(float);
    //checkCudaErrors(cudaMalloc(&d_input, input_size));
    //checkCudaErrors(cudaMemset(d_input, 0, input_size));
    //// expand original input to (width * width) * (3 * 3 * numChannels + 1) with a 1 at last for bias
    //if (numChannels == 3) 
	//{
	//	size_t raw_input_size = width * width * numChannels * sizeof(float);
    //    checkCudaErrors(cudaMemcpy(featureOut, image, raw_input_size, cudaMemcpyHostToDevice));
    //    transformImageCudnn <<< width, width >>> (d_input, featureOut, width, numChannels);
	//}
	//else
	//{
	//	// d_output has width*width rows and numChannels cols.
    //    transformCudnn <<< width, width >>> (d_input, featureOut, width, numChannels);
	//}

    //float *d_weights = parameters[layerId];
    //// input * weights = ((width * width) * (3 * 3 * numChannels + 1)) * ((3 * 3 * numChannels + 1) * numFilters)
    //checkCudaErrors(cudnnSgemm(cubHandle, CUDNN_OP_N, CUBLAS_OP_N, 
	//			numFilters, width * width, hidden_width,
    //            &alpha, d_weights, numFilters, d_input, hidden_width,
    //            &beta, featureOut, numFilters));
	//// d_output has width*width rows and numFilters cols.
	//
	//checkCudaErrors(cudaFree(d_input));
}

