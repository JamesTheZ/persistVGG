#include "cudnnFunction.h"

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

	checkCudaErrors(cudaMemcpy(featureOut, image, 224 * 224 * 3 * sizeof(float), cudaMemcpyDefault));
}

void CNNCudnnFunction::fullyConnected(int width, int channels, int numFilters, int layerId)
{
	int filterSize = width * width * channels;
	float *featureIn = nullptr;
	checkCudaErrors(cudaMalloc(&featureIn, filterSize * sizeof(float)));
	checkCudaErrors(cudaMemcpy(featureIn, featureOut, filterSize * sizeof(float), cudaMemcpyDefault));

	// output = filter * featureMap
	// CUBLAS is column major, which needs extra transform
	checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
				numFilters, 1, filterSize, &alpha, weights[layerId], filterSize, 
				featureIn, filterSize, &beta, featureOut, filterSize));

	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(featureIn));

	//int num_weights = (width * width * channels + 1) * num_filters;
	//int filter_size = width * width * channels;
	//float *d_weights = parameters[layerId];

	//float *d_input;
    //size_t input_size = (width * width * channels + 1) * sizeof(float);
    //checkCudaErrors(cudaMalloc(&d_input, input_size));

	//if(width == 1)
	//{
	//	checkCudaErrors(cudaMemcpy(d_input, featureOut, channels*sizeof(float), cudaMemcpyDefault));
	//	float val = 1.0f;
	//	checkCudaErrors(cudaMemcpy(d_input + channels, &val, sizeof(float), cudaMemcpyDefault));
	//}
	//else
	//{
	//	transformFCCudnn<<< 1, channels >>>(d_input, featureOut, width, channels);
	//}

	//checkCudaErrors(cudnnSgemm(cubHandle, CUDNN_OP_N, 
	//			CUDNN_OP_N, 1, num_filters, filter_size+1,
	//			&alpha, d_input, 1, d_weights, filter_size+1,
	//			&beta, featureOut, 1));

	//checkCudaErrors(cudaFree(d_input));
}

void CNNCudnnFunction::maxpool(int width, int channels)
{
	cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
	cudnnDataType_t type = CUDNN_DATA_FLOAT;
	checkCudaErrors(cudnnSetTensor4dDescriptor(cudnnIDesc,
				format, type, 1, channels, width, width));
	checkCudaErrors(cudnnSetTensor4dDescriptor(cudnnODesc,
				format, type, 1, channels, width / 2, width / 2));

	float* featureIn = nullptr;
	int featureSize = width * width * channels;
	checkCudaErrors(cudaMalloc(&featureIn, featureSize * sizeof(float)));
	checkCudaErrors(cudaMemcpy(featureIn, featureOut, featureSize, cudaMemcpyDefault));

	checkCudaErrors(cudnnPoolingForward(cudnnHandle, cudnnPoolDesc,
				&alpha, cudnnIDesc, featureIn, &beta, cudnnODesc, featureOut));

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaFree(featureIn));
	
    //float *d_temp;
    //size_t mem_size = width * width * channels * sizeof(float);
    //checkCudaErrors(cudaMalloc(&d_temp, mem_size));
    //checkCudaErrors(cudaMemcpy(d_temp, featureOut, mem_size, cudaMemcpyDefault));
    //maxpoolingCudnn <<< width / 2, width / 2 >>> (featureOut, d_temp, width, channels);
    //cudaFree(d_temp);
}

void CNNCudnnFunction::convolution(int width, int channels, int num_filters, int layerId)
{
	cudnnDataType_t type = CUDNN_DATA_FLOAT;
	cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;

	checkCudaErrors(cudnnSetTensor4dDescriptor(cudnnIDesc,
				format, type, 1, channels, width, width));
	checkCudaErrors(cudnnSetFilter4dDescriptor(cudnnFDesc,
				type, format, num_filters, channels, 3, 3));
	checkCudaErrors(cudnnSetTensor4dDescriptor(cudnnODesc,
				format, type, 1, num_filters, width, width));

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

	std::size_t inputSize = width * width * channels * sizeof(float);
	float* dInput= nullptr;
    checkCudaErrors(cudaMalloc(&dInput, inputSize));
	// memcpy and activation: relu
	checkCudaErrors(cudnnActivationForward(cudnnHandle, cudnnActDesc, 
				&alpha, cudnnIDesc, featureOut, &beta, cudnnIDesc, dInput));

    float *dFilter = weights[layerId];

	checkCudaErrors(cudnnConvolutionForward(cudnnHandle,
				&alpha, cudnnIDesc, dInput, cudnnFDesc, dFilter, 
				cudnnConvDesc, cudnnConvFwdAlgo, dWorkspace, workspaceSize,
				&beta, cudnnODesc, featureOut));
	
	// add bias
	checkCudaErrors(cudnnSetTensor4dDescriptor(cudnnBiasDesc,
				format, type, 1, num_filters, 1, 1));
	checkCudaErrors(cudnnAddTensor(cudnnHandle, 
				&alpha, cudnnBiasDesc, bias[layerId], 
				&alpha, cudnnODesc, featureOut));

	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(dInput));
	checkCudaErrors(cudaFree(dWorkspace));

	//checkCudaErrors(cudnnCreateTensorDescriptor(&cudnnIDesc));
	//checkCudaErrors(cudnnCreateFilterDescriptor(&cudnnFDesc));
	//checkCudaErrors(cudnnCreateTensorDescriptor(&cudnnODesc));

    //int num_weights = (3 * 3 * channels + 1) * num_filters;
    //int output_size = width * width * num_filters;
    //int filter_size = 3 * 3 * channels;
    //int hidden_width = 3 * 3 * channels + 1;

    //float *d_raw_input;
    //float *d_input;
    //size_t input_size = width * width * hidden_width * sizeof(float);
    //checkCudaErrors(cudaMalloc(&d_input, input_size));
    //checkCudaErrors(cudaMemset(d_input, 0, input_size));
    //// expand original input to (width * width) * (3 * 3 * channels + 1) with a 1 at last for bias
    //if (channels == 3) 
	//{
	//	size_t raw_input_size = width * width * channels * sizeof(float);
    //    checkCudaErrors(cudaMemcpy(featureOut, image, raw_input_size, cudaMemcpyHostToDevice));
    //    transformImageCudnn <<< width, width >>> (d_input, featureOut, width, channels);
	//}
	//else
	//{
	//	// d_output has width*width rows and channels cols.
    //    transformCudnn <<< width, width >>> (d_input, featureOut, width, channels);
	//}

    //float *d_weights = parameters[layerId];
    //// input * weights = ((width * width) * (3 * 3 * channels + 1)) * ((3 * 3 * channels + 1) * num_filters)
    //checkCudaErrors(cudnnSgemm(cubHandle, CUDNN_OP_N, CUBLAS_OP_N, 
	//			num_filters, width * width, hidden_width,
    //            &alpha, d_weights, num_filters, d_input, hidden_width,
    //            &beta, featureOut, num_filters));
	//// d_output has width*width rows and num_filters cols.
	//
	//checkCudaErrors(cudaFree(d_input));
}

/*
__global__ void transformImageCudnn(float *input, const float *raw_input, const int width, const int channels)
{
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	int start_i = thread_id / width - 1;
	int start_j = thread_id % width - 1;
	int per_channel_width = width * width;
	int hidden_width = 3 * 3 * channels + 1;
	int global_offset = thread_id * hidden_width;

	for (int c = 0; c < channels; c++) {
		int offset = 0;
		for (int i = start_i; i < start_i + 3; i++) {
			if (i < 0 || i == width)
				continue;
			for (int j = start_j; j < start_j + 3; j++) {
				if (j < 0 || j == width)
					continue;
				input[global_offset + c * 9 + offset] 
					= raw_input[c * per_channel_width + i * width + j];
				offset++;
			}
		}
		// padding ?? added by Zhen
		// while(offset < 9)
		// {
		// 	input[offset++] = 0;
		// }
	}
	input[(thread_id + 1) * hidden_width - 1] = 1;
}

__global__ void transformFCCudnn(float *input, const float *raw_input, const int width, const int channels)
{
	int thread_id = threadIdx.x;
	int size = width * width;

	for (int s = 0; s < size; s++)
	{
		input[thread_id * size + s] 
			= raw_input[s * channels + thread_id];
	}
	if (thread_id == 0)
	{
		input[width * width * channels] = 1;
	}
}

__global__ void transformCudnn(float *input, const float *raw_input, const int width, const int channels)
{
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	int start_i = thread_id / width - 1;
	int start_j = thread_id % width - 1;
	int hidden_width = 3 * 3 * channels + 1;
	int global_offset = thread_id * hidden_width;

	float relu;
	for (int c = 0; c < channels; c++) {
		int offset = 0;
		for (int i = start_i; i < start_i + 3; i++) {
			if (i < 0 || i == width)
				continue;
			for (int j = start_j; j < start_j + 3; j++) {
				if (j < 0 || j == width)
					continue;
				relu = raw_input[(i * width + j) * channels + c];
				input[global_offset + c * 9 + offset] = relu < 0 ? 0 : relu;
				offset++;
			}
		}
		// padding, is this correct ?? added by Zhen
		// while(offset < 9)
		// {
		// 	input[offset++] = 0;
		// }
	}
	input[(thread_id + 1) * hidden_width - 1] = 1;
}

__global__ void maxpoolingCudnn(float *output, const float *input, const int width, const int channels)
{
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	int new_width = width / 2;
	int i = thread_id / new_width * 2;
	int j = thread_id % new_width * 2;
	int index = i * width + j;

	for (int c = 0; c < channels; c++) {
		float max = 0;
		if (max < input[index * channels + c])
			max = input[index * channels + c];
		if (max < input[(index + 1) * channels + c])
			max = input[(index + 1) * channels + c];
		if (max < input[(index + width) * channels + c])
			max = input[(index + width) * channels + c];
		if (max < input[(index + width + 1) * channels + c])
			max = input[(index + width + 1) * channels + c];
		output[thread_id * channels + c] = max;
	}
}
*/

