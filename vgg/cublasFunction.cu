#include "cublasFunction.h"

void CNNCublasFunction::init()
{
	CNNFunction::init();
    checkCudaErrors(cublasCreate(&cubHandle));
}

void CNNCublasFunction::fullyConnected(int width, int channels, int num_filters, int layerId)
{
	int num_weights = (width * width * channels + 1) * num_filters;
	int filter_size = width * width * channels;
	float *d_weights = parameters[layerId];

	float *d_input;
    size_t input_size = (width * width * channels + 1) * sizeof(float);
    checkCudaErrors(cudaMalloc(&d_input, input_size));

	if(width == 1)
	{
		checkCudaErrors(cudaMemcpy(d_input, featureOut, channels*sizeof(float), cudaMemcpyDefault));
		float val = 1.0f;
		checkCudaErrors(cudaMemcpy(d_input + channels, &val, sizeof(float), cudaMemcpyDefault));
	}
	else
	{
		transformFCCublas<<< 1, channels >>>(d_input, featureOut, width, channels);
	}

	// featureOut will be in NCHW format
	checkCudaErrors(cublasSgemm(cubHandle, CUBLAS_OP_N, 
				CUBLAS_OP_N, 1, num_filters, filter_size+1,
				&alpha, d_input, 1, d_weights, filter_size+1,
				&beta, featureOut, 1));

	checkCudaErrors(cudaFree(d_input));
}

void CNNCublasFunction::maxpool(int width, int channels)
{
    float *d_temp;
    size_t mem_size = width * width * channels * sizeof(float);
    checkCudaErrors(cudaMalloc(&d_temp, mem_size));
    checkCudaErrors(cudaMemcpy(d_temp, featureOut, mem_size, cudaMemcpyDefault));
    maxpoolingCublas <<< width / 2, width / 2 >>> (featureOut, d_temp, width, channels);
    cudaFree(d_temp);
}

void CNNCublasFunction::convolution(int width, int channels, int num_filters, int layerId)
{
    int num_weights = (3 * 3 * channels + 1) * num_filters;
    int output_size = width * width * num_filters;
    int filter_size = 3 * 3 * channels;
    int hidden_width = 3 * 3 * channels + 1;

    float *d_raw_input;
    float *d_input;
    size_t input_size = width * width * hidden_width * sizeof(float);
    checkCudaErrors(cudaMalloc(&d_input, input_size));
    checkCudaErrors(cudaMemset(d_input, 0, input_size));
    // expand original input to (width * width) * (3 * 3 * channels + 1) with a 1 at last for bias
    if (channels == 3) 
	{
		size_t raw_input_size = width * width * channels * sizeof(float);
        checkCudaErrors(cudaMemcpy(featureOut, image, raw_input_size, cudaMemcpyHostToDevice));
        transformImageCublas <<< width, width >>> (d_input, featureOut, width, channels);
	}
	else
	{
		// featureOut is in NHWC format, width*width rows and channels cols.
        transformCublas <<< width, width >>> (d_input, featureOut, width, channels);
	}

    float *d_weights = parameters[layerId];
    // input * weights = ((width * width) * (3 * 3 * channels + 1)) * ((3 * 3 * channels + 1) * num_filters)
    checkCudaErrors(cublasSgemm(cubHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
				num_filters, width * width, hidden_width,
                &alpha, d_weights, num_filters, d_input, hidden_width,
                &beta, featureOut, num_filters));
	// d_output has width*width rows and num_filters cols.
	
	checkCudaErrors(cudaFree(d_input));
}

__global__ void transformImageCublas(float *input, const float *raw_input, const int width, const int channels)
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

__global__ void transformFCCublas(float *input, const float *raw_input, const int width, const int channels)
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

// raw_input is in NHWC format
__global__ void transformCublas(float *input, const float *raw_input, const int width, const int channels)
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

__global__ void maxpoolingCublas(float *output, const float *input, const int width, const int channels)
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


