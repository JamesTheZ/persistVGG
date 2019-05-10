#include "cublasFunction.h"

void CNNCublasFunction::init()
{
	CNNFunction::init();
    checkCudaErrors(cublasCreate(&cubHandle));
}

void CNNCublasFunction::fullyConnected(int width, int nChannels, int nFilters, int layerId)
{
	//int num_weights = (width * width * nChannels + 1) * nFilters;
	int filter_size = width * width * nChannels;
	float *d_weights = parameters[layerId];

	float *d_input;
    size_t input_size = (width * width * nChannels + 1) * sizeof(float);
    checkCudaErrors(cudaMalloc(&d_input, input_size));

	if(width == 1)
	{
		checkCudaErrors(cudaMemcpy(d_input, featureOut, nChannels*sizeof(float), cudaMemcpyDefault));
		float val = 1.0f;
		checkCudaErrors(cudaMemcpy(d_input + nChannels, &val, sizeof(float), cudaMemcpyDefault));
	}
	else
	{
		transformFCCublas<<< 1, nChannels >>>(d_input, featureOut, width, nChannels);
	}

	// featureOut will be in NCHW format
	checkCudaErrors(cublasSgemm(cubHandle, CUBLAS_OP_N, 
				CUBLAS_OP_N, 1, nFilters, filter_size+1,
				&alpha, d_input, 1, d_weights, filter_size+1,
				&beta, featureOut, 1));

	checkCudaErrors(cudaFree(d_input));
}

void CNNCublasFunction::maxpool(int width, int nChannels)
{
    float *d_temp;
    size_t mem_size = width * width * nChannels * sizeof(float);
    checkCudaErrors(cudaMalloc(&d_temp, mem_size));
    checkCudaErrors(cudaMemcpy(d_temp, featureOut, mem_size, cudaMemcpyDefault));
    maxpoolingCublas <<< width / 2, width / 2 >>> (featureOut, d_temp, width, nChannels);
    cudaFree(d_temp);
}

void CNNCublasFunction::convolution(int width, int nChannels, int nFilters, int layerId)
{
    //int num_weights = (3 * 3 * nChannels + 1) * nFilters;
    //int output_size = width * width * nFilters;
    //int filter_size = 3 * 3 * nChannels;
    int hidden_width = 3 * 3 * nChannels + 1;

    //float *d_raw_input;
    float *d_input;
    size_t input_size = width * width * hidden_width * sizeof(float);
    checkCudaErrors(cudaMalloc(&d_input, input_size));
    checkCudaErrors(cudaMemset(d_input, 0, input_size));
    // expand original input to (width * width) * (3 * 3 * nChannels + 1) with a 1 at last for bias
    if (nChannels == 3) 
	{
		size_t raw_input_size = width * width * nChannels * sizeof(float);
        checkCudaErrors(cudaMemcpy(featureOut, image, raw_input_size, cudaMemcpyHostToDevice));
        transformImageCublas <<< width, width >>> (d_input, featureOut, width, nChannels);
	}
	else
	{
		// featureOut is in NHWC format, width*width rows and nChannels cols.
        transformCublas <<< width, width >>> (d_input, featureOut, width, nChannels);
	}

    float *d_weights = parameters[layerId];
    // input * weights = ((width * width) * (3 * 3 * nChannels + 1)) * ((3 * 3 * nChannels + 1) * nFilters)
    checkCudaErrors(cublasSgemm(cubHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
				nFilters, width * width, hidden_width,
                &alpha, d_weights, nFilters, d_input, hidden_width,
                &beta, featureOut, nFilters));
	// d_output has width*width rows and nFilters cols.
	
	checkCudaErrors(cudaFree(d_input));
}

__global__ void transformImageCublas(float *input, const float *raw_input, const int width, const int nChannels)
{
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	int start_i = thread_id / width - 1;
	int start_j = thread_id % width - 1;
	int per_channel_width = width * width;
	int hidden_width = 3 * 3 * nChannels + 1;
	int global_offset = thread_id * hidden_width;

	for (int c = 0; c < nChannels; c++) {
		int offset = 0;
		for (int i = start_i; i < start_i + 3; i++) {
			for (int j = start_j; j < start_j + 3; j++) {
				// zero padding
				if(i < 0 || i == width || j < 0 || j == width)
				{
					input[global_offset + c * 9 + offset] = 0;
				}
				else
				{
					input[global_offset + c * 9 + offset] 
						= raw_input[c * per_channel_width + i * width + j];
				}
				offset++;
			}
		}
	}
	input[(thread_id + 1) * hidden_width - 1] = 1;
}

__global__ void transformFCCublas(float *input, const float *raw_input, const int width, const int nChannels)
{
	int thread_id = threadIdx.x;
	int size = width * width;

	for (int s = 0; s < size; s++)
	{
		input[thread_id * size + s] 
			= raw_input[s * nChannels + thread_id];
	}
	if (thread_id == 0)
	{
		input[width * width * nChannels] = 1;
	}
}

// raw_input is in NHWC format
__global__ void transformCublas(float *input, const float *raw_input, const int width, const int nChannels)
{
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	int start_i = thread_id / width - 1;
	int start_j = thread_id % width - 1;
	int hidden_width = 3 * 3 * nChannels + 1;
	int global_offset = thread_id * hidden_width;

	float relu;
	for (int c = 0; c < nChannels; c++) {
		int offset = 0;
		for (int i = start_i; i < start_i + 3; i++) {
			for (int j = start_j; j < start_j + 3; j++) {
				// zero padding
				if(i < 0 || i == width || j < 0 || j == width)
				{
					relu = 0;
				}
				else
				{
					relu = raw_input[(i * width + j) * nChannels + c];
				}
				input[global_offset + c * 9 + offset] = relu < 0 ? 0 : relu;
				offset++;
			}
		}
	}
	input[(thread_id + 1) * hidden_width - 1] = 1;
}

__global__ void maxpoolingCublas(float *output, const float *input, const int width, const int nChannels)
{
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	int new_width = width / 2;
	int i = thread_id / new_width * 2;
	int j = thread_id % new_width * 2;
	int index = i * width + j;

	for (int c = 0; c < nChannels; c++) {
		float max = 0; // this is a relu
		if (max < input[index * nChannels + c])
			max = input[index * nChannels + c];
		if (max < input[(index + 1) * nChannels + c])
			max = input[(index + 1) * nChannels + c];
		if (max < input[(index + width) * nChannels + c])
			max = input[(index + width) * nChannels + c];
		if (max < input[(index + width + 1) * nChannels + c])
			max = input[(index + width + 1) * nChannels + c];
		output[thread_id * nChannels + c] = max;
	}
}


