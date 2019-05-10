#include "cudaFunction.h"
#include <cassert>

//void printDArray(float *dArray, int size)
//{
//	float *hArray = new float[size];
//
//	checkCudaErrors(cudaMemcpy(hArray, dArray, size * sizeof(float), cudaMemcpyDefault));
//
//	for(int i=0; i<size; i++)
//	{
//		printf("%f\n", hArray[i]);
//	}
//	fflush(NULL);
//}

__global__ void convBias(float* fIn, float* filter, 
		const int nFilters, const int nChannels, const int width, 
		float* bias, float* fOut)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= nFilters * width * width)
	{
		return;
	}

	int filterId = tid / (width * width);
	int row = (tid / width) % width;
	int col = tid % width;

	float sum = bias[filterId];
	for(int ch = 0; ch < nChannels; ch++)
	{
		// filter dim is 3
		int i = 0;
		for(int r = row-1; r <= row+1; r++, i++) 
		{
			if(r < 0 || r >= width)
			{
				continue;
			}
			int j = 0;
			for(int c = col-1; c <= col+1; c++, j++)
			{
				if(c < 0 || c >= width)
				{
					continue;
				}
				sum += filter[filterId * nChannels * 9 + ch * 9 + i * 3 + j]
					* fIn[ch * width * width + r * width + c];
			}
		}
	}
	fOut[filterId * width * width + row * width + col] = sum;
}

__global__ void reluForward(float* fIn, float* fOut, const int size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= size)
	{
		return;
	}

	fOut[tid] = fIn[tid] < 0 ? 0 : fIn[tid];
}

// stride: {2, 2}, filterSize: {2, 2}
// make sure that (width % 2 == 0)
__global__ void maxPooling(float *fIn, float *fOut, const int width, const int nChannels)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid >= nChannels * width * width / 4)
	{
		return;
	}

	int outWidth = width / 2;
	int channelId = tid / (outWidth * outWidth);
	int row = (tid / outWidth) % outWidth;
	int col = tid % outWidth;
	int oldRow = row * 2;
	int oldCol = col * 2;

	fOut[channelId * outWidth * outWidth + row * outWidth + col] = fmaxf(
			fmaxf(fIn[channelId * width * width + oldRow * width + oldCol],
				fIn[channelId * width * width + (oldRow+1) * width + oldCol]),
			fmaxf(fIn[channelId * width * width + oldRow * width + (oldCol+1)],
				fIn[channelId * width * width + (oldRow+1) * width + (oldCol+1)]));
}

	template <int BLOCK_HEIGHT, int BLOCK_WIDTH>
__global__ void fullyConnectCUDA(float *fIn, float *filter, 
		int batchSize, int nChannels, int height, int width, int nFilters,
		float *bias, float *fOut)
{
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int wFeature = nChannels * height * width; // width of feature matrix

	if(bx * BLOCK_WIDTH + tx >= nFilters || by * BLOCK_HEIGHT + ty >= batchSize)
	{
		return;
	}

	// Index of the first sub-matrix of A processed by the block
	int aBegin = wFeature * BLOCK_HEIGHT * by;

	// Index of the last sub-matrix of A processed by the block
	int aEnd   = aBegin + wFeature - 1;
	aEnd = aEnd <= batchSize - 1 ? aEnd : batchSize - 1;

	// Step size used to iterate through the sub-matrices of A
	int aStep  = BLOCK_WIDTH;

	// Index of the first sub-matrix of B processed by the block
	// Note that matrix B should be transformed implicit
	int bBegin = wFeature * BLOCK_HEIGHT * bx;

	// Step size used to iterate through the sub-matrices of B
	// Note that matrix B should be transformed implicit
	int bStep  = BLOCK_WIDTH;

	// Csub is used to store the element of the block sub-matrix
	// that is computed by the thread
	float Csub = 0;

	// Loop over all the sub-matrices of A and B
	// required to compute the block sub-matrix
	for (int a = aBegin, b = bBegin;
			a <= aEnd;
			a += aStep, b += bStep) {
		// Declaration of the shared memory array As used to
		// store the sub-matrix of A
		__shared__ float As[BLOCK_HEIGHT][BLOCK_WIDTH];

		// Declaration of the shared memory array Bs used to
		// store the sub-matrix of B
		__shared__ float Bs[BLOCK_HEIGHT][BLOCK_WIDTH];

		// Load the matrices from device memory
		// to shared memory; each thread loads
		// one element of each matrix
		As[ty][tx] = fIn[a + wFeature * ty + tx];
		Bs[ty][tx] = filter[b + wFeature * ty + tx];

		// Synchronize to make sure the matrices are loaded
		__syncthreads();

		// Multiply the two matrices together;
		// each thread computes one element
		// of the block sub-matrix
#pragma unroll

		for (int k = 0; k < BLOCK_WIDTH; ++k) {
			Csub += As[ty][k] * Bs[tx][k];
			//Csub += As[ty][k] * Bs[k][tx];
		}

		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		__syncthreads();
	}

	// Write the block sub-matrix to device memory;
	// each thread writes one element
	int c = nFilters * BLOCK_HEIGHT * by + BLOCK_WIDTH * bx;
	fOut[c + nFilters * ty + tx] = Csub + bias[BLOCK_HEIGHT * by + ty];
}

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
//dim3 threads(block_size, block_size);
//dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);
//    MatrixMulCUDA<16> <<< grid, threads >>>(d_C, d_A, d_B,
//                                            dimsA.x, dimsB.x);
template <int BLOCK_SIZE> __global__ void matrixMul(
		float *C, float *A, float *B, int wA, int wB) {
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Index of the first sub-matrix of A processed by the block
	int aBegin = wA * BLOCK_SIZE * by;

	// Index of the last sub-matrix of A processed by the block
	int aEnd   = aBegin + wA - 1;

	// Step size used to iterate through the sub-matrices of A
	int aStep  = BLOCK_SIZE;

	// Index of the first sub-matrix of B processed by the block
	int bBegin = BLOCK_SIZE * bx;

	// Step size used to iterate through the sub-matrices of B
	int bStep  = BLOCK_SIZE * wB;

	// Csub is used to store the element of the block sub-matrix
	// that is computed by the thread
	float Csub = 0;

	// Loop over all the sub-matrices of A and B
	// required to compute the block sub-matrix
	for (int a = aBegin, b = bBegin;
			a <= aEnd;
			a += aStep, b += bStep) {
		// Declaration of the shared memory array As used to
		// store the sub-matrix of A
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

		// Declaration of the shared memory array Bs used to
		// store the sub-matrix of B
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		// Load the matrices from device memory
		// to shared memory; each thread loads
		// one element of each matrix
		As[ty][tx] = A[a + wA * ty + tx];
		Bs[ty][tx] = B[b + wB * ty + tx];

		// Synchronize to make sure the matrices are loaded
		__syncthreads();

		// Multiply the two matrices together;
		// each thread computes one element
		// of the block sub-matrix
#pragma unroll

		for (int k = 0; k < BLOCK_SIZE; ++k) {
			Csub += As[ty][k] * Bs[k][tx];
		}

		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		__syncthreads();
	}

	// Write the block sub-matrix to device memory;
	// each thread writes one element
	int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
	C[c + wB * ty + tx] = Csub;
}

void CNNCudaFunction::init()
{
	CNNFunction::init();

	checkCudaErrors(cudnnCreate(&cudnnHandle));
	checkCudaErrors(cublasCreate(&cublasHandle));
	checkCudaErrors(cudnnCreateTensorDescriptor(&cudnnIDesc));
	checkCudaErrors(cudnnCreateFilterDescriptor(&cudnnFDesc));
	checkCudaErrors(cudnnCreateTensorDescriptor(&cudnnODesc));
	checkCudaErrors(cudnnCreateTensorDescriptor(&cudnnBiasDesc));

	// all activations in VGGNET are the same.
	checkCudaErrors(cudnnCreateActivationDescriptor(&cudnnActDesc));
	checkCudaErrors(cudnnSetActivationDescriptor(cudnnActDesc,
				CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));
}

void CNNCudaFunction::fullyConnected(int width, int nChannels, int nFilters, int layerId)
{
	int filterSize = width * width * nChannels;
	float *featureIn = nullptr;
	checkCudaErrors(cudaMalloc(&featureIn, filterSize * sizeof(float)));
	checkCudaErrors(cudaMemcpy(featureIn, featureOut, filterSize * sizeof(float), cudaMemcpyDefault));

	const int batchSize = 1; // should be 2^n and lt 256 
	const int blockDimX = 256 / batchSize;
	const int blockDimY = batchSize;
	dim3 threads(blockDimX, blockDimY);
	const int gridDimX = (nFilters + blockDimX - 1) / blockDimX;
	const int gridDimY = (batchSize + blockDimY - 1) / blockDimY;
	dim3 grid(gridDimX, gridDimY);
	fullyConnectCUDA<blockDimY, blockDimX> <<<grid, threads>>>(
			featureIn, weights[layerId], 
			batchSize, nChannels, width, width, nFilters,
			bias[layerId], featureOut);

	//int filterSize = width * width * nChannels;
	//float *featureIn = nullptr;
	//checkCudaErrors(cudaMalloc(&featureIn, filterSize * sizeof(float)));
	//checkCudaErrors(cudaMemcpy(featureIn, featureOut, filterSize * sizeof(float), cudaMemcpyDefault));

	//// CUBLAS is column major, which needs extra transform
	//checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
	//			nFilters, 1, filterSize, &alpha, weights[layerId], filterSize, 
	//			featureIn, filterSize, &beta, featureOut, filterSize));

	//// add bias
	//checkCudaErrors(cudnnSetTensor4dDescriptor(cudnnODesc,
	//			CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, nFilters, 1, 1));
	//checkCudaErrors(cudnnSetTensor4dDescriptor(cudnnBiasDesc,
	//			CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, nFilters, 1, 1));
	//checkCudaErrors(cudnnAddTensor(cudnnHandle, 
	//			&alpha, cudnnBiasDesc, bias[layerId], 
	//			&alpha, cudnnODesc, featureOut));

	// activation
	//checkCudaErrors(cudnnActivationForward(cudnnHandle, cudnnActDesc, 
	//			&alpha, cudnnODesc, featureOut, &beta, cudnnODesc, featureOut));

	// activation: relu
	reluForward<<<(nFilters + 255) / 256, 256>>>(
			featureOut, featureOut, nFilters);

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaFree(featureIn));
}

void CNNCudaFunction::maxpool(int width, int nChannels)
{
	float* featureIn = nullptr;
	int featureSize = width * width * nChannels;
	checkCudaErrors(cudaMalloc(&featureIn, featureSize * sizeof(float)));
	checkCudaErrors(cudaMemcpy(featureIn, featureOut, featureSize * sizeof(float), cudaMemcpyDefault));

	// only deal with width of even number
	assert(width % 2 == 0);
	int blockDim = 256;
	int gridDim = (nChannels * width * width / 4 + blockDim - 1) / blockDim;
	maxPooling<<<gridDim, blockDim>>>(featureIn, featureOut, width, nChannels);

	checkCudaErrors(cudaFree(featureIn));
}

void CNNCudaFunction::convolution(int width, int nChannels, int nFilters, int layerId)
{
	std::size_t inputSize = width * width * nChannels * sizeof(float);
	float* dInput= nullptr;
	checkCudaErrors(cudaMalloc(&dInput, inputSize));
	checkCudaErrors(cudaMemcpy(dInput, featureOut, inputSize, cudaMemcpyDefault));
	float *dFilter = weights[layerId];

	int blockDim = 256;
	int gridDim = (nFilters * width * width + blockDim - 1) / blockDim;
	convBias<<<gridDim, blockDim>>>(dInput, dFilter, nFilters, nChannels, 
			width, bias[layerId], featureOut);

	// activation: relu
	reluForward<<<gridDim, blockDim>>>(featureOut, featureOut, nFilters * width * width);

	checkCudaErrors(cudaFree(dInput));
}

