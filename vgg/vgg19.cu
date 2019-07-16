#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cublas_v2.h>

#include "util.h"

#include "cudnnFunction.h"
#include "cublasFunction.h"
#include "cudaFunction.h"

__global__ void isSame(float *array1, float *array2, int height, int width, bool transform)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if(threadId >= height * width)
	{
		return;
	}

	int r = threadId / width;
	int c = threadId % width;

	//if(threadId > 1024) // for debugging
	//{
	//	return;
	//}

	if(transform)
	{
		if(fabsf(array1[r * width + c] - array2[c * height + r]) > 1e-3)
		{
			printf("not same at %d, %f ~ %f\n", 
					threadId, array1[r * width + c], array2[c * height + r]);
		}
	}
	else
	{
		if(fabsf(array1[r * width + c] - array2[r * width + c]) > 1e-3)
		{
			printf("not same at %d, %f ~ %f\n", 
					threadId, array1[r * width + c], array2[r * width + c]);
		}
	}
}

__global__ void printArray(float *array, int nBatch, int nChannels, int height, int width, int n)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if(threadId >= n)
	{
		return;
	}

	int posN = threadId / (nChannels * height * width);
	int posC = threadId % (nChannels * height * width) / (height * width);
	int posH = threadId % (height * width) / width;
	int posW = threadId % (height * width) % width;

	printf("%d-%d-%d-%d: %f\n", posN, posC, posH, posW, array[threadId]);
}

int main(int argc, char **argv)
{
	//checkCudaErrors(cudaSetDevice(1));
	printf("num args: %d\n", argc);
	char *image_file = argv[1];
	char *weights_file = argv[2];
	char *bias_file = argv[3];
	char *output_file = argv[4];

	struct timeval start, end;

	int blockDim = 256;
	int gridDim;

	CNNFunction *func = new CNNCudaFunction();
	func->init();
	func->readImage(image_file);
	func->readParameters(weights_file, bias_file);

	/*
	{ // for debugging
		CNNFunction *funcCudnn = new CNNCudnnFunction();
		funcCudnn->init();
		funcCudnn->readImage(image_file);
		funcCudnn->readParameters(weights_file, bias_file);

		//		checkCudaErrors(cudaDeviceSynchronize());
		//		printf("filter out:\n");
		//		fflush(NULL);
		//		printArray<<<1, 3 * 3 * 1>>>(func->weights[0], 1, 1, 3, 3, 3 * 3 * 1);
		//		checkCudaErrors(cudaDeviceSynchronize());
		//		printf("featuremap:\n");
		//		fflush(NULL);
		//		printArray<<<1, 4 * 4 * 1>>>(func->featureOut, 1, 1, 4, 4, 4 * 4 * 1);

		int theWidth = 224;
		int theChannels = 3;
		int theFilters = 64;
		func->convolution(theWidth, theChannels, theFilters, 0);
		checkCudaErrors(cudaDeviceSynchronize());
		funcCudnn->convolution(theWidth, theChannels, theFilters, 0);
		checkCudaErrors(cudaDeviceSynchronize());

		gridDim = (theWidth * theWidth * theFilters + blockDim - 1) / blockDim;
		isSame<<<gridDim, blockDim>>>(func->featureOut, funcCudnn->featureOut, theWidth * theWidth, theFilters, false);

		//printf("cuda out:\n");
		//fflush(NULL);
		//printArray<<<gridDim, blockDim>>>(func->featureOut, 1, theFilters, theWidth, theWidth, theFilters * theWidth * theWidth);
		//checkCudaErrors(cudaDeviceSynchronize());

		//printf("cudnn out:\n");
		//fflush(NULL);
		//printArray<<<1, 4 * 4 * 2>>>(funcCudnn->featureOut, 1, 2, 4, 4, 4 * 4 * 2);
		checkCudaErrors(cudaDeviceSynchronize());

		return 0;
	}
	*/
	//CNNFunction *funcCudnn = new CNNCudnnFunction();
	//funcCudnn->init();
	//funcCudnn->readImage(image_file);
	//funcCudnn->readParameters(weights_file, bias_file);

	//func->convolution(224, 3, 64, 0);
	//funcCudnn->convolution(224, 3, 64, 0);

	int theWidth = 224;
	int theChannels = 3;
	int theFilters = 64;

	checkCudaErrors(cudaDeviceSynchronize());
	gettimeofday(&start, NULL);

	func->convolution(theWidth, theChannels, theFilters, 0);

#ifdef DEBUG
	return 0;
#endif

	//funcCudnn->convolution(theWidth, theChannels, theFilters, 0);

	//gridDim = (theWidth * theWidth * theFilters + blockDim - 1) / blockDim;
	//isSame<<<gridDim, blockDim>>>(func->featureOut, funcCudnn->featureOut, theWidth * theWidth, theFilters, false);
	//checkCudaErrors(cudaDeviceSynchronize());
	//return 0;

	//func->convolution(4, 2, 1, 0);
	//funcCudnn->convolution(4, 2, 1, 0);

	//checkCudaErrors(cudaDeviceSynchronize());
	//gridDim = (4 * 4 * 1 + blockDim - 1) / blockDim;
	//isSame<<<gridDim, blockDim>>>(func->featureOut, funcCudnn->featureOut, 4 * 4, 1, false);
	//checkCudaErrors(cudaDeviceSynchronize());
	//return 0;

	func->convolution(224, 64, 64, 1);
	//funcCudnn->convolution(224, 64, 64, 1);

	func->maxpool(224, 64);
	//funcCudnn->maxpool(224, 64);

	func->convolution(112, 64, 128, 2);
	//funcCudnn->convolution(112, 64, 128, 2);

	func->convolution(112, 128, 128, 3);
	//funcCudnn->convolution(112, 128, 128, 3);

	func->maxpool(112, 128);
	//funcCudnn->maxpool(112, 128);

	func->convolution(56, 128, 256, 4);
	//funcCudnn->convolution(56, 128, 256, 4);

	func->convolution(56, 256, 256, 5);
	//funcCudnn->convolution(56, 256, 256, 5);

	func->convolution(56, 256, 256, 6);
	//funcCudnn->convolution(56, 256, 256, 6);

	func->convolution(56, 256, 256, 7);
	//funcCudnn->convolution(56, 256, 256, 7);

	func->maxpool(56, 256);
	//funcCudnn->maxpool(56, 256);

	func->convolution(28, 256, 512, 8);
	//funcCudnn->convolution(28, 256, 512, 8);

	func->convolution(28, 512, 512, 9);
	//funcCudnn->convolution(28, 512, 512, 9);

	func->convolution(28, 512, 512, 10);
	//funcCudnn->convolution(28, 512, 512, 10);

	func->convolution(28, 512, 512, 11);
	//funcCudnn->convolution(28, 512, 512, 11);

	func->maxpool(28, 512);
	//funcCudnn->maxpool(28, 512);

	func->convolution(14, 512, 512, 12);
	//funcCudnn->convolution(14, 512, 512, 12);

	func->convolution(14, 512, 512, 13);
	//funcCudnn->convolution(14, 512, 512, 13);

	func->convolution(14, 512, 512, 14);
	//funcCudnn->convolution(14, 512, 512, 14);

	func->convolution(14, 512, 512, 15 );
	//funcCudnn->convolution(14, 512, 512, 15 );

	func->maxpool(14, 512);
	//funcCudnn->maxpool(14, 512);
	
	checkCudaErrors(cudaDeviceSynchronize());
	gettimeofday(&end, NULL);
	struct timeval delta = timeDelta(start, end);
	printTime(delta);

	func->fullyConnected(7, 512, 4096, 16); // most time consuming file input
	//funcCudnn->fullyConnected(7, 512, 4096, 16); // most time consuming file input

	func->fullyConnected(1, 4096, 4096, 17);
	//funcCudnn->fullyConnected(1, 4096, 4096, 17);

	func->fullyConnected(1, 4096, 1000, 18);
	//funcCudnn->fullyConnected(1, 4096, 1000, 18);

	// write 1000 dimension
	func->writeOutput(output_file);

	return 0;
}


