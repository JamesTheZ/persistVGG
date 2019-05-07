#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cublas_v2.h>

#include "cudnnFunction.h"
#include "cublasFunction.h"

__global__ void isSame(float *array1, float *array2, int width, int height, bool transform)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	int size = width * height;
	int r = threadId / width;
	int c = threadId % width;

	//if(threadId > 1024) // for debugging
	//{
	//	return;
	//}

	if(threadId < size)
	{
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
}

int main(int argc, char **argv)
{
    char *image_file = argv[1];
    char *weights_file = argv[2];
    char *bias_file = argv[3];
    char *output_file = argv[4];

	int blockDim = 256;
	int gridDim;

	CNNFunction *func = new CNNCudnnFunction();
	CNNFunction *funcCublas = new CNNCublasFunction();

	func->init();
	func->readImage(image_file);
	func->readParameters(weights_file, bias_file);

	checkCudaErrors(cudaDeviceSynchronize());
	
	funcCublas->init();
	funcCublas->readImage(image_file);
	funcCublas->readParameters(weights_file, bias_file);

	checkCudaErrors(cudaDeviceSynchronize());

	//gridDim = (224 * 224 * 3 + blockDim - 1) / blockDim;
	//isSame<<<gridDim, blockDim>>>(func->image, funcCublas->image, 224 * 224, 3, false);
	//checkCudaErrors(cudaDeviceSynchronize());

	//return 0;

    // ReLU layers in transform kernel or maxpooling
    func->convolution(224, 3, 64, 0);
    func->convolution(224, 64, 64, 1);
    func->maxpool(224, 64);
    func->convolution(112, 64, 128, 2);
    func->convolution(112, 128, 128, 3);
    func->maxpool(112, 128);
    func->convolution(56, 128, 256, 4);
    func->convolution(56, 256, 256, 5);
    func->convolution(56, 256, 256, 6);
    func->convolution(56, 256, 256, 7);
    func->maxpool(56, 256);
    func->convolution(28, 256, 512, 8);
    func->convolution(28, 512, 512, 9);
    func->convolution(28, 512, 512, 10);
    func->convolution(28, 512, 512, 11);
    func->maxpool(28, 512);
    func->convolution(14, 512, 512, 12);
    func->convolution(14, 512, 512, 13);
    func->convolution(14, 512, 512, 14);
    func->convolution(14, 512, 512, 15 );
    func->maxpool(14, 512);
    func->fullyConnected(7, 512, 4096, 16); // most time consuming file input
    func->fullyConnected(1, 4096, 4096, 17);
    func->fullyConnected(1, 4096, 1000, 18);

    funcCublas->convolution(224, 3, 64, 0);
    funcCublas->convolution(224, 64, 64, 1);
    funcCublas->maxpool(224, 64);
    funcCublas->convolution(112, 64, 128, 2);
    funcCublas->convolution(112, 128, 128, 3);
    funcCublas->maxpool(112, 128);
    funcCublas->convolution(56, 128, 256, 4);
    funcCublas->convolution(56, 256, 256, 5);
    funcCublas->convolution(56, 256, 256, 6);
    funcCublas->convolution(56, 256, 256, 7);
    funcCublas->maxpool(56, 256);
    funcCublas->convolution(28, 256, 512, 8);
    funcCublas->convolution(28, 512, 512, 9);
    funcCublas->convolution(28, 512, 512, 10);
    funcCublas->convolution(28, 512, 512, 11);
    funcCublas->maxpool(28, 512);
    funcCublas->convolution(14, 512, 512, 12);
    funcCublas->convolution(14, 512, 512, 13);
    funcCublas->convolution(14, 512, 512, 14);
    funcCublas->convolution(14, 512, 512, 15 );
    funcCublas->maxpool(14, 512);
    funcCublas->fullyConnected(7, 512, 4096, 16); // most time consuming file input
    funcCublas->fullyConnected(1, 4096, 4096, 17);
    funcCublas->fullyConnected(1, 4096, 1000, 18);

	checkCudaErrors(cudaDeviceSynchronize());
	gridDim = (14 * 14 * 512 + blockDim - 1) / blockDim;
	isSame<<<gridDim, blockDim>>>(func->featureOut, funcCublas->featureOut, 14 * 14, 512, true);
	checkCudaErrors(cudaDeviceSynchronize());

	return 0;

    // write 1000 dimension
    func->writeOutput(output_file);

    return 0;
}


