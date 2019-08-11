#include "function.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

void CNNFunction::readImage(char *imageFile)
{
	// required to normalize by mean pixel (in rgb order)
	float mean_pixel[3] = {123.68, 116.779, 103.939};

	FILE *fin = fopen(imageFile, "r");
	int total = 224 * 224 * 3;
	for (int index = 0; index < total; index++) {
		fscanf(fin, "%f", &image[index]);
		image[index] -= mean_pixel[index / 50176]; // 50176 = 224 * 224
	}
	fclose(fin);

	if(featureOut == nullptr)
	{
		printf("featureOut is not malloced. exit.\n");
		fflush(NULL);
		exit(1);
	}

	checkCudaErrors(cudaMemcpy(featureOut, image, 224 * 224 * 3 * sizeof(float), cudaMemcpyDefault));
}

void CNNFunction::init()
{
	// malloc weights
	for(int i=0; i<19; i++)
	{
		checkCudaErrors(cudaMalloc(&weights[i], 
					filterSize[i]*channels[i]*numFilters[i]*sizeof(float)));
		//printf("addr: %p\n", weights[i]);
	}

	// malloc bias
	for(int i=0; i<19; i++)
	{
		checkCudaErrors(cudaMalloc(&bias[i], 
					numFilters[i]*sizeof(float)));
	}

	// malloc parameters 
	for(int i=0; i<19; i++)
	{
		checkCudaErrors(cudaMalloc(&parameters[i], 
					((filterSize[i]*channels[i]+1)*numFilters[i])*sizeof(float)));
	}

	// malloc features, max feature map: 224 * 224 * 64
	checkCudaErrors(cudaMalloc(&featureOut, (224*224*64)*sizeof(float)));
}

void CNNFunction::readParameters(char *weightsFile, char *biasFile)
{
	FILE *fw = fopen(weightsFile, "r");
	FILE *fb = fopen(biasFile, "r");

	if(fw == NULL || fb == NULL)
	{
		printf("failed to open parameter files.\n");
		exit(1);
	}

	// init weights & bias. row major
	// fused paramters, colum major for conv layers, row major for fc layer
	int maxWeightSize = 7 * 7 * 512 * 4096;
	float *hWeights = (float*)malloc(maxWeightSize*sizeof(float));
	float *hParameters = (float*)malloc((maxWeightSize+4096)*sizeof(float));
	float *hBias= (float*)malloc(4096*sizeof(float));
	if(hWeights == NULL || hParameters == NULL || hBias == NULL)
	{
		printf("fail to malloc for weights or bias\n");
		exit(1);
	}
	for(int i=0; i<16; i++)
	{
#ifdef DEBUG
		if(i == 1) // for debugging
		{
			printf("For debugging, only load layer 1.\n");
			fflush(NULL);
			return;
		}
#endif
		int hiddenSize = filterSize[i] * channels[i];
		int weightSize = hiddenSize * numFilters[i];
		int biasSize = numFilters[i];
		int parameterSize = weightSize + biasSize;
		for (int nf = 0; nf < numFilters[i]; nf++) 
		{
			for (int hs = 0; hs < hiddenSize; hs++)
			{
				fscanf(fw, "%f", &hWeights[nf * hiddenSize + hs]);
				hParameters[hs * numFilters[i] + nf] 
					= hWeights[nf * hiddenSize + hs];
			}
			fscanf(fb, "%f", &hBias[nf]);
			// hBias[nf] = 0; // for debugging
			hParameters[hiddenSize * numFilters[i] + nf] = hBias[nf];
		}
		//printf("%p, %p, %d\n", weights[i], hWeights, weightSize*sizeof(float));
		checkCudaErrors(cudaMemcpy(weights[i], hWeights, 
					weightSize*sizeof(float), cudaMemcpyDefault));
		checkCudaErrors(cudaMemcpy(parameters[i], hParameters, 
					parameterSize*sizeof(float), cudaMemcpyDefault));
		checkCudaErrors(cudaMemcpy(bias[i], hBias, 
					biasSize*sizeof(float), cudaMemcpyDefault));
	}
	for(int i=16; i<19; i++)
	{
		int hiddenSize = filterSize[i] * channels[i];
		int weightSize = hiddenSize * numFilters[i];
		int biasSize = numFilters[i];
		int parameterSize = weightSize + biasSize;
		for (int nf = 0; nf < numFilters[i]; nf++) 
		{
			for (int hs = 0; hs < hiddenSize; hs++)
			{
				fscanf(fw, "%f", &hWeights[nf * hiddenSize + hs]);
				hParameters[nf * (hiddenSize + 1) + hs] 
					= hWeights[nf * hiddenSize + hs];
			}
			fscanf(fb, "%f", &hBias[nf]);
			hParameters[nf * (hiddenSize + 1) + hiddenSize] = hBias[nf];
		}
		checkCudaErrors(cudaMemcpy(weights[i], hWeights, 
					weightSize*sizeof(float), cudaMemcpyDefault));
		checkCudaErrors(cudaMemcpy(parameters[i], hParameters, 
					parameterSize*sizeof(float), cudaMemcpyDefault));
		checkCudaErrors(cudaMemcpy(bias[i], hBias, 
					biasSize*sizeof(float), cudaMemcpyDefault));
	}
	checkCudaErrors(cudaDeviceSynchronize());
	free(hWeights);
	free(hParameters);
	free(hBias);

	fclose(fw);
	fclose(fb);
}

void CNNFunction::writeOutput(char *output_file)
{
	FILE *fout = fopen(output_file, "w");

	float *output = (float *)malloc(1000 * sizeof(float));
	checkCudaErrors(cudaMemcpy(output, featureOut, 1000 * sizeof(float), cudaMemcpyDeviceToHost));

	for (int i = 0; i < 1000; i++)
		fprintf(fout, "%f\n", output[i]);

	free(output);
	fclose(fout);
}

void CNNFunction::fullyConnected(int width, int nChannels, int nFilters, int layerId)
{
	printf("ERROR: %s Not defined\n", __FUNCTION__);
	exit(1);
}

void CNNFunction::maxpool(int width, int nChannels)
{
	printf("ERROR: %s Not defined\n", __FUNCTION__);
	exit(1);
}

void CNNFunction::convolution(int width, int nChannels, int nFilters, int layerId)
{
	printf("ERROR: %s Not defined\n", __FUNCTION__);
	exit(1);
}

void CNNFunction::convPersist(int width, int nChannels, int nFilters, int layerId)
{
	printf("ERROR: %s Not defined\n", __FUNCTION__);
	exit(1);
}
