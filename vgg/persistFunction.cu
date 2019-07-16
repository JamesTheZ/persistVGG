#include "persistFunction.h"

void CNNPersistFunction::init()
{
	CNNFunction::init();

	checkCudaErrors(cudaMallocManaged(&firstSignalIn, sizeof(int)*MAX_QUERY));
	checkCudaErrors(cudaMallocManaged(&lastSignalOut, sizeof(int)*MAX_QUERY));

	for(int i=0; i<19; i++)
	{
		checkCudaErrors(cudaMalloc(&signalIn[i], sizeof(int)*MAX_QUERY));
		checkCudaErrors(cudaMallocManaged(&SMs[i], sizeof(int)*MAX_QUERY));
	}

	// assign SM here.

}

void CNNCudaFunction::convolution(int width, int nChannels, int nFilters, int layerId)
{





	std::size_t inputSize = width * width * nChannels * sizeof(float);
	float* dInput = nullptr;
	checkCudaErrors(cudaMalloc(&dInput, inputSize));
	checkCudaErrors(cudaMemcpy(dInput, featureOut, inputSize, cudaMemcpyDefault));
	float *dFilter = weights[layerId];

	const int tileDepth = 4;
	const int blockWidth = 32;
	const int blockHeight = 8;
	const int blockSize = blockWidth * blockHeight;

	// numblocks per sm
	int numBlocksPerSM = 0;
	int sharedUsage = sizeof(float) * tileDepth * 
		(3 * 3 + (blockWidth + 2) * (blockHeight + 2));
	checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
				&numBlocksPerSM, convBiasPersist, 
				blockSize, sharedUsage));
	int layerSMs = SMs[layerId];
	int maxBlocks = numBlocksPerSM * prop.multiProcessorCount;
	int gDimX = (max(width * width, maxBlocks * blockSize) + blockSize - 1) / blockSize;

	//int nBlockW = (width + blockWidth - 1) / blockWidth;
	//int nBlockH = (width + blockHeight - 1) / blockHeight;
	//int blockDimConv = blockWidth * blockHeight;

	dim3 gridDimConv(gDimX, (nFilters + ITEM_PER_THREAD - 1) / ITEM_PER_THREAD);
	convBiasPersist<tileDepth, blockWidth, blockHeight><<<gridDimConv, blockDimConv>>>(
			dInput, dFilter, nFilters, nChannels, 
			width, bias[layerId], featureOut);

	// activation: relu
	int blockDimAct = 256;
	int gridDimAct = (nFilters * width * width + blockDimAct - 1) / blockDimAct;
	reluForward<<<gridDimAct, blockDimAct>>>(featureOut, featureOut, nFilters * width * width);

	checkCudaErrors(cudaFree(dInput));
}

