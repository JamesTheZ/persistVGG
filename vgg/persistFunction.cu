#include "persistFunction.h"

// TODO: confirm this macro
#define ITEM_PER_THREAD 16

int volatile __managed__ signalIn[20];
int volatile __managed__ SMs[20];

void CNNPersistFunction::init()
{
	CNNFunction::init();

	// using unified memory for convinent, to optimize..
	checkCudaErrors(cudaMallocManaged(&firstSignalIn, sizeof(int)*MAX_QUERY));
	checkCudaErrors(cudaMallocManaged(&lastSignalOut, sizeof(int)*MAX_QUERY));

	//for(int i=0; i<20; i++)
	//{
	//	checkCudaErrors(cudaMalloc(&signalIn, sizeof(int)*MAX_QUERY));
	//	checkCudaErrors(cudaMallocManaged(&SMs, sizeof(int)*MAX_QUERY));
	//}

	// TODO: assign SM here.
	for(int i=0; i<20; i++)
	{
		signalIn[i] = 0;
		SMs[i] = 1;
	}

	__sync_synchronize();
}

__device__ inline uint getSMId()
{
	uint smid;
	asm("mov.u32 %0, %smid;" : "=r" (smid));
	return smid;
}

__device__ int initBlockId[32] = {0};

// TODO: using cooperate group
	template <int TILE_DEPTH, int BLOCK_WIDTH, int BLOCK_HEIGHT>
__global__ void convBiasReluPersist(float* fIn, float* filter, 
		const int nFilters, const int nChannels, const int width, 
		float* bias, float* fOut, int layerId, int totalBlocks,
		int blockCapacity, int minSM, int maxSM,
		int nDimX, int nDimY, int nDimZ)
{
	int smid = getSMId();
	//printf("id: %d, min: %d, max: %d\n", smid, minSM, maxSM);
	if(smid < minSM || smid >= maxSM)
	{
		return;
	}

	//printf("id: %d, min: %d, max: %d\n", smid, minSM, maxSM);

	int const TILE_WIDTH = BLOCK_WIDTH + 2;
	int const TILE_HEIGHT = BLOCK_HEIGHT + 2;
	int const blockDim = BLOCK_WIDTH * BLOCK_HEIGHT;
	int const tileDim = TILE_WIDTH * TILE_HEIGHT;

	__shared__ float sFilter[TILE_DEPTH * 3 * 3];
	__shared__ float sFeature[TILE_DEPTH * tileDim];

	// row/col in the block
	int blockR = threadIdx.x / BLOCK_WIDTH;
	int blockC = threadIdx.x % BLOCK_WIDTH;

	int split = (nChannels + TILE_DEPTH - 1) / TILE_DEPTH;

	// persistent-thread begin
	__shared__ int isContinue;
	if(threadIdx.x == 0)
	{
		isContinue = 1;
	}
	__syncthreads();
	while(isContinue)
	{
		__shared__ int blockId;
		if(threadIdx.x == 0)
		{
			if(atomicCAS_system((int*)&signalIn[layerId], 1, 2) != 1)
			// if(signalIn[layerId] != 1) // 1 means current layer shoud be processed
			{
				//printf("signalIn[%d] = %d\n", layerId, signalIn[layerId]);
				blockId = -1;
				//continue;
			}
			else
			{
				//signalIn[layerId] = 2; // 2 means current layer is beging processing
				//__threadfence();
				blockId = atomicAdd(&initBlockId[layerId], 1);
			}
		}
		__syncthreads();
		if(blockId == -1)
		{
			continue;
		}
		printf("read data for layer %d\n", layerId);
		// blockId is the virtual global block ID for current layer
#pragma unroll 1
		for(; blockId < totalBlocks;)
		{
			if(threadIdx.x == 0)
			{
				//printf("blockId: %d/%d\n", blockId, totalBlocks);
			}
			// blockIdx_x/y/z are virtual blockIdx.x/y/z
			int blockIdx_x = blockId % nDimX;
			int blockIdx_y = (blockId / nDimX) % nDimY;
			int blockIdx_z = blockId / (nDimX * nDimY); // to get filter id

			// row/col in the input feature map 
			int row = blockIdx_y * BLOCK_HEIGHT + blockR;
			int col = blockIdx_x * BLOCK_WIDTH + blockC;

			int filterId = blockIdx_z * ITEM_PER_THREAD;
			float sum[ITEM_PER_THREAD] = {0.0};
#pragma unroll 1
			for(int ft = 0; 
					ft < ITEM_PER_THREAD && ft + filterId < nFilters; 
					ft++)
			{
				sum[ft] = bias[ft + filterId];
			}

			int ch = 0; // current processed channel ID.
#pragma unroll 1
			for(int sp = 0; sp < split; sp++)
			{
				int destY, destX; // dest in smem
				int srcY, srcX; // src in gmem
				float inFeatureScope;

				// load feature map to shared mem
				//ch = sp * TILE_DEPTH;
				//int iterSize = min(nChannels - ch, TILE_DEPTH) * tileDim;
				//for(int idx = threadIdx.x; idx < iterSize; idx += blockDim)
				//{
				//	int subCh = idx / tileDim;
				//	destY = idx / TILE_WIDTH % TILE_HEIGHT;
				//	destX = idx % TILE_WIDTH;
				//	srcY = blockIdx.y * BLOCK_HEIGHT + destY - 1;
				//	srcX = blockIdx.x * BLOCK_WIDTH + destX - 1;

				//	sFeature[idx] = (srcY >= 0 && srcY < width && srcX >= 0 && srcX < width) ? 
				//		fIn[(ch + subCh) * width * width + srcY * width + srcX] : 0;
				//}

				// load feature map to shared mem
#pragma unroll 1
				for (int iter = 0; 
						iter <= tileDim / blockDim; 
						iter++)
				{
					destY = (threadIdx.x + iter * blockDim) / TILE_WIDTH;
					destX = (threadIdx.x + iter * blockDim) % TILE_WIDTH;
					if (destY < TILE_HEIGHT) // && destX < TILE_WIDTH)
					{
						srcY = blockIdx_y * BLOCK_HEIGHT + destY - 1;
						srcX = blockIdx_x * BLOCK_WIDTH + destX - 1;
						inFeatureScope = (srcY >= 0 && srcY < width && srcX >= 0 && srcX < width);
						ch = sp * TILE_DEPTH;
#pragma unroll 1
						for(int subCh = 0; 
								subCh < TILE_DEPTH && ch < nChannels;
								subCh++, ch++)
						{
							// feature map
							sFeature[subCh * tileDim + destY * TILE_WIDTH + destX]
								= inFeatureScope ? fIn[ch * width * width + srcY * width + srcX] : 0;
						}
					}
				}

				//__syncthreads();

#pragma unroll 1
				for(int ft = 0; 
						ft < ITEM_PER_THREAD && ft + filterId < nFilters; 
						ft++)
				{

					// put filter into shared memory
#pragma unroll 1
					for (int iter = 0; 
							iter <= TILE_DEPTH * 9 / blockDim; 
							iter++)
					{
						int idx = threadIdx.x + iter * blockDim;
						if(idx < TILE_DEPTH * 9)
						{
							sFilter[idx] = filter[((ft + filterId) * nChannels + sp * TILE_DEPTH) * 9 + idx];
						}
					}

					__syncthreads();

					// calculate convolution
					if(row < width && col < width)
					{
						ch = sp * TILE_DEPTH;
#pragma unroll 1
						for(int subCh = 0; subCh < TILE_DEPTH && ch < nChannels; subCh++, ch++)
						{
							for(int i=0; i<3; i++)
							{
								for(int j=0; j<3; j++)
								{
									sum[ft] += sFilter[subCh * 9 + i * 3 + j]
										* sFeature[subCh * tileDim + (blockR + i) * TILE_WIDTH + blockC + j];
								}
							}
						}
					}
				}
			}

			if(row < width && col < width)
			{
#pragma unroll 1
				for(int ft = 0; 
						ft < ITEM_PER_THREAD && ft + filterId < nFilters; 
						ft++)
				{
					fOut[(ft + filterId) * width * width + row * width + col] 
						= sum[ft] >= 0 ? sum[ft] : 0;
				}
			}

			if(threadIdx.x == 0)
			{
				blockId += blockCapacity;
				if(blockId >= totalBlocks)
				{
					isContinue = false; // TODO: this is only for debugging
				}
			}
			__syncthreads(); // TODO: not efficient
		}

		// TODO: sync with cooperate group, otherwise there might be data race
		initBlockId[layerId] = 0;
		signalIn[layerId] = 0;
		__threadfence_system();

		// triger next layer
		//while(signalIn[layerId + 1] != 0)
		while(atomicCAS_system((int*)&signalIn[layerId + 1], 0, 1) != 0)
		{
			__threadfence();
		}
		//signalIn[layerId + 1] = 1;
	}
}


void CNNPersistFunction::convPersist(int width, int nChannels, int nFilters, int layerId)
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
				&numBlocksPerSM, (void*)convBiasReluPersist<tileDepth, blockWidth, blockHeight>, 
				blockSize, sharedUsage));
	printf("numBlocksPerSM: %d\n", numBlocksPerSM);
	fflush(NULL);
	int layerSMs = SMs[layerId];
	int minSM = 0;
	for(int i=0; i<layerId; i++)
	{
		minSM += SMs[i];
	}
	int maxSM = minSM + layerSMs;
	int blockCapacity = layerSMs * numBlocksPerSM;
	int maxBlocks = numBlocksPerSM * prop.multiProcessorCount;
	int totalThreads = (width * width * nFilters + ITEM_PER_THREAD - 1) / ITEM_PER_THREAD;
	int totalBlocks = (max(totalThreads, maxBlocks * blockSize) + blockSize - 1) / blockSize;
	printf("%d, %d, %d\n", width * width * nFilters, prop.multiProcessorCount, blockSize);

	printf("totalBlocks: %d\n", totalBlocks);
	fflush(NULL);

	int nDimX = (width + blockWidth - 1) / blockWidth;
	int nDimY = (width + blockHeight - 1) / blockHeight;
	int nDimZ = (nFilters + ITEM_PER_THREAD - 1) / ITEM_PER_THREAD;

	int blockDimConv = blockWidth * blockHeight;
	int gridDimConv = totalBlocks;
	cudaStream_t stream;
	checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	
	printf("persist kernel launch now.\n");
	fflush(NULL);
#ifdef DEBUG
	signalIn[0] = 1;
	__sync_synchronize();
#endif

	convBiasReluPersist<tileDepth, blockWidth, blockHeight><<<gridDimConv, blockDimConv, 0, stream>>>(
			dInput, dFilter, nFilters, nChannels, 
			width, bias[layerId], featureOut, layerId, 
			totalBlocks, blockCapacity, minSM, maxSM,
			nDimX, nDimY, nDimZ);

	checkCudaErrors(cudaDeviceSynchronize());
	

	/*
	// activation: relu
	int blockDimAct = 256;
	int gridDimAct = (nFilters * width * width + blockDimAct - 1) / blockDimAct;
	reluForward<<<gridDimAct, blockDimAct>>>(featureOut, featureOut, nFilters * width * width);
	*/

	//checkCudaErrors(cudaFree(dInput));
}

