#include "persistFunction.h"
#include "persistInfer.h"

//#include <cooperative_groups.h>
//using namespace cooperative_groups;

// TODO: confirm this macro
#define ITEM_PER_THREAD 1

//namespace PersistInfer
//{
//	extern int volatile __managed__ signalIn[20];
//	extern int volatile __managed__ SMs[20];
//};

void CNNPersistFunction::init()
{
	CNNFunction::init();

	// using unified memory for convinent, to optimize..
	checkCudaErrors(cudaMallocManaged(&firstSignalIn, sizeof(int)*MAX_QUERY));
	checkCudaErrors(cudaMallocManaged(&lastSignalOut, sizeof(int)*MAX_QUERY));

	// NOTE: currently we use device 0 as default.
	checkCudaErrors(cudaGetDeviceProperties(&prop, 0));

	//for(int i=0; i<20; i++)
	//{
	//	checkCudaErrors(cudaMalloc(&signalIn, sizeof(int)*MAX_QUERY));
	//	checkCudaErrors(cudaMallocManaged(&SMs, sizeof(int)*MAX_QUERY));
	//}

	// TODO: assign SM here.
    if(newsignalIn == NULL)
    {
        printf("re-allocating signalIn\n");
        int * temp = NULL;
        checkCudaErrors(cudaMallocManaged(&temp, 20 * sizeof(int)));
        newsignalIn = temp;
    }
    if(newSMs == NULL)
    {
        printf("re-allocating SMs\n");
        int * temp = NULL;
        checkCudaErrors(cudaMallocManaged(&temp, 20 * sizeof(int)));
        newSMs = temp;
    }
	for(int i=0; i<20; i++)
	{
		newsignalIn[i] = 0;
		newSMs[i] = 1;
	}
	__sync_synchronize();

	for(int i=0; i<MAX_LAYER_GROUP; i++)
	{
		checkCudaErrors(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
	}
}

__device__ inline uint getSMId()
{
	uint smid;
	asm("mov.u32 %0, %smid;" : "=r" (smid));
	return smid;
}

__device__ int initBlockId[32] = {0};
__device__ int nProcessedBlocd[32] = {0};
__managed__ int nActiveBlock[20] = {0};

// TODO: using cooperate group
	template <int TILE_DEPTH, int BLOCK_WIDTH, int BLOCK_HEIGHT>
__global__ void convBiasReluPersist(float* fIn, float* filter, 
		const int nFilters, const int nChannels, const int width, 
		float* bias, float* fOut, int layerId, int totalBlocks,
		int numBlockPerSM, int blockCapacity, int minSM, int maxSM,
		int nDimX, int nDimY, int nDimZ, int* newsignalIn, int* newSMs)
{
	int smid = getSMId();
	//printf("id: %d, min: %d, max: %d\n", smid, minSM, maxSM);
	if(smid < minSM || smid >= maxSM)
	{
		// TODO: make sure #block does not exceed the maximum num on each SM
		return;
	}

	if(threadIdx.x == 0)
	{
		if(atomicAdd(&nActiveBlock[smid], 1) >= numBlockPerSM)
		{
			return;
		}
		printf("layer: %d, blockIdx.x : %d, smid: %d\n", layerId, blockIdx.x, smid);
	}

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
			int currentSignal = atomicAdd((int*)&newsignalIn[layerId], 0);
			if(currentSignal == 1)
				// if(signalIn[layerId] != 1) // 1 means current layer shoud be processed
			{
				//signalIn[layerId] = 2; // 2 means current layer is beging processing
				//__threadfence();
				blockId = atomicAdd(&initBlockId[layerId], 1);
				//printf("layer %d will be processed\n", layerId);
			}
			else
			{
				//printf("signalIn[%d] = %d\n", layerId, signalIn[layerId]);
				blockId = -1;
				//continue;
			}
		}
		__syncthreads();
		if(blockId == -1)
		{
			continue;
		}
		//printf("read data for layer %d\n", layerId);
		// blockId is the virtual global block ID for current layer
#pragma unroll 1
		for(int bid = blockId; bid < totalBlocks; bid += blockCapacity)
		{
			// blockIdx_x/y/z are virtual blockIdx.x/y/z
			int blockIdx_x = bid % nDimX;
			int blockIdx_y = (bid / nDimX) % nDimY;
			int blockIdx_z = bid / (nDimX * nDimY); // to get filter id

			// row/col in the input feature map 
			int row = blockIdx_y * BLOCK_HEIGHT + blockR;
			int col = blockIdx_x * BLOCK_WIDTH + blockC;

			int filterId = blockIdx_z * ITEM_PER_THREAD;
			if(threadIdx.x == 0)
			{
				//printf("bid: %d/%d\n", bid , totalBlocks); // for debugging
				//printf("bid: %d, blockIdx_x: %d, blockIdx_y: %d, blockIdx_z: %d, filterId: %d\n",
				//		bid, blockIdx_x, blockIdx_y, blockIdx_z, filterId);
			}

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
						= sum[ft] >= 0 ? sum[ft] : 0; // relu
				}
				__threadfence();
			}

			if(threadIdx.x == 0)
			{
				atomicAdd(&nProcessedBlocd[layerId], 1);
			}

			//__syncthreads(); // TODO: not efficient
		}

		__syncthreads();
		if(threadIdx.x == 0)
		{
			isContinue = 0; // TODO: this is only for debugging
		}

		if(blockId == 0 && threadIdx.x == 0) // first thread in first active block
		{
			while(atomicCAS(&nProcessedBlocd[layerId], totalBlocks, 0) != totalBlocks)
			{
				__threadfence();
			}

			initBlockId[layerId] = 0;
			newsignalIn[layerId] = 0;
			__threadfence_system();

			while(atomicCAS_system((int*)&newsignalIn[layerId + 1], 0, 1) != 0)
			{
				__threadfence();
			}
		}

		// TODO: sync with cooperate group, otherwise there might be data race
		//grid_group grid = this_grid();
		//grid.sync();

		// triger next layer
		//while(signalIn[layerId + 1] != 0)
		//if(false) // TODO: check whether all blocks are processed
		//{
		//}
	}
}

template <int TILE_DEPTH, int BLOCK_WIDTH, int BLOCK_HEIGHT, int FILTER_NUM, int CHANNEL_NUM>
__global__ void convBiasReluPersistReuseSharedMem(float* fIn, float* filter, 
		const int nFilters, const int nChannels, const int width, 
		float* bias, float* fOut, int layerId, int totalBlocks,
		int numBlockPerSM, int blockCapacity, int minSM, int maxSM,
		int nDimX, int nDimY, int nDimZ, int* newsignalIn, int* newSMs)
{
	int smid = getSMId();
	//printf("id: %d, min: %d, max: %d\n", smid, minSM, maxSM);
	if(smid < minSM || smid >= maxSM)
	{
		// TODO: make sure #block does not exceed the maximum num on each SM
		return;
	}

	if(threadIdx.x == 0)
	{
		if(atomicAdd(&nActiveBlock[smid], 1) >= numBlockPerSM)
		{
			return;
		}
		printf("layer: %d, blockIdx.x : %d, smid: %d\n", layerId, blockIdx.x, smid);
	}

	int const TILE_WIDTH = BLOCK_WIDTH + 2;
	int const TILE_HEIGHT = BLOCK_HEIGHT + 2;
	int const blockDim = BLOCK_WIDTH * BLOCK_HEIGHT;
	int const tileDim = TILE_WIDTH * TILE_HEIGHT;

	__shared__ float sFilter[3 * 3 * FILTER_NUM * CHANNEL_NUM];
	__shared__ float sFeature[TILE_DEPTH * tileDim];

	// row/col in the block
	int blockR = threadIdx.x / BLOCK_WIDTH;
	int blockC = threadIdx.x % BLOCK_WIDTH;

	int split = (nChannels + TILE_DEPTH - 1) / TILE_DEPTH;

	if(threadIdx.x == 0)
	{
		for(int i = 0; i < 3*3*FILTER_NUM * CHANNEL_NUM; ++i)
		{
			sFilter[i] = filter[i];
		}
	}

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
			int currentSignal = atomicAdd((int*)&newsignalIn[layerId], 0);
			if(currentSignal == 1)
				// if(signalIn[layerId] != 1) // 1 means current layer shoud be processed
			{
				//signalIn[layerId] = 2; // 2 means current layer is beging processing
				//__threadfence();
				blockId = atomicAdd(&initBlockId[layerId], 1);
				//printf("layer %d will be processed\n", layerId);
			}
			else
			{
				//printf("signalIn[%d] = %d\n", layerId, signalIn[layerId]);
				blockId = -1;
				//continue;
			}
		}
		__syncthreads();
		if(blockId == -1)
		{
			continue;
		}
		//printf("read data for layer %d\n", layerId);
		// blockId is the virtual global block ID for current layer


			


#pragma unroll 1
		for(int bid = blockId; bid < totalBlocks; bid += blockCapacity)
		{
			// blockIdx_x/y/z are virtual blockIdx.x/y/z
			int blockIdx_x = bid % nDimX;
			int blockIdx_y = (bid / nDimX) % nDimY;
			int blockIdx_z = bid / (nDimX * nDimY); // to get filter id

			// row/col in the input feature map 
			int row = blockIdx_y * BLOCK_HEIGHT + blockR;
			int col = blockIdx_x * BLOCK_WIDTH + blockC;

			int filterId = blockIdx_z * ITEM_PER_THREAD;
			if(threadIdx.x == 0)
			{
				//printf("bid: %d/%d\n", bid , totalBlocks); // for debugging
				//printf("bid: %d, blockIdx_x: %d, blockIdx_y: %d, blockIdx_z: %d, filterId: %d\n",
				//		bid, blockIdx_x, blockIdx_y, blockIdx_z, filterId);
			}


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

				__syncthreads();

#pragma unroll 1
				for(int ft = 0; 
						ft < ITEM_PER_THREAD && ft + filterId < nFilters; 
						ft++)
				{

					// put filter into shared memory
// #pragma unroll 1
// 					for (int iter = 0; 
// 							iter <= TILE_DEPTH * 9 / blockDim; 
// 							iter++)
// 					{
// 						int idx = threadIdx.x + iter * blockDim;
// 						if(idx < TILE_DEPTH * 9)
// 						{
// 							sFilter[idx] = filter[((ft + filterId) * nChannels + sp * TILE_DEPTH) * 9 + idx];
// 						}
// 					}
//
//					 __syncthreads();
//
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
                                    //if(sFilter[subCh * 9 + i * 3 + j] != filter[((ft + filterId) * nChannels + sp * TILE_DEPTH) * 9 + subCh * 9 + i * 3 + j])
                                    //{
                                    //    printf("filter diff %f %f\n", sFilter[subCh * 9 + i * 3 + j] , filter[((ft + filterId) * nChannels + sp * TILE_DEPTH) * 9 + subCh * 9 + i * 3 + j]);
                                    //}
                                   // if(filter[((ft + filterId) * nChannels + sp * TILE_DEPTH) * 9 + subCh * 9 + i * 3 + j] != sFilter[((ft + filterId) * nChannels + sp * TILE_DEPTH) * 9 + subCh * 9 + i * 3 + j]){
                                   //     printf("pos=%d\n", ((ft + filterId) * nChannels + sp * TILE_DEPTH) * 9 + subCh * 9 + i * 3 + j);
                                   //     printf("filter diff %f %f\n", filter[((ft + filterId) * nChannels + sp * TILE_DEPTH) * 9 + subCh * 9 + i * 3 + j], sFilter[((ft + filterId) * nChannels + sp * TILE_DEPTH) * 9 + subCh * 9 + i * 3 + j]);
                                   // }
									sum[ft] += /*sFilter[subCh * 9 + i * 3 + j]*/sFilter[((ft + filterId) * nChannels + sp * TILE_DEPTH) * 9 + subCh * 9 + i * 3 + j]
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
						= sum[ft] >= 0 ? sum[ft] : 0; // relu
				}
				__threadfence();
			}

			if(threadIdx.x == 0)
			{
				atomicAdd(&nProcessedBlocd[layerId], 1);
			}

			//__syncthreads(); // TODO: not efficient
		}

		__syncthreads();
		if(threadIdx.x == 0)
		{
			isContinue = 0; // TODO: this is only for debugging
		}

		if(blockId == 0 && threadIdx.x == 0) // first thread in first active block
		{
			while(atomicCAS(&nProcessedBlocd[layerId], totalBlocks, 0) != totalBlocks)
			{
				__threadfence();
			}

			initBlockId[layerId] = 0;
			newsignalIn[layerId] = 0;
			__threadfence_system();

			while(atomicCAS_system((int*)&newsignalIn[layerId + 1], 0, 1) != 0)
			{
				__threadfence();
			}
			// printf("cas layer %d to %d\n", layerId + 1, newsignalIn[layerId+1]);
		}

		// TODO: sync with cooperate group, otherwise there might be data race
		//grid_group grid = this_grid();
		//grid.sync();

		// triger next layer
		//while(signalIn[layerId + 1] != 0)
		//if(false) // TODO: check whether all blocks are processed
		//{
		//}
	}
}



void CNNPersistFunction::convPersist(int width, int nChannels, int nFilters, int layerId)
{
	std::size_t inputSize = width * width * nChannels * sizeof(float);
	float* dInput = featureMap[layerId];
	float* dOutput = layerId == 18 ? featureOut : featureMap[layerId + 1];
	float *dFilter = weights[layerId];
	//checkCudaErrors(cudaMemcpyAsync(dInput, featureOut, inputSize, cudaMemcpyDefault, streams[layerId]));

	//float* dInput = nullptr; //featureMap[layerId];
	//checkCudaErrors(cudaMalloc(&dInput, inputSize));
	//checkCudaErrors(cudaMemcpy(dInput, featureOut, inputSize, cudaMemcpyDefault));

	//checkCudaErrors(cudaDeviceSynchronize()); // for debugging

	const int tileDepth = 4;
	const int blockWidth = 32;
	const int blockHeight = 8;
	const int blockSize = blockWidth * blockHeight;
	const int tempFilterNum = 64;

	// numblocks per sm
	int numBlocksPerSM = 0;
	int sharedUsage = sizeof(float) * tileDepth * 
		(3 * 3 + (blockWidth + 2) * (blockHeight + 2));
	checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
				&numBlocksPerSM, (void*)convBiasReluPersistReuseSharedMem<tileDepth, blockWidth, blockHeight, tempFilterNum, 3>, 
				blockSize, sharedUsage));

	printf("numBlocksPerSM: %d, numSMs: %d, for layer %d\n", 
			numBlocksPerSM, prop.multiProcessorCount, layerId);
	fflush(NULL);

	int layerSMs = newSMs[layerId];
	layerSMs = 20;
	printf("kernel using sm num=%d\n", layerSMs);
	int minSM = 0;
	for(int i=0; i<layerId; i++)
	{
		minSM += newSMs[i];
	}
	int maxSM = minSM + layerSMs;
	int blockCapacity = layerSMs * numBlocksPerSM;
	int maxBlocks = numBlocksPerSM * prop.multiProcessorCount;

	int nDimX = (width + blockWidth - 1) / blockWidth;
	int nDimY = (width + blockHeight - 1) / blockHeight;
	int nDimZ = (nFilters + ITEM_PER_THREAD - 1) / ITEM_PER_THREAD;

	int totalBlocks = nDimX * nDimY * nDimZ;

	printf("%d, %d, %d\n", width * width * nFilters, prop.multiProcessorCount, blockSize);
	printf("totalBlocks: %d\n", totalBlocks);
	fflush(NULL);

	int blockDimConv = blockWidth * blockHeight;
	//int gridDimConv = totalBlocks;
	int gridDimConv = maxBlocks * 2; // more blocks to guarantee each SM have enough blocks

	printf("persist kernel of layer %d launch now.\n", layerId);
	fflush(NULL);
//#ifdef DEBUG
//	if(layerId == 0)
//	{
//		PersistInfer::signalIn[0] = 1;
//	}
//	__sync_synchronize();
//#endif


	//void *params[] =
	//{
	//	&dInput, &dFilter, &nFilters, &nChannels, 
	//	&width, &bias[layerId], &dOutput, &layerId, 
	//	&totalBlocks, &blockCapacity, &minSM, &maxSM,
	//	&nDimX, &nDimY, &nDimZ
	//};
	//checkCudaErrors(cudaLaunchCooperativeKernel(
	//			(void*)convBiasReluPersist<tileDepth, blockWidth, blockHeight>,
	//			gridDimConv, blockDimConv, params, 0, stream));

	//convBiasReluPersist<tileDepth, blockWidth, blockHeight>
	convBiasReluPersistReuseSharedMem<tileDepth, blockWidth, blockHeight, tempFilterNum, 3>
		<<<gridDimConv, blockDimConv, 0, streams[layerId]>>>(
			dInput, dFilter, nFilters, nChannels, 
			width, bias[layerId], dOutput, layerId, 
			totalBlocks, numBlocksPerSM, blockCapacity, minSM, maxSM,
			nDimX, nDimY, nDimZ, newsignalIn, newSMs);

	//checkCudaErrors(cudaDeviceSynchronize());

	printf("persist kernel call of layer %d ends async\n", layerId);

	//checkCudaErrors(cudaFree(dInput));
}

