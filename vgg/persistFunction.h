#ifndef CNN_CUDA_FUNCTION_H
#define CNN_CUDA_FUNCTION_H

#include "cudnn.h"
#include "cublas_v2.h"
#include "function.h"

#define MAX_QUERY 32

class CNNPersistFunction : public CNNFunction
{
	private:
		const float alpha = 1.0f;
		const float beta = 0.0f;

		// each has 32 vals
		int* firstSignalIn;
		int* lastSignalOut;
		int* signalIn[19]; // 19 layers maximum
		int* SMs[19];
		cudaDeviceProp prop;

	public:
		virtual void init() override;
		//virtual void convolution(int width, int nChannels, int nFilters, int layerId) override;
		//virtual void fullyConnected(int width, int nChannels, int nFilters, int layerId) override;
		//virtual void maxpool(int width, int nChannels) override;

		void convPersist(int width, int nChannels, int nFilters, int layerId);
		void maxpoolPersist(int width, int nChannels, int id);

	//private:
	//	void relu(float* dArray, int size);

};

#endif
