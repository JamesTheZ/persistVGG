#ifndef CNN_PERSIST_FUNCTION_H
#define CNN_PERSIST_FUNCTION_H

#include "cudnn.h"
#include "cublas_v2.h"
#include "function.h"
#include "persistInfer.h"

#define MAX_QUERY 32

class CNNPersistFunction : public CNNFunction
{
	private:
		const float alpha = 1.0f;
		const float beta = 0.0f;

		// each has 32 vals
		int* firstSignalIn;
		int* lastSignalOut;
		//int volatile signalIn[20]; // 19 layers maximum
		//int volatile SMs[20];
		cudaDeviceProp prop;

		cudaStream_t streams[MAX_LAYER_GROUP];

	public:
		virtual void init() override;
		//virtual void convolution(int width, int nChannels, int nFilters, int layerId) override;
		//virtual void fullyConnected(int width, int nChannels, int nFilters, int layerId) override;
		//virtual void maxpool(int width, int nChannels) override;

		void convPersist(int width, int nChannels, int nFilters, int layerId) override;
		void maxpoolPersist(int width, int nChannels, int id);

	//private:
	//	void relu(float* dArray, int size);

};

#endif
