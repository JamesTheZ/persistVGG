#ifndef CNN_FUNCTION_H
#define CNN_FUNCTION_H

#include "helper_cuda.h"

// weights & bias size: (filter size * channels + 1 bias) * #filters
const int conv1_1_w = (3 * 3 * 3    + 1) * 64;
const int conv1_2_w = (3 * 3 * 64   + 1) * 64;
const int conv2_1_w = (3 * 3 * 64   + 1) * 128;
const int conv2_2_w = (3 * 3 * 128  + 1) * 128;
const int conv3_1_w = (3 * 3 * 128  + 1) * 256;
const int conv3_2_w = (3 * 3 * 256  + 1) * 256;
const int conv3_3_w = (3 * 3 * 256  + 1) * 256;
const int conv3_4_w = (3 * 3 * 256  + 1) * 256;
const int conv4_1_w = (3 * 3 * 256  + 1) * 512;
const int conv4_2_w = (3 * 3 * 512  + 1) * 512;
const int conv4_3_w = (3 * 3 * 512  + 1) * 512;
const int conv4_4_w = (3 * 3 * 512  + 1) * 512;
const int conv5_1_w = (3 * 3 * 512  + 1) * 512;
const int conv5_2_w = (3 * 3 * 512  + 1) * 512;
const int conv5_3_w = (3 * 3 * 512  + 1) * 512;
const int conv5_4_w = (3 * 3 * 512  + 1) * 512;
const int fc1_w     = (7 * 7 * 512  + 1) * 4096;
const int fc2_w     = (1 * 1 * 4096 + 1) * 4096;
const int fc3_w     = (1 * 1 * 4096 + 1) * 1000;
// layer output size
const int conv1_1  = 224 * 224 * 64;
const int conv1_2  = 224 * 224 * 64;
const int maxpool1 = 112 * 112 * 64;
const int conv2_1  = 112 * 112 * 128;
const int conv2_2  = 112 * 112 * 128;
const int maxpool2 = 56  * 56  * 128;
const int conv3_1  = 56  * 56  * 256;
const int conv3_2  = 56  * 56  * 256;
const int conv3_3  = 56  * 56  * 256;
const int conv3_4  = 56  * 56  * 256;
const int maxpool3 = 28  * 28  * 256;
const int conv4_1  = 28  * 28  * 512;
const int conv4_2  = 28  * 28  * 512;
const int conv4_3  = 28  * 28  * 512;
const int conv4_4  = 28  * 28  * 512;
const int maxpool4 = 14  * 14  * 512;
const int conv5_1  = 14  * 14  * 512;
const int conv5_2  = 14  * 14  * 512;
const int conv5_3  = 14  * 14  * 512;
const int conv5_4  = 14  * 14  * 512;
const int maxpool5 = 7   * 7   * 512;
const int fc1      = 1   * 1   * 4096;
const int fc2      = 1   * 1   * 4096;
const int fc3      = 1   * 1   * 1000;

const int filterSize[19] = 
{
	3*3,
	3*3,
	3*3,
	3*3,
	3*3,
	3*3,
	3*3,
	3*3,
	3*3,
	3*3,
	3*3,
	3*3,
	3*3,
	3*3,
	3*3,
	3*3,
	7*7,
	1*1,
	1*1
};

const int channels[19] = 
{
	3,
	64,
	64,
	128,
	128,
	256,
	256,
	256,
	256,
	512,
	512,
	512,
	512,
	512,
	512,
	512,
	512,
	4096,
	4096
};

const int numFilters[19] = 
{
	64,
	64,
	128,
	128,
	256,
	256,
	256,
	256,
	512,
	512,
	512,
	512,
	512,
	512,
	512,
	512,
	4096,
	4096,
	1000
};

class CNNFunction
{
	public:
		float image[224 * 224 * 3];
		float *featureOut = nullptr;
		float *weights[19];
		float *bias[19];
		float *parameters[19]; // fuse weights and bias together

	public:
		virtual void init();

		virtual void readImage(char *imageFile);
		virtual void readParameters(char *weightsFile, char *biasFile);
		virtual void writeOutput(char *output_file);

		virtual void convolution(int width, int channels, int num_filters, int layerId);
		virtual void fullyConnected(int width, int channels, int num_filters, int layerId);
		virtual void maxpool(int width, int channels);
};

#endif
