#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cublas_v2.h>

#include "cublasFunction.h"

int main(int argc, char **argv)
{
    char *image_file = argv[1];
    char *weights_file = argv[2];
    char *bias_file = argv[3];
    char *output_file = argv[4];

	CNNFunction *func = new CNNCublasFunction();

	func->init();
	func->readImage(image_file);
	func->readParameters(weights_file, bias_file);

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

    // write 1000 dimension
    func->writeOutput(output_file);

    return 0;
}


//FILE *fw;
//FILE *fb;
//cublasHandle_t cubHandle;
//// for cublas dummy constant
//const float alpha = 1.0f;
//const float beta = 0.0f;
//
//// required to normalize by mean pixel (in rgb order)
//float mean_pixel[3] = {123.68, 116.779, 103.939};
//// input image
//float image[224 * 224 * 3];
//// ouput of each layer, device pointer
//float *d_output;

//__global__ void maxpooling(float *output, const float *input, const int width, const int channels)
//{
//    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
//    int new_width = width / 2;
//    int i = thread_id / new_width * 2;
//    int j = thread_id % new_width * 2;
//    int index = i * width + j;
//
//    for (int c = 0; c < channels; c++) {
//        float max = 0;
//        if (max < input[index * channels + c])
//            max = input[index * channels + c];
//        if (max < input[(index + 1) * channels + c])
//            max = input[(index + 1) * channels + c];
//        if (max < input[(index + width) * channels + c])
//            max = input[(index + width) * channels + c];
//        if (max < input[(index + width + 1) * channels + c])
//            max = input[(index + width + 1) * channels + c];
//        output[thread_id * channels + c] = max;
//    }
//}

//__global__ void transform_image(float *input, const float *raw_input, const int width, const int channels)
//{
//    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
//    int start_i = thread_id / width - 1;
//    int start_j = thread_id % width - 1;
//    int per_channel_width = width * width;
//    int hidden_width = 3 * 3 * channels + 1;
//    int global_offset = thread_id * hidden_width;
//
//    for (int c = 0; c < channels; c++) {
//        int offset = 0;
//        for (int i = start_i; i < start_i + 3; i++) {
//            if (i < 0 || i == width)
//                continue;
//            for (int j = start_j; j < start_j + 3; j++) {
//                if (j < 0 || j == width)
//                    continue;
//                input[global_offset + c * 9 + offset] = raw_input[c * per_channel_width + i * width + j];
//                offset++;
//            }
//        }
//    }
//    input[(thread_id + 1) * hidden_width - 1] = 1;
//}

//__global__ void transform_fc(float *input, const float *raw_input, const int width, const int channels)
//{
//    int thread_id = threadIdx.x;
//    int size = width * width;
//
//    for (int s = 0; s < size; s++)
//        input[thread_id * size + s] = raw_input[s * channels + thread_id];
//    if (thread_id == 0)
//        input[width * width * channels] = 1;
//}

//__global__ void transform(float *input, const float *raw_input, const int width, const int channels)
//{
//    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
//    int start_i = thread_id / width - 1;
//    int start_j = thread_id % width - 1;
//    int hidden_width = 3 * 3 * channels + 1;
//    int global_offset = thread_id * hidden_width;
//
//    float relu;
//    for (int c = 0; c < channels; c++) {
//        int offset = 0;
//        for (int i = start_i; i < start_i + 3; i++) {
//            if (i < 0 || i == width)
//                continue;
//            for (int j = start_j; j < start_j + 3; j++) {
//                if (j < 0 || j == width)
//                    continue;
//                relu = raw_input[(i * width + j) * channels + c];
//                input[global_offset + c * 9 + offset] = relu < 0 ? 0 : relu;
//                offset++;
//            }
//        }
//    }
//    input[(thread_id + 1) * hidden_width - 1] = 1;
//}

//void fully_connected(int width, int channels, int num_filters)
//{
//    int num_weights = (width * width * channels + 1) * num_filters;
//    int filter_size = width * width * channels;
//    int hidden_width = filter_size + 1;
//    float *weights = (float *)malloc(num_weights * sizeof(float));
//    for (int i = 0; i < num_filters; i++) {
//        for (int j = 0; j < filter_size; j++)
//            fscanf(fw, "%f", &weights[i * hidden_width + j]);
//        fscanf(fb, "%f", &weights[i * hidden_width + filter_size]);
//    }
//
//    float *d_input;
//    size_t input_size = (width * width * channels + 1) * sizeof(float);
//    checkCudaErrors(cudaMalloc(&d_input, input_size));
//    if (width == 1) {
//        // previous output vector (channels * 1), expand to ((channels + 1) * 1) with a 1 at last
//        float *output = (float *)malloc((channels + 1) * sizeof(float));
//        checkCudaErrors(cudaMemcpy(output, d_output, channels * sizeof(float), cudaMemcpyDeviceToHost));
//        output[channels] = 1;
//        checkCudaErrors(cudaMemcpy(d_input, output, (channels + 1) * sizeof(float), cudaMemcpyHostToDevice));
//        free(output);
//    }
//    else {
//        // only the first fc needs to transform previous output to a vector (width * width * channels)
//        transform_fc <<< 1, channels >>> (d_input, d_output, width, channels);
//        checkCudaErrors(cudaDeviceSynchronize());
//    }
//
//    float *d_weights;
//    checkCudaErrors(cudaMalloc(&d_weights, num_weights * sizeof(float)));
//    cudaFree(d_output);
//    checkCudaErrors(cudaMalloc(&d_output, num_filters * sizeof(float)));
//    checkCudaErrors(cublasSetMatrix(hidden_width, num_filters, sizeof(float), weights, hidden_width, d_weights, hidden_width));
//    // weights * input = (num_filters * (channels + 1)) * ((channels + 1) * 1), consider vector as matrix
//    checkCudaErrors(cublasSgemm(cubHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1, num_filters, hidden_width,
//                            &alpha, d_input, 1, d_weights, hidden_width,
//                            &beta, d_output, 1));
//
//    free(weights);
//    cudaFree(d_input);
//    cudaFree(d_weights);
//}

//void maxpool(int width, int channels)
//{
//    float *d_temp;
//    size_t mem_size = width * width * channels * sizeof(float);
//    checkCudaErrors(cudaMalloc(&d_temp, mem_size));
//    checkCudaErrors(cudaMemcpy(d_temp, d_output, mem_size, cudaMemcpyDeviceToDevice));
//    cudaFree(d_output);
//    checkCudaErrors(cudaMalloc(&d_output, mem_size / 4));
//    maxpooling <<< width / 2, width / 2 >>> (d_output, d_temp, width, channels);
//    checkCudaErrors(cudaDeviceSynchronize());
//}

// 'num_filters' of former layer is 'channels' of latter layer
//void convolution(int width, int channels, int num_filters)
//{
//    int num_weights = (3 * 3 * channels + 1) * num_filters;
//    int output_size = width * width * num_filters;
//    int filter_size = 3 * 3 * channels;
//    int hidden_width = 3 * 3 * channels + 1;
//    float *weights = (float *)malloc(num_weights * sizeof(float));
//    for (int i = 0; i < num_filters; i++) {
//        for (int j = 0; j < filter_size; j++)
//            fscanf(fw, "%f", &weights[j * num_filters + i]); // column major
//        fscanf(fb, "%f", &weights[filter_size * num_filters + i]);
//    }
//
//    float *d_raw_input;
//    float *d_input;
//    size_t input_size = width * width * hidden_width * sizeof(float);
//    checkCudaErrors(cudaMalloc(&d_input, input_size));
//    checkCudaErrors(cudaMemset(d_input, 0, input_size));
//    // expand original input to (width * width) * (3 * 3 * channels + 1) with a 1 at last for bias
//    if (channels == 3) {
//        size_t raw_input_size = width * width * channels * sizeof(float);
//        checkCudaErrors(cudaMalloc(&d_raw_input, raw_input_size));
//        checkCudaErrors(cudaMemcpy(d_raw_input, image, raw_input_size, cudaMemcpyHostToDevice));
//        transform_image <<< width, width >>> (d_input, d_raw_input, width, channels);
//    }
//    else 
//	{
//		// d_output has width*width rows and channels cols.
//        transform <<< width, width >>> (d_input, d_output, width, channels);
//	}
//    checkCudaErrors(cudaDeviceSynchronize());
//
//    float *d_weights;
//    checkCudaErrors(cudaMalloc(&d_weights, num_weights * sizeof(float)));
//    cudaFree(d_output);
//    checkCudaErrors(cudaMalloc(&d_output, output_size * sizeof(float)));
//    checkCudaErrors(cublasSetMatrix(num_filters, hidden_width, sizeof(float), weights, num_filters, d_weights, num_filters));
//    // input * weights = ((width * width) * (3 * 3 * channels + 1)) * ((3 * 3 * channels + 1) * num_filters)
//    checkCudaErrors(cublasSgemm(cubHandle, CUBLAS_OP_N, CUBLAS_OP_N, num_filters, width * width, hidden_width,
//                            &alpha, d_weights, num_filters, d_input, hidden_width,
//                            &beta, d_output, num_filters));
//	// d_output has width*width rows and num_filters cols.
//
//    free(weights);
//    if (channels == 3)
//        cudaFree(d_raw_input);
//    cudaFree(d_input);
//    cudaFree(d_weights);
//}

// debug use, print out each element of output after a layer
//void debug_print(int width, int channels)
//{
//    int output_size = width * width * channels;
//    float *output = (float *)malloc(output_size * sizeof(float));
//    checkCudaErrors(cublasGetMatrix(num_filters, width * width, sizeof(float), d_output, num_filters, output, num_filters));
//    for (int i = 0; i < channels; i++) {
//        for (int j = 0; j < width * width; j++)
//            printf("%f ", output[j * channels + i]);
//        printf("\n");
//    }
//    free(output);
//}

//void write_output(char *output_file)
//{
//    FILE *fout = fopen(output_file, "w");
//
//    float *output = (float *)malloc(1000 * sizeof(float));
//    checkCudaErrors(cudaMemcpy(output, d_output, 1000 * sizeof(float), cudaMemcpyDeviceToHost));
//
//    for (int i = 0; i < 1000; i++)
//        fprintf(fout, "%f\n", output[i]);
//
//    free(output);
//    cudaFree(d_output);
//    fclose(fout);
//}
//
//void read_image(char *image_file)
//{
//    FILE *fin = fopen(image_file, "r");
//    int total = 224 * 224 * 3;
//    for (int index = 0; index < total; index++) {
//        fscanf(fin, "%f", &image[index]);
//        image[index] -= mean_pixel[index / 50176]; // 50176 = 224 * 224
//    }
//    fclose(fin);
//}


/*
int main(int argc, char **argv)
{
    char *image_file = argv[1];
    char *weights_file = argv[2];
    char *bias_file = argv[3];
    char *output_file = argv[4];

    // read image file
    read_image(image_file);

    // initialize
    fw = fopen(weights_file, "r");
    fb = fopen(bias_file, "r");
    checkCudaErrors(cublasCreate(&cubHandle));

    // ReLU layers in transform kernel or maxpooling
    // read file input in each layer beginning to save memory cost
    convolution(224, 3, 64);
    convolution(224, 64, 64);
    maxpool(224, 64);
    convolution(112, 64, 128);
    convolution(112, 128, 128);
    maxpool(112, 128);
    convolution(56, 128, 256);
    convolution(56, 256, 256);
    convolution(56, 256, 256);
    convolution(56, 256, 256);
    maxpool(56, 256);
    convolution(28, 256, 512);
    convolution(28, 512, 512);
    convolution(28, 512, 512);
    convolution(28, 512, 512);
    maxpool(28, 512);
    convolution(14, 512, 512);
    convolution(14, 512, 512);
    convolution(14, 512, 512);
    convolution(14, 512, 512);
    maxpool(14, 512);
    fully_connected(7, 512, 4096); // most time consuming file input
    fully_connected(1, 4096, 4096);
    fully_connected(1, 4096, 1000);

    // write 1000 dimension
    write_output(output_file);

    fclose(fw);
    fclose(fb);
    checkCudaErrors(cublasDestroy(cubHandle));

    return 0;
}
*/
