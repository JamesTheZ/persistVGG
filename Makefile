all: vgg

vgg: vgg.cu
	nvcc vgg.cu -O3 -o vgg

clean:
	rm vgg
