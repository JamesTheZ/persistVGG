CC = g++
NVCC = nvcc
NVFLAGS = -O0 -std=c++11 -I. -I./inc -g -G #-arch=sm_30
CXXFLAGS = -O0 -Wunused-result
LDFLAGS = -lm -lcublas -lcusparse -lcudnn
TARGETS = vgg19 

all: $(TARGETS)

vgg19: vgg19.cu cudnnFunction.cu cublasFunction.cu function.cu
	$(NVCC) $(NVFLAGS) -Xcompiler="$(CXXFLAGS)" $(LDFLAGS) -o $@ vgg19.cu cudnnFunction.cu cublasFunction.cu function.cu

%: %.cpp
	$(CC) $(CXXFLAGS) $(LDFLAGS) -o $@ $?

vgg: vgg19.py
	python vgg19.py

image: image_converter.py
	python image_converter.py

softmax: softmax.py
	python softmax.py

clean:
	rm -rf $(TARGETS)

run: vgg19
	./vgg19 image/cat1.txt vgg19_weight/vgg19_weight.txt vgg19_weight/vgg19_bias.txt vgg19_output_1000.txt
