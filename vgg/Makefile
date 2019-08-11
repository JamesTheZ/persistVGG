CC = g++
NVCC = nvcc
NVFLAGS = -O3 -std=c++11 -I. -I./inc -gencode arch=compute_61,code=sm_61 --use_fast_math --ptxas-options=-v -lineinfo #-g -G 
CXXFLAGS = -O3 -Wunused-result
LDFLAGS = -lm -lcublas -lcusparse -lcudnn
TARGETS = vgg19 

DEBUGFLAGS = -DDEBUG

SRC = vgg19.cu cudaFunction.cu cudnnFunction.cu cublasFunction.cu function.cu util.cu persistFunction.cu

all: $(TARGETS)

vgg19:$(SRC) Makefile
	$(NVCC) $(NVFLAGS) ${DEBUGFLAGS} -Xcompiler="$(CXXFLAGS)" $(LDFLAGS) -o $@ $(SRC)

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

runprof: vgg19
	nvprof ./vgg19 image/cat1.txt vgg19_weight/vgg19_weight.txt vgg19_weight/vgg19_bias.txt vgg19_output_1000.txt

runmemcheck:vgg19
	cuda-memcheck ./vgg19 image/cat1.txt vgg19_weight/vgg19_weight.txt vgg19_weight/vgg19_bias.txt vgg19_output_1000.txt
