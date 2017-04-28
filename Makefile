CXXFLAGS += -std=c++14 -I ../ -L ../ebt -L ../la -L ../autodiff -L ../opt -L ../speech
NVCCFLAGS += -std=c++11 -I ../ -L ../ebt -L ../la -L ../autodiff -L ../opt -L ../speech
AR = gcc-ar

obj = nn.o cnn.o tensor-tree.o lstm.o lstm-frame.o lstm-tensor-tree.o rsg.o

.PHONY: all clean

all: libnn.a

gpu: libnngpu.a

clean:
	-rm *.o
	-rm libnn.a

libnn.a: $(obj)
	$(AR) rcs $@ $^

libnngpu.a: $(obj) tensor-tree-gpu.o
	$(AR) rcs $@ $^

nn.o: nn.h
cnn.o: cnn.h
lstm.o: lstm.h
lstm-seg.o: lstm-seg.h
gru.o: gru.h
pred.o: pred.h
residual.o: residual.h
tensor-tree.o: tensor-tree.h

tensor-tree-gpu.o: tensor-tree-gpu.cu
	nvcc $(NVCCFLAGS) -c tensor-tree-gpu.cu
