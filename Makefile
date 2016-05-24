CXXFLAGS += -std=c++11 -I ../ -L ../ebt -L ../la -L ../autodiff -L ../opt -L ../speech
AR = gcc-ar

.PHONY: all clean gpu

bin = \
    learn \
    predict \
    learn-lstm \
    predict-lstm \
    learn-gru \
    predict-gru \
    learn-residual \
    libnn.a

all: $(bin)

gpu: learn-gpu learn-lstm-gpu libnngpu.a

clean:
	-rm *.o
	-rm $(bin)
	-rm learn-lstm-gpu libnngpu.a

libnn.a: nn.o lstm.o pred.o residual.o
	$(AR) rcs $@ $^

learn: nn.o learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lla -lebt -lblas

predict: nn.o predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lla -lebt -lblas

learn-lstm: lstm.o learn-lstm.o pred.o nn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lspeech -lopt -lla -lebt -lblas

predict-lstm: lstm.o predict-lstm.o pred.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lspeech -lopt -lla -lebt -lblas

learn-gru: gru.o learn-gru.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lspeech -lopt -lla -lebt -lblas

predict-gru: gru.o predict-gru.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lspeech -lopt -lla -lebt -lblas

learn-residual: learn-residual.o residual.o pred.o nn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lspeech -lopt -lla -lebt -lblas

nn.o: nn.h
lstm.o: lstm.h
gru.o: gru.h
pred.o: pred.h
residual.o: residual.h

libnngpu.a: nn.o nn-gpu.o
	$(AR) rcs $@ $^

nn-gpu.o: nn-gpu.cu
	nvcc $(CXXFLAGS) -c nn-gpu.cu

lstm-gpu.o: lstm-gpu.cu
	nvcc $(CXXFLAGS) -c lstm-gpu.cu

learn-gpu.o: learn-gpu.cu
	nvcc $(CXXFLAGS) -c learn-gpu.cu

learn-gpu: learn-gpu.o nn-gpu.o nn.o
	$(CXX) $(CXXFLAGS) -L /opt/cuda/lib64 -o $@ $^ -lautodiffgpu -loptgpu -llagpu -lblas -lebt -lcublas -lcudart

learn-lstm-gpu.o: learn-lstm-gpu.cu
	nvcc $(CXXFLAGS) -c learn-lstm-gpu.cu

learn-lstm-gpu: learn-lstm-gpu.o lstm-gpu.o lstm.o
	$(CXX) $(CXXFLAGS) -L /opt/cuda/lib64 -o $@ $^ -lautodiffgpu -loptgpu -lspeech -llagpu -lblas -lebt -lcublas -lcudart

nn-gpu.o: nn-gpu.h nn.h
lstm-gpu.o: lstm-gpu.h
learn-gpu.o: nn-gpu.h
learn-lstm-gpu.o: lstm-gpu.h
