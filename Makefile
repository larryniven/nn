CXXFLAGS += -std=c++11 -I ../ -L ../ebt -L ../la -L ../autodiff -L ../opt -L ../speech
AR = gcc-ar

.PHONY: all clean gpu

all: learn predict learn-rnn predict-rnn libnn.a

gpu: learn-gpu libnngpu.a

clean:
	-rm *.o
	-rm learn predict learn-rnn predict-rnn learn-gpu

learn: nn.o learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lla -lebt -lblas

predict: nn.o predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lla -lebt -lblas

learn-rnn: rnn.o learn-rnn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lspeech -lopt -lla -lebt -lblas

predict-rnn: rnn.o predict-rnn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lspeech -lopt -lla -lebt -lblas

libnn.a: nn.o
	$(AR) rcs $@ $^

libnngpu.a: nn.o nn-gpu.o
	$(AR) rcs $@ $^

nn.o: nn.h

nn-gpu.o: nn-gpu.cu
	nvcc $(CXXFLAGS) -c nn-gpu.cu

learn-gpu.o: learn-gpu.cu
	nvcc $(CXXFLAGS) -c learn-gpu.cu

learn-gpu: learn-gpu.o nn-gpu.o nn.o
	$(CXX) $(CXXFLAGS) -L /opt/cuda/lib64 -o $@ $^ -lautodiffgpu -loptgpu -llagpu -lblas -lebt -lcublas -lcudart

nn-gpu.o: nn-gpu.h nn.h
learn-gpu.o: nn-gpu.h
rnn.o: rnn.h
