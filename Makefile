CXXFLAGS += -std=c++11 -I ../ -L ../ebt -L ../la -L ../autodiff -L ../opt -L ../speech
AR = gcc-ar

.PHONY: all clean gpu

bin = \
    learn \
    predict \
    learn-lstm \
    predict-lstm \
    loss-lstm \
    learn-gru \
    predict-gru \
    learn-residual \
    predict-residual \
    learn-lstm-seg \
    predict-lstm-seg \
    learn-lstm-seg-li \
    predict-lstm-seg-li \
    lstm-seg-li-avg \
    lstm-seg-li-grad \
    lstm-seg-li-update \
    libnn.a

all: $(bin)

gpu: learn-gpu learn-lstm-gpu libnngpu.a

clean:
	-rm *.o
	-rm $(bin)
	-rm learn-lstm-gpu libnngpu.a

libnn.a: nn.o lstm.o pred.o residual.o tensor_tree.o
	$(AR) rcs $@ $^

learn: nn.o learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lla -lebt -lblas

predict: nn.o predict.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lla -lebt -lblas

learn-lstm: learn-lstm.o tensor_tree.o lstm.o pred.o nn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lspeech -lopt -lla -lebt -lblas

predict-lstm: predict-lstm.o tensor_tree.o lstm.o pred.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lspeech -lopt -lla -lebt -lblas

loss-lstm: loss-lstm.o tensor_tree.o lstm.o pred.o nn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lspeech -lopt -lla -lebt -lblas

learn-gru: gru.o learn-gru.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lspeech -lopt -lla -lebt -lblas

predict-gru: gru.o predict-gru.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lspeech -lopt -lla -lebt -lblas

learn-residual: learn-residual.o residual.o pred.o nn.o tensor_tree.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lspeech -lopt -lla -lebt -lblas

predict-residual: predict-residual.o residual.o pred.o tensor_tree.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lspeech -lopt -lla -lebt -lblas

learn-lstm-seg: learn-lstm-seg.o tensor_tree.o lstm.o pred.o nn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lspeech -lopt -lla -lebt -lblas

predict-lstm-seg: predict-lstm-seg.o tensor_tree.o lstm.o pred.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lspeech -lopt -lla -lebt -lblas

learn-lstm-seg-li: learn-lstm-seg-li.o lstm-seg.o tensor_tree.o lstm.o pred.o nn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lspeech -lopt -lla -lebt -lblas

predict-lstm-seg-li: predict-lstm-seg-li.o lstm-seg.o tensor_tree.o lstm.o pred.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lspeech -lopt -lla -lebt -lblas

lstm-seg-li-avg: lstm-seg-li-avg.o lstm-seg.o tensor_tree.o lstm.o pred.o nn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lspeech -lopt -lla -lebt -lblas

lstm-seg-li-grad: lstm-seg-li-grad.o lstm-seg.o tensor_tree.o lstm.o pred.o nn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lspeech -lopt -lla -lebt -lblas

lstm-seg-li-update: lstm-seg-li-update.o lstm-seg.o tensor_tree.o lstm.o pred.o nn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lspeech -lopt -lla -lebt -lblas

nn.o: nn.h
lstm.o: lstm.h
gru.o: gru.h
pred.o: pred.h
residual.o: residual.h
tensor_tree.o: tensor_tree.h

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
