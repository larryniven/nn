CXXFLAGS += -std=c++11 -I ../ -L ../ebt -L ../la -L ../autodiff -L ../opt -L ../speech
AR = gcc-ar

.PHONY: all clean

all: libnn.a

clean:
	-rm *.o
	-rm libnn.a

libnn.a: nn.o cnn.o tensor-tree.o lstm.o lstm-frame.o lstm-tensor-tree.o rsg.o
	$(AR) rcs $@ $^

nn.o: nn.h
cnn.o: cnn.h
lstm.o: lstm.h
lstm-seg.o: lstm-seg.h
gru.o: gru.h
pred.o: pred.h
residual.o: residual.h
tensor-tree.o: tensor-tree.h

