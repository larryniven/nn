CXXFLAGS += -std=c++11 -I ../ -L ../ebt -L ../la -L ../autodiff -L ../opt -L ../speech
AR = gcc-ar

.PHONY: all clean

all: libnn.a

clean:
	-rm *.o
	-rm libnn.a

libnn.a: nn.o rhn.o gru.o lstm.o pred.o lstm-seg.o residual.o tensor-tree.o lstm-frame.o lstm-tensor-tree.o
	$(AR) rcs $@ $^

nn.o: nn.h
lstm.o: lstm.h
lstm-seg.o: lstm-seg.h
gru.o: gru.h
pred.o: pred.h
residual.o: residual.h
tensor-tree.o: tensor-tree.h

