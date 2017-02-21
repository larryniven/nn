CXXFLAGS += -std=c++11 -I ../../ -L ../../la -L ../../ebt -L ../../autodiff -L ../../opt -L ../

.PHONY: all clean

all: test-nn

clean:
	-rm *.o
	-rm test-nn

test-nn: test-nn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lnn -lautodiff -lopt -lla -lebt -lblas

