CXXFLAGS += -std=c++11 -I ../ -L ../ebt -L ../la -L ../autodiff -L ../opt

.PHONY: all clean

all: learn

clean:
	-rm *.o
	-rm learn

learn: learn.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lautodiff -lopt -lla -lebt -lblas
