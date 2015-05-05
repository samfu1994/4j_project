# vpath %.c blas
# vpath %.o build
# vpath %.h blas
VPATH = build : src : liblinear : mlp : blas

# OPTFLAGS = -g -Wall

BLAS_HEADERS 	= blas.h blasp.h
BLAS_FILES 		= dnrm2.o daxpy.o ddot.o dscal.o 
BLAS_CFLAGS 	= $(OPTFLAGS)

LIBLINEAR_HEADERS 	= linear.h  tron.h
LIBLINEAR_FILES 	= linear.o  tron.o
LIBLINEAR_CFLAGS 	= $(OPTFLAGS)

MLP_HEADERS = mlp.h
MLP_FILES   = mlp.o
MLP_CFLAGS  = $(OPTFLAGS)

MLP_TEST_HEADERS = mlp.h
MLP_TEST_FILES   = test.o
MLP_TEST_CFLAGS  = $(OPTFLAGS)

THREADS_HEADERS	= threads.h
THREADS_FILES	= threads.o
THREADS_CFLAGS	= $(OPTFLAGS)

PROB_HEADERS	= 
PROB_FILES		= prob.o
PROB_CFLAGS		= $(OPTFLAGS)
PROB_LIBS		= -lpthread

PROB_MLP_HEADERS	= 
PROB_MLP_FILES		= prob_mlp.o
PROB_MLP_CFLAGS		= $(OPTFLAGS)
PROB_MLP_LIBS		= -lpthread

all: prob prob_mlp test

prob_mlp: $(PROB_MLP_FILES) $(LIBLINEAR_FILES) $(BLAS_FILES) $(THREADS_FILES) $(MLP_FILES)
	cd build && g++ $(PROB_MLP_CFLAGS) $(patsubst build/%, % ,$^) -o prob_mlp $(PROB_MLP_LIBS)

$(PROB_MLP_FILES)::%.o:%.cpp $(PROB_HEADERS)
	cd build && g++ $(THREADS_CFLAGS) -c ../$*.cpp

prob: $(PROB_FILES) $(LIBLINEAR_FILES) $(BLAS_FILES) $(THREADS_FILES)
	cd build && g++ $(PROB_CFLAGS) $(patsubst build/%, % ,$^) -o prob $(PROB_LIBS)

$(PROB_FILES)::%.o:%.cpp $(PROB_HEADERS)
	cd build && g++ $(THREADS_CFLAGS) -c ../$*.cpp

$(THREADS_FILES):%.o:%.cpp $(THREADS_HEADERS)
	cd build && g++ $(THREADS_CFLAGS) -c ../src/$*.cpp

test: $(MLP_TEST_FILES) $(MLP_FILES)
	cd build && g++ $(patsubst build/%, % ,$^) -o test

$(MLP_TEST_FILES):%.o:%.cpp $(MLP_HEADERS)
	cd build && g++ $(MLP_CFLAGS) -c ../mlp/$*.cpp

$(MLP_FILES):%.o:%.cpp $(MLP_HEADERS)
	cd build && g++ $(MLP_CFLAGS) -c ../mlp/$*.cpp

$(LIBLINEAR_FILES):%.o:%.cpp $(LIBLINEAR_HEADERS) $(BLAS_FILES)
	cd build && g++ $(LIBLINEAR_CFLAGS) -c ../liblinear/$*.cpp

$(BLAS_FILES):%.o:%.c $(BLAS_HEADERS)
	cd build && gcc $(BLAS_CFLAGS) -c ../blas/$*.c

runmlp:
	make && ./build/prob_mlp

runprob:
	make &&	./build/prob

runtest:
	make &&	./build/test

clean:
	rm build/*