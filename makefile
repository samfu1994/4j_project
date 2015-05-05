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
MLP_FILES   = mlp.o test.o
MLP_CFLAGS  = $(OPTFLAGS)

THREADS_HEADERS	= threads.h
THREADS_FILES	= threads.o
THREADS_CFLAGS	= $(OPTFLAGS)

PROB_HEADERS	= 
PROB_FILES		= prob.o
PROB_CFLAGS		= $(OPTFLAGS)
PROB_LIBS		= -lpthread

all: prob mlp

prob: $(PROB_FILES) $(LIBLINEAR_FILES) $(BLAS_FILES) $(THREADS_FILES)
	g++ $(PROB_CFLAGS) $^ -o build/prob $(PROB_LIBS)

$(PROB_FILES)::%.o:%.cpp $(PROB_HEADERS)
	cd build && g++ $(THREADS_CFLAGS) -c ../$*.cpp

# threads:   $(THREADS_HEADERS) $(THREADS_FILES)

$(THREADS_FILES):%.o:%.cpp $(THREADS_HEADERS)
	cd build && g++ $(THREADS_CFLAGS) -c ../src/$*.cpp

mlp: $(MLP_FILES) $(MLP_HEADERS)
	cd build && g++ $(MLP_FILES) -o mlp

$(MLP_FILES):%.o:%.cpp $(MLP_HEADERS)
	cd build && g++ $(MLP_CFLAGS) -c ../mlp/$*.cpp



liblinear:   $(LIBLINEAR_FILES) $(LIBLINEAR_HEADERS)

$(LIBLINEAR_FILES):%.o:%.cpp $(LIBLINEAR_HEADERS) blas
	cd build && g++ $(LIBLINEAR_CFLAGS) -c ../liblinear/$*.cpp



blas: $(BLAS_FILES) $(BLAS_HEADERS)

$(BLAS_FILES):%.o:%.c $(BLAS_HEADERS)
	cd build && gcc $(BLAS_CFLAGS) -c ../blas/$*.c


clean:
	rm build/*