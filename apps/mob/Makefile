PROG = mob
MAIN = mob.cpp

CC      = g++    # Use the g++ Compiler
LIBS    = -lm  -fopenmp  # Link the math library
CFLAGS  = -O3 -Wall -std=c++20
INCLUDE = -I../../include


all: $(PROG)

$(PROG):$(MAIN) Makefile
	$(CC) $(CFLAGS) $(INCLUDE) -o $(PROG) $(MAIN) $(LIBS)

clean:
	rm -f *.bak *.o $(PROG)
