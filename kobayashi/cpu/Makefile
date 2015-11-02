PROG = KobayashiCPU

MAIN = KobayashiCPU.cpp

CC      = g++    # Use the g++ Compiler
STDLIBS = -lm  -fopenmp  # Link the math library
CFLAGS  = -O

all: $(PROG)

$(PROG):$(MAIN) Makefile
	$(CC) $(CFLAGS) -o $(PROG) $(MAIN) $(STDLIBS)

clean:
	rm -f *.bak *.o $(PROG)
