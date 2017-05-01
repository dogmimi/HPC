CC=mpicc
FLAGS=-O3
EXECS=rof tvl1
LDFLAGS=-lm
 
all: ${EXECS}

rof: rof.c
	${CC} ${FLAGS} $^ -o rof ${LDFLAGS}
rvl1: tvl1.c
	${CC} ${FLAGS} $^ -o tvl1 ${LDFLAGS}

clean:
	rm -f ${EXECS}
