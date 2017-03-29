CC=mpicc
FLAGS=-O3
EXECS=int_ring

all: ${EXECS}

debug: FLAGS += -DDEBUG
debug: int_ring

int_ring: int_ring.c
	$(CC) ${FLAGS} $^ -o int_ring

clean:
	rm -f ${EXECS}
