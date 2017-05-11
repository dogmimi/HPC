CC=mpicc
FLAGS=-O3 -mavx
#FLAGS=-O3 -xMIC-AVX512
EXECS=rof tvl1 rof_avx tvl1_avx
#EXECS=rof tvl1 rof_avx tvl1_avx rof_avx512
LDFLAGS=-lm
 
all: ${EXECS}

rof: rof.c
	${CC} ${FLAGS} $^ -o rof ${LDFLAGS}
tvl1: tvl1.c
	${CC} ${FLAGS} $^ -o tvl1 ${LDFLAGS}
rof_avx: rof_avx.c
	${CC} ${FLAGS} $^ -o rof_avx ${LDFLAGS}
tvl1_avx: tvl1_avx.c
	${CC} ${FLAGS} $^ -o tvl1_avx ${LDFLAGS}
#rof_avx512: rof_avx512.c
#	${CC} ${FLAGS} $^ -o rof_avx512 ${LDFLAGS}

clean:
	rm -f ${EXECS}
