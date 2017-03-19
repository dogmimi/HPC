CC=gcc61
FLAGS=-O3 -fopenmp -Wall -g -lm
EXECS=omp_solved2 omp_solved3 omp_solved4 omp_solved5 omp_solved6 jacobi2D-omp gs2D-omp
 
all: ${EXECS}

omp_solved2: omp_solved2.c
	${CC} ${FLAGS} $^ -o omp_solved2

omp_solved3: omp_solved3.c
	${CC} ${FLAGS} $^ -o omp_solved3

omp_solved4: omp_solved4.c
	${CC} ${FLAGS} $^ -o omp_solved4

omp_solved5: omp_solved5.c
	${CC} ${FLAGS} $^ -o omp_solved5

omp_solved6: omp_solved6.c
	${CC} ${FLAGS} $^ -o omp_solved6

jacobi2D-omp: jacobi2D-omp.c
	${CC} ${FLAGS} $^ -o jacobi2D-omp

gs2D-omp: gs2D-omp.c
	${CC} ${FLAGS} $^ -o gs2D-omp

clean:
	rm -f ${EXECS}
