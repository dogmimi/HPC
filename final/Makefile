CC=g++
#FLAGS=-I/usr/local/include/opencv -I/usr/local/include/opencv2 -L/usr/local/lib/ -L/usr/local/cuda/lib64 -I/usr/local/cuda/include -g -lcudart -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_stitching
FLAGS=-I/usr/local/include/opencv -I/usr/local/include/opencv2 -L/usr/local/lib/ -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_stitching
EXECS=main

all: ${EXECS}

main: main.cpp core/solver.o core/solver.h utility/imageutility.o utility/imageutility.h
	${CC} ${FLAGS} $^ -o main

clean:
	rm -f main core/solver.o utility/imageutility.o utility/util.o
