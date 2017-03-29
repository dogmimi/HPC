 /* program spawn multiple processes, use rank to communicate
 * 0 -> 1 -> 2 -> ... -> N -> 0 -> 1 -> ...
 */

#include "util.h"
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
//#define BANDWIDTH

void checkResult(int result, int rounds, int totalRounds, int rank){
  #ifdef DEBUG
  int expectedResult = (totalRounds * (totalRounds + 1) * rounds + rank * (rank - 1)) / 2;
  if(result != expectedResult){
    printf("rounds: %d, totalRounds: %d, rank: %d 's is NOT correct!!\n", rounds, totalRounds, rank);
  }else{
    printf("rounds: %d, totalRounds: %d, rank: %d 's is Correct!!\n", rounds, totalRounds, rank);
  }
  printf("result: %d, expected: %d\n", result, expectedResult);
  #endif
}

int main(int argc, char *argv[])
{
  int rank, tag, origin, destination, size, i;
  MPI_Status status;
  timestamp_type time_start, time_finish;
    
  
  char hostname[1024];
  gethostname(hostname, 1024);

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int rounds = 10;

  if(argc != 2){
    printf("Parameters are not matched! use default parameters\n");
    printf("Should be ./int_ring 10\n");
  }else{
    rounds = strtol(argv[1], NULL, 10);
  }

  if(rank == 0){
    printf("Total rounds: %d, size: %d\n", rounds, size);
  }
 
  #ifdef BANDWIDTH
  int data_size = 10000000;
  double *message_out = calloc(data_size, sizeof(double));
  double *message_in = calloc(data_size, sizeof(double));
  #endif
  #ifndef BANDWIDTH
  int message_out = rank;
  int message_in = -1;
  #endif
  tag = 99;

  for(i = 0; i < rounds; i++){
    #ifndef BANDWIDTH
    if(rank == 0){//send 0 -> 1, receive N -> 0
      if(i == 0){
        get_timestamp(&time_start);

        message_out = 0;
      }

      destination = 1;
      origin = size - 1;

      MPI_Send(&message_out, 1, MPI_INT, destination, tag, MPI_COMM_WORLD);
      MPI_Recv(&message_in,  1, MPI_INT, origin,      tag, MPI_COMM_WORLD, &status);
      
      message_out = message_in;

      //only print time at node 0 at last round 
      if(i == (rounds - 1)){
        get_timestamp(&time_finish);
        double elapsed = timestamp_diff_in_seconds(time_start, time_finish);
        printf("Time elapsed is %f seconds.\n", elapsed);
      }
      checkResult(message_in, i + 1, size - 1, rank);
    }else if(rank < size - 1){//send x -> x + 1, receive x - 1 -> x
      destination = rank + 1;
      origin = rank - 1;

      MPI_Recv(&message_in,  1, MPI_INT, origin,      tag, MPI_COMM_WORLD, &status);
      message_out = message_in + rank;
      MPI_Send(&message_out, 1, MPI_INT, destination, tag, MPI_COMM_WORLD);
    
      checkResult(message_in, i, size - 1, rank);
    }else{//send N -> 0, receive N - 1 -> N
      destination = 0;
      origin = rank - 1;

      MPI_Recv(&message_in,  1, MPI_INT, origin,      tag, MPI_COMM_WORLD, &status);
      message_out = message_in + rank;
      MPI_Send(&message_out, 1, MPI_INT, destination, tag, MPI_COMM_WORLD);
    
      checkResult(message_in, i, size - 1, rank);
    }
    
      #ifdef DEBUG  
      printf("round: %d, rank %d hosted on %s received from %d the message %d\n", i + 1, rank, hostname, origin, message_in);
      #endif
    #endif

    #ifdef BANDWIDTH

    if(rank == 0){//send 0 -> 1, receive N -> 0
      if(i == 0){
        get_timestamp(&time_start);
      }

      destination = 1;
      origin = size - 1;

      MPI_Send(message_out, data_size, MPI_DOUBLE, destination, tag, MPI_COMM_WORLD);
      MPI_Recv(message_in,  data_size, MPI_DOUBLE, origin,      tag, MPI_COMM_WORLD, &status);
      
      //only print time at node 0 at last round 
      if(i == (rounds - 1)){
        get_timestamp(&time_finish);
        double elapsed = timestamp_diff_in_seconds(time_start, time_finish);
        printf("Time elapsed is %f seconds.\n", elapsed);
        double bandwidth = size * sizeof(double) * data_size / (double)(rounds * 1024 * 1024);//MB/s
        printf("Bandwidth: %fMB/s\n", bandwidth); 
      }
    }else if(rank < size - 1){//send x -> x + 1, receive x - 1 -> x
      destination = rank + 1;
      origin = rank - 1;

      MPI_Recv(message_in,  data_size, MPI_DOUBLE, origin,      tag, MPI_COMM_WORLD, &status);
      MPI_Send(message_out, data_size, MPI_DOUBLE, destination, tag, MPI_COMM_WORLD);
    }else{//send N -> 0, receive N - 1 -> N
      destination = 0;
      origin = rank - 1;

      MPI_Recv(message_in,  data_size, MPI_DOUBLE, origin,      tag, MPI_COMM_WORLD, &status);
      MPI_Send(message_out, data_size, MPI_DOUBLE, destination, tag, MPI_COMM_WORLD);
    }
    #endif
  }

  #ifdef BANDWIDTH  
  free(message_in);
  free(message_out);
  #endif

  MPI_Finalize();
  return 0;
}
