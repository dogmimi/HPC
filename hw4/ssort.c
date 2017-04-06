/* Parallel sample sort
 */
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>


static int compare(const void *a, const void *b)
{
  int *da = (int *)a;
  int *db = (int *)b;

  if (*da > *db)
    return 1;
  else if (*da < *db)
    return -1;
  else
    return 0;
}

int main( int argc, char *argv[]){
  int rank;
  int i, N, num_tasks, totalN, j, k, count;
  int *vec;
  int *splitter, *totalSplitter, *buckets, *gatheredBuckets, *localBuckets;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  /* use broadcast to send N to each tasks */
  if(argc != 2){
    if(rank == 0){
      printf("should be run like ./ssort 100\n");
      MPI_Abort(MPI_COMM_WORLD, 0);
      exit(0);
    }
  }

  /* Number of random numbers per processor (this should be increased
   * for actual tests or could be passed in through the command line */
  N = atoi(argv[1]);
  MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
   
  totalN = N * num_tasks;

  vec = calloc(N, sizeof(int));
  /* seed random number generator differently on every core */
  srand((unsigned int) (rank + 393919));

  /* fill vector with random integers */
  for (i = 0; i < N; ++i) {
    vec[i] = rand();
  }
  //printf("rank: %d, first entry: %d\n", rank, vec[0]);

  /* sort locally */
  qsort(vec, N, sizeof(int), compare);

  /* randomly sample s entries from vector or select local splitters,
   * i.e., every N/P-th entry of the sorted vector */

  splitter = (int*)malloc(sizeof(int) * (num_tasks - 1));
  for(i = 0; i < num_tasks - 1; i++){
    splitter[i] = vec[totalN / (num_tasks * num_tasks) * (i + 1)];
  } 

  /* every processor communicates the selected entries
   * to the root processor; use for instance an MPI_Gather */
  
  totalSplitter = (int*)malloc(sizeof(int) * num_tasks * (num_tasks - 1));
  MPI_Gather(splitter, num_tasks - 1, MPI_INT, totalSplitter, num_tasks - 1, MPI_INT, 0, MPI_COMM_WORLD);

  /* root processor does a sort, determinates splitters that
   * split the data into P buckets of approximately the same size */

  if(rank == 0){
    qsort(totalSplitter, num_tasks * (num_tasks - 1), sizeof(int), compare);

    //determine splitter for all elements
    for(i = 0; i < num_tasks - 1; i++){
      splitter[i] = totalSplitter[(num_tasks - 1) * (i + 1)];
    }
  }
  

  /* root process broadcasts splitters */

  MPI_Bcast(splitter, num_tasks - 1, MPI_INT, 0, MPI_COMM_WORLD);

  /* every processor uses the obtained splitters to decide
   * which integers need to be sent to which other processor (local bins) */

  buckets = (int*)malloc(sizeof(int) * (totalN + num_tasks));
  
  j = 0;
  k = 1; //used to record element counts between splitter(i) and splitter(i + 1)

  for(i = 0; i < N; i++){
    if(j < (num_tasks - 1)){
      if(vec[i] < splitter[j]){ 
        buckets[((N + 1) * j) + k] = vec[i];
        k++;
      }else{
        buckets[(N + 1) * j] = k - 1;
        k = 1; //to next splitter, reset k
        j++; //to next splitter element
        i--; //to next splitter so fallback 1 element or this element will be ignored!
      }
    }else{
      buckets[((N + 1) * j) + k] = vec[i];
      k++;
    }
  }
  buckets[(N + 1) * j] = k - 1; 

  /* send and receive: either you use MPI_AlltoallV, or
   * (and that might be easier), use an MPI_Alltoall to share
   * with every processor how many integers it should expect,
   * and then use MPI_Send and MPI_Recv to exchange the data */

  gatheredBuckets = (int*)malloc(sizeof(int) * (totalN + num_tasks));
  MPI_Alltoall(buckets, N + 1, MPI_INT, gatheredBuckets, N + 1, MPI_INT, MPI_COMM_WORLD); 

  /* do a local sort */
  //arrange data first then do sorting
  localBuckets = (int*)malloc(sizeof(int) * 2 * totalN / num_tasks);
  count = 0;

  for(j = 0; j < num_tasks; j++){
    k = 1;
    for(i = 0; i < gatheredBuckets[(totalN / num_tasks + 1) * j]; i++){ 
      localBuckets[count] = gatheredBuckets[(totalN / num_tasks + 1) * j + k];
      count++;
      k++;
    }
  }
  qsort(localBuckets, count, sizeof(int), compare);

  printf("task %d finished sorting %d numbers\n", rank, count); 

  /* every processor writes its result to a file */
  {
    FILE* fd = NULL;
    char filename[256];
    snprintf(filename, 256, "output%02d.txt", rank);
    fd = fopen(filename, "w+");

    if(NULL == fd){
      printf("Error opening file \n");
      return 1;
    }

    for(i = 0; i < count; i++)
      fprintf(fd, "  %d\n", localBuckets[i]);

    fclose(fd);
  }

  /* free resources */
  free(vec);
  free(splitter);
  free(totalSplitter);
  free(buckets);
  free(gatheredBuckets);
  free(localBuckets);

  MPI_Finalize();
  return 0;
}
