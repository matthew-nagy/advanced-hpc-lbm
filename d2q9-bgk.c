/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <mpi.h>

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

const float c_sq = 1.f / 3.f; /* square of speed of sound */
const float w0 = 4.f / 9.f;  /* weighting factor */
const float w1 = 1.f / 9.f;  /* weighting factor */
const float w2 = 1.f / 36.f; /* weighting factor */



/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int nxm1;
  int nym1;
  int nxBitMask;
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */

  float totCells;
  float totVel;
} t_param;

float** cells;
float** tmp_cells;
t_param params;

int fullGridHeight;
int fullGridWidth;
float** collatedCells;

int nprocs, rank, upRank, downRank;

typedef struct
{
  int rowStartOn;
  int numOfRows;
} rankData;
rankData myRank;
int* fullObstacles;

#define IS_OBSTACLE(x, y) ( (x == 0) || (y == 0) || (x == params.nxm1) || (y == params.nym1) )

/* struct to hold the 'speed' values */

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               int** obstacles_ptr, float** av_vels_ptr);



/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
float timestep(int const*const restrict obstacles);
int accelerate_flow(int const*const restrict obstacles);
int propagate();
int rebound(int* obstacles);
float collision(int const*const restrict obstacles);
int write_values(int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density();

/* compute average velocity */
float av_velocity(int* obstacles);

/* calculate Reynolds number */
float calc_reynolds(int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);


void halo(){
  int bytesPerRow = sizeof(float) * params.nx;
  for(int i = 0; i < NSPEEDS; i++){
    //Send up
    MPI_Sendrecv(
      (const void*)&cells[i][(params.ny - 2) * params.nx], bytesPerRow, MPI_CHAR, upRank, 0,
      (void*)cells[i], bytesPerRow, MPI_CHAR, downRank, 0,
      MPI_COMM_WORLD, MPI_STATUS_IGNORE 
    );
    //Send down
    MPI_Sendrecv(
      (const void*)&cells[i][params.nx], bytesPerRow, MPI_CHAR, downRank, 0,
      (void*)&cells[i][(params.ny - 1) * params.nx], bytesPerRow, MPI_CHAR, upRank, 0,
      MPI_COMM_WORLD, MPI_STATUS_IGNORE 
    );
  }
}


rankData getRankData(int rank){
  rankData dat;
  const int remainder = fullGridHeight % nprocs;



  const int rowPerProc = fullGridHeight / nprocs;
  dat.numOfRows = rowPerProc;
  if(rank < remainder)
    dat.numOfRows += 1;

  dat.rowStartOn = rowPerProc * rank;
  const int numOfPriorRanks = rank;
  int remainderConsideration;
  
  if(numOfPriorRanks > remainder)
    remainderConsideration = remainder;
  else
    remainderConsideration = numOfPriorRanks;
  
  dat.rowStartOn += remainderConsideration;

  return dat;
}

float* collateOnZero(float* av_vels){
  //Create the final grid
  collatedCells = (float**)malloc(sizeof(float*) * NSPEEDS);
  for(int i = 0; i < NSPEEDS; i++)
    collatedCells[i] = (float*)malloc(sizeof(float) * fullGridHeight * fullGridWidth);


  int* bytesPerRank = (int*)malloc(sizeof(int) * nprocs);
  int* addressesPerRank = (int*)malloc(sizeof(int) * nprocs);

  for(int i = 0; i < nprocs; i++){
    rankData rd = getRankData(i);
    bytesPerRank[i] = rd.numOfRows * sizeof(float) * params.nx;
    addressesPerRank[i] = rd.rowStartOn * sizeof(float) * params.nx;
  }

  float* trueVel = malloc(sizeof(float) * params.maxIters);
  MPI_Reduce(
    (void*)av_vels, trueVel, sizeof(float) * params.maxIters,
    MPI_CHAR, MPI_SUM, 0, MPI_COMM_WORLD
  );

  const int speedsSize = sizeof(float) * params.nx * (params.ny - 2);//Don't include the halo regions
  for(int i = 0; i < NSPEEDS; i++){
    MPI_Gatherv(
      (void*)&cells[i][params.nx], speedsSize, MPI_CHAR,
      (void*)collatedCells[i], bytesPerRank, addressesPerRank,
      MPI_CHAR, 0, MPI_COMM_WORLD
    );
  }
  return trueVel;
}
void collate(float* av_vels){

  MPI_Reduce(
    (void*)av_vels, NULL, sizeof(float) * params.maxIters,
    MPI_CHAR, MPI_SUM, 0, MPI_COMM_WORLD
  );

  const int speedsSize = sizeof(float) * params.nx * (params.ny - 2);//Don't include the halo regions
  for(int i = 0; i < NSPEEDS; i++){
    MPI_Gatherv(
      (void*)&cells[i][params.nx], speedsSize, MPI_CHAR,
      //These don't matter to non roots
      NULL, NULL, NULL,
      MPI_CHAR, 0, MPI_COMM_WORLD
    );
  }
}

float* velStorage;

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{

  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;                                                             /* structure to hold elapsed time */
  double tot_tic, tot_toc, init_tic, init_toc, comp_tic, comp_toc, col_tic, col_toc; /* floating point numbers to calculate elapsed wallclock time */

  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2]; 
  }

  /* Total/init time starts here: initialise our data structures and load values from file */
  gettimeofday(&timstr, NULL);
  tot_tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  init_tic=tot_tic;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  upRank = rank + 1;
  downRank = rank - 1;
  printf("Rank %d precheck up to %d, down to %d\n", rank, upRank, downRank);
  if(upRank == nprocs)
    upRank = 0;
  if(downRank == -1)
    downRank = nprocs - 1;
  
  printf("Rank %d sends up to %d, down to %d\n", rank, upRank, downRank);

  initialise(paramfile, obstaclefile, &obstacles, &av_vels);

  printf("Rank %d has dimensions %d %d\n", rank, params.nx, params.ny);

  /* Init time stops here, compute time starts*/
  gettimeofday(&timstr, NULL);
  init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  comp_tic=init_toc;
  
  for(int i = 0; i < nprocs; i++){
    MPI_Barrier(MPI_COMM_WORLD);
    if(i == rank){
      for(int jj = 0; jj < params.ny; jj++){
        for(int ii = 0; ii < params.nx; ii++){
          printf("%d", obstacles[ii + jj * params.nx]);
        }
        printf("\n");
      }
    }
    printf("\n");
  }
  return 0;

  for (int tt = 0; tt < params.maxIters; tt++)
  {
    halo();
    av_vels[tt] = timestep(obstacles);
    float** tmp = tmp_cells;
    tmp_cells = cells;
    cells = tmp; 
#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells));
#endif
  }
  
  /* Compute time stops here, collate time starts*/
  gettimeofday(&timstr, NULL);
  comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  col_tic=comp_toc;

  // Collate data from ranks here 

  if(rank == 0){
    velStorage = av_vels;
    av_vels = collateOnZero(av_vels);
  }
  else{
    collate(av_vels);
    finalise(&obstacles, &av_vels);

    MPI_Finalize();
    return EXIT_SUCCESS;
  }

  for(int i = 0; i < 20; i++)
    printf("%f\n", av_vels[i]);

  /* Total/collate time stops here.*/
  gettimeofday(&timstr, NULL);
  col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  tot_toc = col_toc;
  
  /* write final values and free memory */
  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(obstacles));
  printf("Elapsed Init time:\t\t\t%.6lf (s)\n",    init_toc - init_tic);
  printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
  printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc  - col_tic);
  printf("Elapsed Total time:\t\t\t%.6lf (s)\n",   tot_toc  - tot_tic);
  write_values(obstacles, av_vels);
  finalise(&obstacles, &av_vels);

  MPI_Finalize();

  return EXIT_SUCCESS;
}

float timestep(int const*const restrict obstacles)
{
  accelerate_flow(obstacles);
  //propagate(params, cells, tmp_cells);
  //rebound(params, cells, tmp_cells, obstacles);
  return collision(obstacles);
}


int accelerate_flow(int const*const restrict obstacles)
{
  //Onluy the bottom pls
  if(rank != (nprocs - 1))
    return EXIT_SUCCESS;

  /* compute weighting factors */
  float w1 = params.density * params.accel * (1.0/9.f);
  float w2 = params.density * params.accel * (1.0f/36.f);
  
  int jj = params.ny - 2;


  for (int ii = 1; ii < params.nx - 1; ii++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
      /* increase 'east-side' densities */
      int index = ii + jj * params.nx;
      cells[1][index] += w1;
      cells[5][index] += w2;
      cells[8][index] += w2;
      /* decrease 'west-side' densities */
      cells[3][index] -= w1;
      cells[6][index] -= w2;
      cells[7][index] -= w2;
  }

  return EXIT_SUCCESS;
}

extern inline float innerCollider(int isOb, int y_n, int y_s, int x_e, int x_w, int jj, int ii){
  float scratch[9];
  const int index = ii + jj * params.nx;
  
  /* propagate densities from neighbouring cells, following
  ** appropriate directions of travel and writing into
  ** scratch space grid */
  scratch[0] = cells[0][index]; /* central cell, no movement */
  scratch[1] = cells[1][x_w + jj*params.nx]; /* east */
  scratch[2] = cells[2][ii + y_s*params.nx]; /* north */
  scratch[3] = cells[3][x_e + jj*params.nx]; /* west */
  scratch[4] = cells[4][ii + y_n*params.nx]; /* south */
  scratch[5] = cells[5][x_w + y_s*params.nx]; /* north-east */
  scratch[6] = cells[6][x_e + y_s*params.nx]; /* north-west */
  scratch[7] = cells[7][x_e + y_n*params.nx]; /* south-west */
  scratch[8] = cells[8][x_w + y_n*params.nx]; /* south-east */

  float u_sq = 0.0f;

  float obMark = isOb;
  float nonObMark = 1.f - obMark;

  /* compute local density total */
  float local_density = 0.f;

  for (int kk = 0; kk < NSPEEDS; kk++)
  {
    local_density += scratch[kk];
  }

  /* compute x velocity component */
  float u_x = (scratch[1]
                + scratch[5]
                + scratch[8]
                - (scratch[3]
                    + scratch[6]
                    + scratch[7]))
                / local_density;
  /* compute y velocity component */
  float u_y = (scratch[2]
                + scratch[5]
                + scratch[6]
                - (scratch[4]
                    + scratch[7]
                    + scratch[8]))
                / local_density;

  /* velocity squared */
  u_sq = u_x * u_x + u_y * u_y;

  /* directional velocity components */
  float u[NSPEEDS];
  u[1] =   u_x;        /* east */
  u[2] =         u_y;  /* north */
  u[3] = - u_x;        /* west */
  u[4] =       - u_y;  /* south */
  u[5] =   u_x + u_y;  /* north-east */
  u[6] = - u_x + u_y;  /* north-west */
  u[7] = - u_x - u_y;  /* south-west */
  u[8] =   u_x - u_y;  /* south-east */

  /* equilibrium densities */
  float d_equ[NSPEEDS];

  const float over2c_sq = 1.0 / (2.0f * c_sq);
  const float over2c_sq_squared = 1.0 / (2.f * c_sq * c_sq);
  const float overC_sq = 1.0 / c_sq;

  /* zero velocity density: weight w0 */
  d_equ[0] = w0 * local_density
              * (1.f - u_sq * over2c_sq);
  /* axis speeds: weight w1 */
  d_equ[1] = w1 * local_density * (1.f + u[1] * overC_sq
                                    + (u[1] * u[1]) * over2c_sq_squared
                                    - u_sq * over2c_sq);
  d_equ[2] = w1 * local_density * (1.f + u[2] * overC_sq
                                    + (u[2] * u[2]) * over2c_sq_squared
                                    - u_sq * over2c_sq);
  d_equ[3] = w1 * local_density * (1.f + u[3] * overC_sq
                                    + (u[3] * u[3]) * over2c_sq_squared
                                    - u_sq * over2c_sq);
  d_equ[4] = w1 * local_density * (1.f + u[4] * overC_sq
                                    + (u[4] * u[4]) * over2c_sq_squared
                                    - u_sq * over2c_sq);
  /* diagonal speeds: weight w2 */
  d_equ[5] = w2 * local_density * (1.f + u[5] * overC_sq
                                    + (u[5] * u[5]) * over2c_sq_squared
                                    - u_sq * over2c_sq);
  d_equ[6] = w2 * local_density * (1.f + u[6] * overC_sq
                                    + (u[6] * u[6]) * over2c_sq_squared
                                    - u_sq * over2c_sq);
  d_equ[7] = w2 * local_density * (1.f + u[7] * overC_sq
                                    + (u[7] * u[7]) * over2c_sq_squared
                                    - u_sq * over2c_sq);
  d_equ[8] = w2 * local_density * (1.f + u[8] * overC_sq
                                    + (u[8] * u[8]) * over2c_sq_squared
                                    - u_sq * over2c_sq);

  local_density = 0.0f;
  /* relaxation step */
  
  tmp_cells[0][index] = ((scratch[0] + params.omega * (d_equ[0] - scratch[0])) * nonObMark) + (scratch[0] * obMark);
  tmp_cells[1][index] = ((scratch[1] + params.omega * (d_equ[1] - scratch[1])) * nonObMark) + (scratch[3] * obMark);
  tmp_cells[2][index] = ((scratch[2] + params.omega * (d_equ[2] - scratch[2])) * nonObMark) + (scratch[4] * obMark);
  tmp_cells[3][index] = ((scratch[3] + params.omega * (d_equ[3] - scratch[3])) * nonObMark) + (scratch[1] * obMark);
  tmp_cells[4][index] = ((scratch[4] + params.omega * (d_equ[4] - scratch[4])) * nonObMark) + (scratch[2] * obMark);
  tmp_cells[5][index] = ((scratch[5] + params.omega * (d_equ[5] - scratch[5])) * nonObMark) + (scratch[7] * obMark);
  tmp_cells[6][index] = ((scratch[6] + params.omega * (d_equ[6] - scratch[6])) * nonObMark) + (scratch[8] * obMark);
  tmp_cells[7][index] = ((scratch[7] + params.omega * (d_equ[7] - scratch[7])) * nonObMark) + (scratch[5] * obMark);
  tmp_cells[8][index] = ((scratch[8] + params.omega * (d_equ[8] - scratch[8])) * nonObMark) + (scratch[6] * obMark);
  
                                                                                                                                                                                                                                                                                                            
  for (int kk = 0; kk < NSPEEDS; kk++)
  {
    local_density += tmp_cells[kk][index];
  }

  /* compute x velocity component */
  u_x = (tmp_cells[1][index]
        + tmp_cells[5][index]
        + tmp_cells[8][index]
        - (tmp_cells[3][index]
            + tmp_cells[6][index]
            + tmp_cells[7][index]))
        / local_density;
  /* compute y velocity component */
  u_y = (tmp_cells[2][index]
        + tmp_cells[5][index]
        + tmp_cells[6][index]
        - (tmp_cells[4][index]
            + tmp_cells[7][index]
            + tmp_cells[8][index]))
        / local_density;


  /* velocity squared */
  u_sq = u_x * u_x + u_y * u_y;

  //tot_u and obs[ii jj] are both 0 if not neccessary, so it all works
  /* accumulate the norm of x- and y- velocity components */
  return sqrtf(u_sq) * nonObMark;

}

extern inline void outerCollide(int const*const restrict obstacles, int y_n, int y_s, int jj){
  float tmp_vel = 0.0f;

  

  __assume((params.nx % 4) == 0);
  #pragma omp simd aligned(cells:64), aligned(tmp_cells:64), reduction(+:tmp_vel)
  for (int ii = 0; ii < params.nx; ii+=1)
  {
    /* determine indices of axis-direction neighbours
    ** respecting periodic boundary conditions (wrap around) */
    int x_e = (ii + 1) & params.nxBitMask;
    int x_w = (ii - 1) & params.nxBitMask;
    tmp_vel += innerCollider(obstacles[ii + jj *params.nx], y_n, y_s, x_e, x_w, jj, ii);
  }
  params.totVel += tmp_vel;
}

float collision(int const*const restrict obstacles)
{
  params.totVel = 0.0f;

  /* loop over the cells in the grid
  ** NB the collision step is called after
  ** the propagate step and so values of interest
  ** are in the scratch-space grid */
  
  for (int jj = 1; jj < params.ny - 1; jj++)
  {
    const int y_n = (jj + 1);
    const int y_s = (jj - 1);
    outerCollide(obstacles, y_n, y_s, jj);
  }
  
  return params.totVel / params.totCells;
}

float av_velocity(int* obstacles)
{
  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

  /* loop over all non-blocked cells */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii + jj*params.nx])
      {
        /* local density total */
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[kk][ii + jj*params.nx];
        }

        /* x-component of velocity */
        float u_x = (cells[1][ii + jj*params.nx]
                      + cells[5][ii + jj*params.nx]
                      + cells[8][ii + jj*params.nx]
                      - (cells[3][ii + jj*params.nx]
                         + cells[6][ii + jj*params.nx]
                         + cells[7][ii + jj*params.nx]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (cells[2][ii + jj*params.nx]
                      + cells[5][ii + jj*params.nx]
                      + cells[6][ii + jj*params.nx]
                      - (cells[4][ii + jj*params.nx]
                         + cells[7][ii + jj*params.nx]
                         + cells[8][ii + jj*params.nx]))
                     / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
      }
    }
  }

  return tot_u / (float)tot_cells;
}

int initialise(const char* paramfile, const char* obstaclefile,
               int** obstacles_ptr, float** av_vels_ptr)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }


  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params.nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &fullGridHeight);

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  fullGridWidth = params.nx;
  params.nxBitMask = params.nx - 1;



  retval = fscanf(fp, "%d\n", &(params.maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params.reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params.density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params.accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params.omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  myRank = getRankData(rank);
  params.ny = myRank.numOfRows + 2;//Give room for the halos

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  params.nxm1 = params.nx - 1;
  params.nym1 = params.ny - 1;

  /* main grid */
  cells = (float**)aligned_alloc(64, sizeof(float*) * (NSPEEDS));
  tmp_cells = (float**)aligned_alloc(64, sizeof(float*) * (NSPEEDS));
  for(int i = 0; i < NSPEEDS; i++){
    // (cells)[i] = (float*)malloc(sizeof(float) * params.nx * params.ny);
    // (tmp_cells)[i] = (float*)malloc(sizeof(float) * params.nx * params.ny);
    (cells)[i] = (float*)aligned_alloc(64, sizeof(float) * params.nx * params.ny);
    (tmp_cells)[i] = (float*)aligned_alloc(64, sizeof(float) * params.nx * params.ny);
  }


  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int) * (params.ny * params.nx));

  if(rank == 0){
    fullObstacles = malloc(sizeof(int) * fullGridHeight * fullGridWidth);
  }

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params.density * 4.f / 9.f;
  float w1 = params.density      / 9.f;
  float w2 = params.density      / 36.f;

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      const int index = ii + jj*params.nx;
      /* centre */
      (cells)[0][index] = w0;
      /* axis directions */
      (cells)[1][index] = w1;
      (cells)[2][index] = w1;
      (cells)[3][index] = w1;
      (cells)[4][index] = w1;
      /* diagonals */
      (cells)[5][index] = w2;
      (cells)[6][index] = w2;
      (cells)[7][index] = w2;
      (cells)[8][index] = w2;
      (*obstacles_ptr)[index] = 0;
    }
  }

  if(rank == 0){
    const int indices = fullGridHeight * fullGridWidth;
    for(int i = 0; i < indices; i++)
      fullObstacles[i] = 0;
  }



  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  params.totCells = fullGridHeight * fullGridWidth;
  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    if(rank == 0){
      /* assign to array */
      fullObstacles[xx + yy*params.nx] = blocked;
    }
    params.totCells -= 1.f;

    int adjustedY = yy - myRank.rowStartOn;
    if(adjustedY >=0 && adjustedY < params.ny)
      (*obstacles_ptr)[xx + adjustedY * params.nx] = blocked;

  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params.maxIters);

  return EXIT_SUCCESS;
}

int finalise(int** obstacles_ptr, float** av_vels_ptr)
{
  /*
  ** free up allocated memory
  */
  for(int i = 0; i < NSPEEDS; i++){
    free((cells)[i]);
    free((tmp_cells)[i]);
  }
  free(cells);
  free(tmp_cells);

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  if(rank == 0){
    for(int i = 0; i < NSPEEDS; i++)
      free(collatedCells[i]);
    free(collatedCells);

    free(fullObstacles);

    free(velStorage);
  }

  return EXIT_SUCCESS;
}


float calc_reynolds(int* obstacles)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity(obstacles) * params.reynolds_dim / viscosity;
}

float total_density()
{
  float total = 0.f;  /* accumulator */

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        total += cells[kk][ii + jj*params.nx];
      }
    }
  }

  return total;
}

int write_values(int* obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < fullGridHeight; jj++)
  {
    for (int ii = 0; ii < fullGridWidth; ii++)
    {
      /* an occupied cell */
      if (fullObstacles[ii + jj*params.nx])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += collatedCells[kk][ii + jj*params.nx];
        }

        /* compute x velocity component */
        u_x = (cells[1][ii + jj*params.nx]
               + cells[5][ii + jj*params.nx]
               + cells[8][ii + jj*params.nx]
               - (cells[3][ii + jj*params.nx]
                  + cells[6][ii + jj*params.nx]
                  + cells[7][ii + jj*params.nx]))
              / local_density;
        /* compute y velocity component */
        u_y = (cells[2][ii + jj*params.nx]
               + cells[5][ii + jj*params.nx]
               + cells[6][ii + jj*params.nx]
               - (cells[4][ii + jj*params.nx]
                  + cells[7][ii + jj*params.nx]
                  + cells[8][ii + jj*params.nx]))
              / local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, fullObstacles[ii * fullGridWidth + jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}