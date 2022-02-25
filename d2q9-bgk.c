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
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */

  int totCells;
  float totVel;
} t_param;

/* struct to hold the 'speed' values */

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, float*** cells_ptr, float*** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr);

typedef float* const restrict*const restrict CellList;



/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
float timestep(t_param*const restrict params, CellList cells, CellList tmp_cells, int const*const restrict obstacles);
int accelerate_flow(const t_param params, CellList cells, int const*const restrict obstacles);
int propagate(const t_param params, float** cells, float** tmp_cells);
int rebound(const t_param params, float** cells, float** tmp_cells, int* obstacles);
float collision(t_param*const restrict params, CellList cells, CellList tmp_cells, int const*const restrict obstacles);
int write_values(const t_param params, float** cells, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, float*** cells_ptr, float*** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, float** cells);

/* compute average velocity */
float av_velocity(const t_param params, float** cells, int* obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, float** cells, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  float** cells     = NULL;    /* grid containing fluid densities */
  float** tmp_cells = NULL;    /* scratch space */
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
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels);

  /* Init time stops here, compute time starts*/
  gettimeofday(&timstr, NULL);
  init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  comp_tic=init_toc;

  for (int tt = 0; tt < params.maxIters; tt++)
  {
    av_vels[tt] = timestep(&params, cells, tmp_cells, obstacles);
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

  /* Total/collate time stops here.*/
  gettimeofday(&timstr, NULL);
  col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  tot_toc = col_toc;
  
  /* write final values and free memory */
  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles));
  printf("Elapsed Init time:\t\t\t%.6lf (s)\n",    init_toc - init_tic);
  printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
  printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc  - col_tic);
  printf("Elapsed Total time:\t\t\t%.6lf (s)\n",   tot_toc  - tot_tic);
  write_values(params, cells, obstacles, av_vels);
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);

  return EXIT_SUCCESS;
}

float timestep(t_param*const restrict params, CellList cells, CellList tmp_cells, int const*const restrict obstacles)
{
  accelerate_flow(*params, cells, obstacles);
  //propagate(params, cells, tmp_cells);
  //rebound(params, cells, tmp_cells, obstacles);
  return collision(params, cells, tmp_cells, obstacles);
}

int accelerate_flow(const t_param params, CellList cells, int const*const restrict obstacles)
{
  /* compute weighting factors */
  float w1 = params.density * params.accel / 9.f;
  float w2 = params.density * params.accel / 36.f;

  /* modify the 2nd row of the grid */
  int jj = params.ny - 2;

  #pragma omp simd aligned(cells : 64)
  for (int ii = 0; ii < params.nx; ii++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ii + jj*params.nx]
        && (cells[3][ii + jj*params.nx] - w1) > 0.f
        && (cells[6][ii + jj*params.nx] - w2) > 0.f
        && (cells[7][ii + jj*params.nx] - w2) > 0.f)
    {
      /* increase 'east-side' densities */
      cells[1][ii + jj*params.nx] += w1;
      cells[5][ii + jj*params.nx] += w2;
      cells[8][ii + jj*params.nx] += w2;
      /* decrease 'west-side' densities */
      cells[3][ii + jj*params.nx] -= w1;
      cells[6][ii + jj*params.nx] -= w2;
      cells[7][ii + jj*params.nx] -= w2;
    }
  }

  return EXIT_SUCCESS;
}

inline void innerCollider(t_param*const restrict params, CellList cells, CellList tmp_cells, int const*const restrict obstacles, int y_n, int y_s, int x_e, int x_w, int jj, int ii, float* dat){
  float scratch[9];
  dat[0] = 0.0f;
  dat[1] = 0.0f;

  /* propagate densities from neighbouring cells, following
  ** appropriate directions of travel and writing into
  ** scratch space grid */
  scratch[0] = cells[0][ii + jj*params->nx]; /* central cell, no movement */
  scratch[1] = cells[1][x_w + jj*params->nx]; /* east */
  scratch[2] = cells[2][ii + y_s*params->nx]; /* north */
  scratch[3] = cells[3][x_e + jj*params->nx]; /* west */
  scratch[4] = cells[4][ii + y_n*params->nx]; /* south */
  scratch[5] = cells[5][x_w + y_s*params->nx]; /* north-east */
  scratch[6] = cells[6][x_e + y_s*params->nx]; /* north-west */
  scratch[7] = cells[7][x_e + y_n*params->nx]; /* south-west */
  scratch[8] = cells[8][x_w + y_n*params->nx]; /* south-east */

  float u_sq = 0.0f;
  /* called after propagate, so taking values from scratch space
  ** mirroring, and writing into main grid */
  tmp_cells[1][ii + jj*params->nx] = scratch[3];
  tmp_cells[2][ii + jj*params->nx] = scratch[4];
  tmp_cells[3][ii + jj*params->nx] = scratch[1];
  tmp_cells[4][ii + jj*params->nx] = scratch[2];
  tmp_cells[5][ii + jj*params->nx] = scratch[7];
  tmp_cells[6][ii + jj*params->nx] = scratch[8];
  tmp_cells[7][ii + jj*params->nx] = scratch[5];
  tmp_cells[8][ii + jj*params->nx] = scratch[6];


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
    /* zero velocity density: weight w0 */
    d_equ[0] = w0 * local_density
                * (1.f - u_sq / (2.f * c_sq));
    /* axis speeds: weight w1 */
    d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                      + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                      + (u[2] * u[2]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                      + (u[3] * u[3]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                      + (u[4] * u[4]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    /* diagonal speeds: weight w2 */
    d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                      + (u[5] * u[5]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                      + (u[6] * u[6]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                      + (u[7] * u[7]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));
    d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                      + (u[8] * u[8]) / (2.f * c_sq * c_sq)
                                      - u_sq / (2.f * c_sq));

    if(!obstacles[jj*params->nx + ii]){
      /* relaxation step */
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        tmp_cells[kk][ii + jj*params->nx] = scratch[kk]
                                                + params->omega
                                                * (d_equ[kk] - scratch[kk]);
      }

      //tot_u and obs[ii jj] are both 0 if not neccessary, so it all works
      /* accumulate the norm of x- and y- velocity components */
      dat[0] += sqrtf(u_sq);
      /* increase counter of inspected cells */
      dat[1] += (1 - obstacles[jj*params->nx + ii]);
    }
}

inline void outerCollide(t_param*const restrict params, CellList cells, CellList tmp_cells, int const*const restrict obstacles, int y_n, int y_s, int jj){

  float tmp_cell = 0.0f;
  float tmp_vel = 0.0f;

  
  float datOut1[2];
  int x_eo = 1;
  int x_wo = params->nx - 1;
  innerCollider(params, cells, tmp_cells, obstacles, y_n, y_s, x_eo, x_wo, jj, 0, datOut1);
  tmp_vel += datOut1[0];
  tmp_cell += datOut1[1];

  #pragma omp simd aligned(cells:64), aligned(tmp_cells:64), reduction(+:tmp_cell), reduction(+:tmp_vel)
  for (int ii = 1; ii < params->nx - 1; ii++)
  {
    /* determine indices of axis-direction neighbours
    ** respecting periodic boundary conditions (wrap around) */
    float dat[2];
    int x_e = ii + 1;
    int x_w = ii - 1;
    innerCollider(params, cells, tmp_cells, obstacles, y_n, y_s, x_e, x_w, jj, ii, dat);
    tmp_vel += dat[0];
    tmp_cell += dat[1];

  }

  float datOut2[2];
  x_eo = 0;
  x_wo = params->nx - 2;
  innerCollider(params, cells, tmp_cells, obstacles, y_n, y_s, x_eo, x_wo, jj, params->nx - 1, datOut2);
  tmp_vel += datOut2[0];
  tmp_cell += datOut2[1];

  params->totCells += tmp_cell;
  params->totVel += tmp_vel;
}

float collision(t_param*const restrict params, CellList cells, CellList tmp_cells, int const*const restrict obstacles)
{

  params->totCells = 0;
  params->totVel = 0.0f;

  /* loop over the cells in the grid
  ** NB the collision step is called after
  ** the propagate step and so values of interest
  ** are in the scratch-space grid */
  
  int y_n = 1;
  int y_s = params->ny;
  outerCollide(params, cells, tmp_cells, obstacles, y_n, y_s, 0);
  y_s = -1;
  for (int jj = 1; jj < params->ny - 1; jj++)
  {
    y_n += 1;
    y_s += 1;
    outerCollide(params, cells, tmp_cells, obstacles, y_n, y_s, jj);
  }

  y_n = 0;
  y_s += 1;
  outerCollide(params, cells, tmp_cells, obstacles, y_n, y_s, y_s + 1);
  
  return params->totVel / (float)params->totCells;
}

float av_velocity(const t_param params, float** cells, int* obstacles)
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
               t_param* params, float*** cells_ptr, float*** tmp_cells_ptr,
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
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

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

  /* main grid */
  *cells_ptr = (float**)aligned_alloc(64, sizeof(float*) * (NSPEEDS));
  *tmp_cells_ptr = (float**)aligned_alloc(64, sizeof(float*) * (NSPEEDS));
  for(int i = 0; i < NSPEEDS; i++){
    // (*cells_ptr)[i] = (float*)malloc(sizeof(float) * params->nx * params->ny);
    // (*tmp_cells_ptr)[i] = (float*)malloc(sizeof(float) * params->nx * params->ny);
    (*cells_ptr)[i] = (float*)aligned_alloc(64, sizeof(float) * params->nx * params->ny);
    (*tmp_cells_ptr)[i] = (float*)aligned_alloc(64, sizeof(float) * params->nx * params->ny);
  }


  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      /* centre */
      (*cells_ptr)[0][ii + jj*params->nx] = w0;
      /* axis directions */
      (*cells_ptr)[1][ii + jj*params->nx] = w1;
      (*cells_ptr)[2][ii + jj*params->nx] = w1;
      (*cells_ptr)[3][ii + jj*params->nx] = w1;
      (*cells_ptr)[4][ii + jj*params->nx] = w1;
      /* diagonals */
      (*cells_ptr)[5][ii + jj*params->nx] = w2;
      (*cells_ptr)[6][ii + jj*params->nx] = w2;
      (*cells_ptr)[7][ii + jj*params->nx] = w2;
      (*cells_ptr)[8][ii + jj*params->nx] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      (*obstacles_ptr)[ii + jj*params->nx] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[xx + yy*params->nx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, float*** cells_ptr, float*** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr)
{
  /*
  ** free up allocated memory
  */
  for(int i = 0; i < NSPEEDS; i++){
    free((*cells_ptr)[i]);
    free((*tmp_cells_ptr)[i]);
  }
  free(*cells_ptr);
  free(*tmp_cells_ptr);

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, float** cells, int* obstacles)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, float** cells)
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

int write_values(const t_param params, float** cells, int* obstacles, float* av_vels)
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

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* an occupied cell */
      if (obstacles[ii + jj*params.nx])
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
          local_density += cells[kk][ii + jj*params.nx];
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
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
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