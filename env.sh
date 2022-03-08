# Add any `module load` or `export` commands that your code needs to
# compile and run to this file.
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export OMP_NUM_THREADS=28
module load languages/gcc/9.3.0
module load languages/intel/2020-u4