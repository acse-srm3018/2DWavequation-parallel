# 2D Wave equation - Parallel programming using MPI
* 2D Wave Equation Solver with MPI
## Overview
This code solves using Finite difference method [https://en.wikipedia.org/wiki/Finite_difference]. The code is parallelized using MPI and domain decomposition techniques. It was mostly written with MPI and C++.

## Basic Information

This project aims to solve 2-D wave equation using finite difference. 

An overview of the files is provided below.
- `report/` contains report in pdf format.
- `log/` contains the output files from my HPC runs (*.o* files).
- `images/` contains images and animation which used in report and also ipynb file for generating files.
- `hpc job file/` contains jobs which used for running in HPC system.
- `Serial_Array.cpp` serial version code of solving 2D wave equation using finite difference using vector.
- `Serial_Wave_Equation.cpp` serial version code of solving 2D wave equation using finite difference using 1-D array.
- `coursework_parallel.cpp` parallel version code of solving 2D wave equation using finite difference.
- `post_processing.ipynb` jupyter notebook of postrprocessing.
- `animation.gif` animation of wave equation solution
- `LICENSE.txt` is the MIT license.
- `README.md` contains basic information for the repository and detailed information for how to compile and reproduce the results.


## Speedup
The time it takes for the code, as measured using the dug HPC system, to execute (Different Number of grid points) relative to the number of processors used can be seen in the figure below. Note that the computational speedup from 2 to 4 processors is 1.9 and from 4 to 8 the speedup is 1.4 which is close Amdahl’s Law.

## Requirements
1. only non-standard library required is `mpi.h`. 
2. Building program using `mpic++ coursework_parallel.cpp -o coursework_parallel`
3. run `mpiexec -n 4 coursework_parallel`
4. run in the following command in hpc `rjs mpi_Raha.job`
5. the output piped to a file (.out) and plotted using the
   postprocessing python code(which can be found as postprocess.ipynb
