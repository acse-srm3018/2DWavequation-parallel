//*************************************************************************************
//                      CourseWork 6.2 
//		MPI Programming Assignment– Solving the Wave Equation
//*************************************************************************************
//
//  Purpose:
//
//    MAIN is the main program for Wave_2D_MPI. Note that this is a parallel implementation.
//  
//	Discussion:	
/*
	This program uses a finite difference scheme to solve
	Wave's equation for a square matrix distributed over a
	(logical)processor topology.

	This program works on the SPMD(single program, multiple data)
	paradigm.It illustrates 2 - d block decomposition, nodes exchanging
	edge values, and convergence checking.

	Each matrix element is updated based on the values of the four
	neighboring matrix elements.This process is repeated until the data
	converges, that is, until the average change in any matrix element(compared
	to the value 20 iterations previous) is smaller than a specified value.

	Each process exchanges edge values with its four neighbors.
	Then new values are calculated for the upper leftand lower right corners
	of each node's matrix.The processes exchange edge values again.The upper right
	and lower left corners are then calculated.

	The program is currently configured for a 100x100 matrix
	distributed over four processors. It can be edited to handle
	different matrix sizes or number of processors.
*/
//  Modified:
//
//    23 April 2021
//
//  Author:
// 
//    Raha Moosavi
//  
//	Global Parameters:
//
//    Global, double dt, the time step.
//	  Global, int id_row, the MPI process id in row.
//	  Global, int id_column, the MPI process id in column.
//    Global, int id, the MPI process id.
//	  Global, int rows, the number of rows in MPI processes  
//	  Global, int columns, the number of columns in MPI processes  
//    Global, int p, the number of MPI processes.
//	  Global, double x_max, maximum size of x direction.
//	  Global, double y_max, maximum size of y direction.
//    Global, int jmax, the total number of points in y direction.
//	  Global, double t_max, maximum of time.
//	  MPI_Datatype for Send data to neighbors : Datatype_left, Datatype_right, Datatype_top, Datatype_bottom;
//	  MPI_Datatype for Send data to neighbors : Datatype_left_recv, Datatype_right_recv, Datatype_top_recv, Datatype_bottom_recv;
//--------------------------------------------------------------------------------------------------------------------------------
//
#define _USE_MATH_DEFINES
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <cmath>
#include <mpi.h>
#include <iomanip>
#include <cstdlib>
#include <time.h>
#include <chrono>

using namespace std;
int id, p;
int tag_num = 0;
int rows, columns;
int id_row, id_column;
int imax_local, jmax_local;
int imax = 10, jmax = 10;
int* requests;
int cnt = 0;
int right_neigh_id, left_neigh_id, top_neigh_id, bottom_neigh_id;
// 1-D array for setup partition
int* row_start, * column_start, * row_final, * column_final, * process_chunk, * num_row, * num_column;
//1-D array grid, new_grid, old_grid, Initial_cond, Initial_cond_old; 
double* grid, * new_grid, * old_grid, * Initial_cond, * Initial_cond_old;
double t_max = 30.0;
double t, t_out = 0.0, dt_out = 0.04, dt;
double y_max = 10.0, x_max = 10.0, dx, dy;
double c = 1;
double t1, t2;

MPI_Datatype Datatype_left, Datatype_right, Datatype_top, Datatype_bottom, Datatype_left_recv, Datatype_right_recv, Datatype_top_recv, Datatype_bottom_recv;

//****************************************************************************************************
void create_left_types(int m, int n)
//****************************************************************************************************
//
//	This function is used to create data type for SEND data to LEFT neighbour
//	We collect all the addresses of data in memory in a vector and their offsets
//	And then we used MPI_Type_create_struct and then MPI_Type_commit to create them 
//	
{
	vector<int> block_lengths;
	vector<MPI_Datatype> typelist;
	vector<MPI_Aint> addresses;
	MPI_Aint add_start;
	add_start = 0;

	//left data for sending to right processors
	for (int i = 1; i < m - 1; i++)
	{
		MPI_Aint temp_address;
		block_lengths.push_back(1);
		typelist.push_back(MPI_DOUBLE);
		MPI_Get_address(&grid[1 + i * n], &temp_address);
		addresses.push_back(temp_address);
	}
	MPI_Get_address(&grid, &add_start);
	for (size_t i = 0; i < addresses.size(); i++)
	{
		addresses[i] = addresses[i] - add_start;
	}
	MPI_Type_create_struct(m - 2, block_lengths.data(), addresses.data(), typelist.data(), &Datatype_left);
	MPI_Type_commit(&Datatype_left);
}

//****************************************************************************************************
void create_right_types(int m, int n)
//****************************************************************************************************
//
//	This function is used to create data type for SEND data to RIGHT neighbour
//	We collect all the addresses of data in memory in a vector and their offsets
//	And then we used MPI_Type_create_struct and then MPI_Type_commit to create them 
//	
{
	//right data for sending to left processor
	vector<int> block_lengths;
	vector<MPI_Datatype> typelist;
	vector<MPI_Aint> addresses;
	MPI_Aint add_start;

	for (int i = 1; i < m - 1; i++)
	{
		block_lengths.push_back(1);
		typelist.push_back(MPI_DOUBLE);
		MPI_Aint temp_address;
		MPI_Get_address(&grid[(n - 2) + n * i], &temp_address);
		addresses.push_back(temp_address);
	}
	MPI_Get_address(&grid, &add_start);
	for (size_t i = 0; i < addresses.size(); i++) addresses[i] = addresses[i] - add_start;
	MPI_Type_create_struct(m - 2, block_lengths.data(), addresses.data(), typelist.data(), &Datatype_right);
	MPI_Type_commit(&Datatype_right);
}

//****************************************************************************************************
void create_top_types(int m, int n)
//****************************************************************************************************
//
//	This function is used to create data type for SEND data to TOP neighbour
//	We collect all the addresses of data in memory in a vector and their offsets
//	And then we used MPI_Type_create_struct and then MPI_Type_commit to create them 
//	
{
	//top - only need one value
	int block_length = n - 2;
	MPI_Datatype typeval = MPI_DOUBLE;
	MPI_Aint address;
	MPI_Aint add_start;

	MPI_Get_address(&grid, &add_start);
	MPI_Get_address(&grid[1 + n * (m - 2)], &address);
	address = address - add_start;

	MPI_Type_create_struct(1, &block_length, &address, &typeval, &Datatype_top);
	MPI_Type_commit(&Datatype_top);
}

//****************************************************************************************************
void create_bottom_types(int m, int n)
//****************************************************************************************************
//
//	This function is used to create data type for SEND data to BOTTOM neighbour
//	We collect all the addresses of data in memory in a vector and their offsets
//	And then we used MPI_Type_create_struct and then MPI_Type_commit to create them 
//	
{
	int block_length = n - 2;
	MPI_Datatype typeval = MPI_DOUBLE;
	MPI_Aint address;
	MPI_Aint add_start;
	address = 0;

	//bottom - only need one value

	MPI_Get_address(&grid[1 + n], &address);
	MPI_Get_address(&grid, &add_start);
	address = address - add_start;

	MPI_Type_create_struct(1, &block_length, &address, &typeval, &Datatype_bottom);
	MPI_Type_commit(&Datatype_bottom);
}

//****************************************************************************************************
void create_left_types_recv(int m, int n)
//****************************************************************************************************
//
//	This function is used to create data type for RECEIVE data to LEFT neighbour
//	We collect all the addresses of data in memory in a vector and their offsets
//	And then we used MPI_Type_create_struct and then MPI_Type_commit to create them 
//	
{
	//Data receive at LEFT ghost cell
	vector<int> block_lengths;
	vector<MPI_Datatype> typelist;
	vector<MPI_Aint> addresses;
	vector<MPI_Aint> displacement;
	MPI_Aint add_start;
	MPI_Aint temp_address;

	for (int i = 1; i < m - 1; i++)
	{
		block_lengths.push_back(1);
		typelist.push_back(MPI_DOUBLE);

		MPI_Get_address(&grid[i * n], &temp_address);
		addresses.push_back(temp_address);
	}
	MPI_Get_address(&grid, &add_start);
	for (size_t i = 0; i < addresses.size(); i++)
	{
		displacement.push_back(addresses[i] - add_start);
	}
	MPI_Type_create_struct(m - 2, block_lengths.data(), displacement.data(), typelist.data(), &Datatype_left_recv);
	MPI_Type_commit(&Datatype_left_recv);
}

//****************************************************************************************************
void create_right_types_recv(int m, int n)
//****************************************************************************************************
//
//	This function is used to create datatype for RECEIVE data to RIGHT neighbour
//	We collect all the addresses of data in memory in a vector and their offsets
//	And then we used MPI_Type_create_struct and then MPI_Type_commit to create them 
//	
{
	//Data receive at RIGHT ghost cell
	vector<int> block_lengths;
	vector<MPI_Datatype> typelist;
	vector<MPI_Aint> addresses;
	vector<MPI_Aint> displacement;
	MPI_Aint add_start;
	MPI_Aint temp_address;
	//add_start = 0;

	for (int i = 1; i < m - 1; i++)
	{
		block_lengths.push_back(1);
		typelist.push_back(MPI_DOUBLE);

		MPI_Get_address(&grid[(n - 1) + i * (n)], &temp_address);
		addresses.push_back(temp_address);
	}

	MPI_Get_address(&grid, &add_start);
	for (size_t i = 0; i < addresses.size(); i++)
	{
		displacement.push_back(addresses[i] - add_start);
	}
	MPI_Type_create_struct(m - 2, block_lengths.data(), displacement.data(), typelist.data(), &Datatype_right_recv);
	MPI_Type_commit(&Datatype_right_recv);
}

//****************************************************************************************************
void create_top_types_recv(int m, int n)
//****************************************************************************************************
//
//	This function is used to create datatype for RECEIVE data to TOP neighbour
//	We collect all the addresses of data in memory in a vector and their offsets
//	And then we used MPI_Type_create_struct and then MPI_Type_commit to create them 
//	
{
	int block_length = n - 2;
	MPI_Datatype typeval = MPI_DOUBLE;
	MPI_Aint address;
	MPI_Aint add_start;

	//top - for recieving at ghost cells

	MPI_Get_address(&grid[1 + (n) * (m - 1)], &address);
	MPI_Get_address(&grid, &add_start);

	address = address - add_start;

	MPI_Type_create_struct(1, &block_length, &address, &typeval, &Datatype_top_recv);
	MPI_Type_commit(&Datatype_top_recv);
}

//****************************************************************************************************
void create_bottom_types_recv(int m, int n)
//****************************************************************************************************
//
//	This function is used to create datatype for RECEIVE data to BOTTOM neighbour
//	We collect all the addresses of data in memory in a vector and their offsets
//	And then we used MPI_Type_create_struct and then MPI_Type_commit to create them 
//
{
	//bottom - for receiving at ghost cells
	int block_length = n - 2;
	MPI_Datatype typeval = MPI_DOUBLE;
	MPI_Aint address;
	MPI_Aint add_start;

	MPI_Get_address(&grid[1], &address);
	MPI_Get_address(&grid, &add_start);
	address = address - add_start;

	MPI_Type_create_struct(1, &block_length, &address, &typeval, &Datatype_bottom_recv);
	MPI_Type_commit(&Datatype_bottom_recv);
}

//****************************************************************************************************
void find_dimensions(int p, int& rows, int& columns)   
//****************************************************************************************************
//
//	This function is used to divided total number of processors into rows * columns
//	Try to decompose it to near square shape
//	A bit brute force - this can definitely be made more efficient!
//
{
	int min_gap = p;
	int top = sqrt(p) + 1;
	for (int i = 1; i <= top; i++)
	{
		if (p % i == 0)
		{
			int gap = abs(p / i - i);

			if (gap < min_gap)
			{
				min_gap = gap;
				rows = i;
				columns = p / i;
			}
		}
	}
	if (id == 0)
		cout << "Divide " << p << " into " << rows << " by " << columns << " grid" << endl;
}

//****************************************************************************************************
void id_to_index(int id, int& id_row, int& id_column)
//****************************************************************************************************
//
//	This function is used to decompose each id to its index of that processor into id_row * id_column
//
{
	id_column = id % columns;
	id_row = id / columns;
}

//****************************************************************************************************
int id_from_index(int id_row, int id_column)
//****************************************************************************************************
//
//	This function is used to get index for each processor to id
//
{
	if (id_row >= rows || id_row < 0)
		return -1;
	if (id_column >= columns || id_column < 0)
		return -1;

	return id_row * columns + id_column;
}

//****************************************************************************************************
void neighbour_ids(int id, int id_row, int id_column, int& right_neigh_id, int& left_neigh_id, int& top_neigh_id, int& bottom_neigh_id)
//****************************************************************************************************
//					This function calculate each neighbour id ids
//
//	Discussion :
//
//  This version of neighbors ids finding are for Periodic boundary conditions where we just force the values on the 
//	right hand edge to wrap around to match those on the left, andthe same top to bottom. 
//
//	Parameters and arguments:
//
//	id : index of processors
//	id_row : index of processors in row (location of processors in row after domain decomposition)
//	id_column :	index of processors in column (location of processors in column after domain decomposition)
//	right_neigh_id : index of neighbour processor in right side
//	leftt_neigh_id : index of neighbour processor in left side
//	top_neigh_id : index of neighbour processor in top side
//	bottom_neigh_id : index of neighbour processor in bottom side
//
//	Strategies:
//	
//	Divided the processors in some parts : Internal processors, corner processors, edges(right, left, top, bottom)
//
//	For internal processors:
//
//	left_neighbor : id - 1
//	right_neighbor : id + 1
//	top_neighbour : id + columns
//	bottom_neighbor : id - columns
// 
//	For Corner processors : 
//
//	top-right, top-left, bottom-right, bottom-left edges
//
//	For edge Processors :
//
//	edge-right, edge-left, edge-top, edge-bottom
//
{
	// internal processors neighbors
	if (id_row > 0 && id_row < rows - 1 && id_column > 0 && id_column < columns - 1)
	{
		right_neigh_id = id + 1;
		left_neigh_id = id - 1;
		top_neigh_id = id + columns;
		bottom_neigh_id = id - columns;
	}
	// most bottom procesoors' nodes neighbors (excep edges)
	else if (id_row == 0 && id_column < columns - 1 && id_column > 0)
	{
		right_neigh_id = id + 1;
		left_neigh_id = id - 1;
		top_neigh_id = id + columns;
		bottom_neigh_id = id + (rows - 1) * columns;
	}
	// Top nodes processors' nodes neighbors (excep edges)
	else if (id_row == rows - 1 && id_column < columns - 1 && id_column > 0)
	{
		right_neigh_id = id + 1;
		left_neigh_id = id - 1;
		top_neigh_id = id - (rows - 1) * columns;
		bottom_neigh_id = id - columns;
	}

	//	Most left nodes processors' nodes neighbors (excep edges)
	else if (id_column == 0 && id_row < rows - 1 && id_row > 0)
	{
		right_neigh_id = id + 1;
		left_neigh_id = id + (columns - 1);
		top_neigh_id = id + columns;
		bottom_neigh_id = id - columns;
	}

	// Most right nodes processors' nodes neighbors (excep edges)
	else if (id_column == columns - 1 && id_row < rows - 1 && id_row > 0)
	{
		right_neigh_id = id - (columns - 1);
		left_neigh_id = id - 1;
		top_neigh_id = id + columns;
		bottom_neigh_id = id - columns;
	}

	//	Top right edge node neighbors 
	else if (id_column == columns - 1 && id_row == rows - 1)
	{
		right_neigh_id = id - (columns - 1);
		left_neigh_id = id - 1;
		top_neigh_id = id - (rows - 1) * columns;
		bottom_neigh_id = id - columns;
	}

	//	Top left edge node neighbors 
	else if (id_column == 0 && id_row == rows - 1)
	{
		right_neigh_id = id + 1;
		left_neigh_id = id + (columns - 1);
		top_neigh_id = id - (rows - 1) * columns;
		bottom_neigh_id = id - columns;
	}

	//  bottom right edge node neighbors 
	else if (id_row == 0 && id_column == columns - 1)
	{
		right_neigh_id = id - (columns - 1);
		left_neigh_id = id - 1;
		top_neigh_id = id + columns;
		bottom_neigh_id = id + (rows - 1) * columns;
	}

	//  bottom left edge node neighbors 
	else if (id_row == 0 && id_column == 0)
	{
		right_neigh_id = id + 1;
		left_neigh_id = id + (columns - 1);
		top_neigh_id = id + columns;
		bottom_neigh_id = id + (rows - 1) * columns;
	}
}


//****************************************************************************************************
void send_matrix(double* data, MPI_Datatype mydatatype, int dest, MPI_Request* requests, int cnt)
//****************************************************************************************************
//  Function to send data to left/right/Top/Bottom neighbour id from right, left, top, bottom datatype. 
{
	MPI_Isend(&data, 1, mydatatype, dest, tag_num, MPI_COMM_WORLD, &requests[cnt]);
	cnt++;
}


//****************************************************************************************************
void recv_matrix(double* data, MPI_Datatype mydatatype, int source, MPI_Request* requests, int cnt)
//****************************************************************************************************
//  Function to recieve data from left/right/Top/Bottom neighbour id from right, left, top, bottom datatype. 
{
	MPI_Irecv(&data, 1, mydatatype, source, tag_num, MPI_COMM_WORLD, &requests[cnt]);
	cnt++;
}


//****************************************************************************************************
void setup_partition()
//****************************************************************************************************
//  Function to assigned each processors a chunk of total data.
//	which column and row start and finish for assigned data to each processors
//	The numeber of rows and columns which assigned to each processors
//
{
	row_start = new int[rows];
	row_final = new int[rows];
	column_start = new int[columns];
	column_final = new int[columns];
	process_chunk = new int[p];
	num_row = new int[p];
	num_column = new int[p];

	int rows_left = imax;
	int columns_left = jmax;
	row_start[0] = 0;
	column_start[0] = 0;

	for (int i = 0; i < rows - 1; i++)
	{
		int rows_assigned = rows_left / (rows - i);
		rows_left -= rows_assigned;
		row_start[i + 1] = row_start[i] + rows_assigned;
		row_final[i] = row_start[i + 1] - 1;
		//cout << row_final[i] << endl;
	}
	row_final[rows - 1] = imax - 1;

	for (int j = 0; j < columns - 1; j++)
	{
		int columns_assigned = columns_left / (columns - j);
		columns_left -= columns_assigned;
		column_start[j + 1] = column_start[j] + columns_assigned;
		column_final[j] = column_start[j + 1] - 1;
	}
	column_final[columns - 1] = jmax - 1;

	for (int n = 0; n < p; n++)
	{
		num_row[n] = row_final[id_row] - row_start[id_row] + 1;
		num_column[n] = column_final[id_column] - column_start[id_column] + 1;
		process_chunk[n] = (row_final[id_row] - row_start[id_row] + 1) * (column_final[id_column] - column_start[id_column] + 1);
	}
}
//****************************************************************************************************
void grid_to_file(int out)
//****************************************************************************************************
//
//  Function to write the result for each processor for all time step in a file.
//  The file name format is : ./out/output_tstep_id
//
{
	stringstream fname;
	fstream f1;
	fname << "./out/output" << "_" << out << "_" << id << ".dat";
	f1.open(fname.str().c_str(), ios_base::out);
	for (int i = 1; i < imax_local - 1; i++)
	{
		for (int j = 1; j < jmax_local - 1; j++)
			f1 << grid[i * (jmax_local) + j] << "\t";
		f1 << endl;
	}
	f1.close();
}
//****************************************************************************************************
void do_iteration(void)
//****************************************************************************************************
//
//  Function to do iteration in each time step and calculate and update the wave equation solution. 
//
//  Discussion:
//    Discretize the equation for u(x,t):
//      d^2 u/dt^2 - c^2 * (d^2 u/dx^2 + d^2 u/dy^2) = 0  for 0 < x < 10, 0 < y < 10, 0 < t < 30;
//
//	with boundary conditions(fixed boundary condition):
//      u(0,y,t) = u0(t) = 0
//      u(x,y,t) = u1(t) = 0
//      u(x,0,t) = u0(t) = 0
//      u(x,y,t) = u1(t) = 0
//
//	with boundary conditions(periodic boundary condition):
//      u(0,t) = u0(t) = u1(t)
//      u(10,t) = u1(t) = u0(t)
//      u(0,t) = u0(t) = u1(t)
//      u(10,t) = u1(t) = u0(t)  
//
//    by:
//
//      alpha = c * dt / dx.
//
//      U(x,t+dt) = 2 U(x,t) - U(x,t-dt) 
//        + alpha^2 ( U(x-dx,y,t) - 2 U(x,y,t) + U(x+dx,y,t) + U(x,y-dy,t) - 2 U(x,y,t) + U(x,y+dy,t) ).
//
//
{
	for (int i = 1; i < imax_local - 1; i++)
	{
		for (int j = 1; j < jmax_local - 1; j++)
		{
			new_grid[j + i * jmax_local] = pow(dt * c, 2.0) * ((grid[j + (i + 1) * jmax_local] - 2.0 * grid[j + i * jmax_local] + grid[j + (i - 1) * jmax_local]) / pow(dx, 2.0) + (grid[(j + 1) * jmax_local + i] - 2.0 * grid[j + i * jmax_local] + grid[(j - 1) + i * jmax_local]) / pow(dy, 2.0)) + 2.0 * grid[j + jmax_local * i] - old_grid[j + i * jmax_local];
			//cout << new_grid[j + i * jmax_local] << endl;
		}
	}
	t += dt;

	std::swap(*old_grid, *new_grid);
	std::swap(*old_grid, *grid);
}

//****************************************************************************************************
void Initial_calc()
//****************************************************************************************************
//
//  Function to do iteration in each time step and calculate and update the wave equation solution. 
//	Initial conditions assigned to each process_chunk processor
//         u(x,y,0) 
//
{
	//sets half sinusoidal intitial disturbance - this is brute force - it can be done more elegantly
	double r_splash = 1.0;
	double x_splash = 3.0;
	double y_splash = 3.0;

	dx = x_max / ((double)imax - 1);
	dy = y_max / ((double)jmax - 1);
	int numrow = imax / rows;
	int numcolumn = jmax / columns;
	for (int i = 0; i < imax_local; i++)
		for (int j = 0; j < jmax_local; j++)
		{
			double x = dx * i * (id_column + 1);
			double y = dy * j * (id_row + 1);

			double dist = sqrt(pow(x - x_splash, 2.0) + pow(y - y_splash, 2.0));

			if (dist < r_splash)
			{
				double h = 5.0 * (cos(dist / r_splash * M_PI) + 1.0);

				grid[j + i * jmax_local] = h;
				old_grid[j + i * jmax_local] = h;
			}
			
			else
			{
				grid[j + i * jmax_local] = 0;
				old_grid[j + i * jmax_local] = 0;
			}
			
		}
}
//******************************************************************************************
int main(int argc, char* argv[])
//******************************************************************************************
//	Main function to calculate wave equation at different timesteps and different locations
//  
//  Local parameters:
//    Local, int imax_local, the number of points in x direction visible to this process.
//    Local, int jmax_local, the number of points in y direction visible to this process.
//	  The size of grid, new_grid and old_grid is (num_row + 2 * num_column+2)
//	  The size of row and column for each processors_chunk added to two because
//	  we want to store the result from neighbor processors to this additional rows and columns
//	  which is known as ghost cells
//
{
	MPI_Init(&argc, &argv);
	//  Record the starting time.
	MPI_Barrier(MPI_COMM_WORLD);
	t1 = MPI_Wtime();

	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	find_dimensions(p, rows, columns);
	id_to_index(id, id_row, id_column);
	neighbour_ids(id, id_row, id_column, right_neigh_id, left_neigh_id, top_neigh_id, bottom_neigh_id);
	setup_partition();

	imax_local = num_row[id] + 2;
	jmax_local = num_column[id] + 2;

	grid = new double[imax_local * jmax_local];
	new_grid = new double[imax_local * jmax_local];
	old_grid = new double[imax_local * jmax_local];

	Initial_calc();
	/*
	for (int i = 0; i < imax_local - 2; i++)
	{
		for (int j = 0; j < jmax_local - 2; j++)
		{
			//cout << column_start[id_column] + j << endl;
			grid[(j + 1) + (i + 1) * jmax_local] = 0;
			old_grid[(j + 1) + (i + 1) * jmax_local] = 0;
		}
	}
	*/
	t = 0.0;
	dt = 0.1 * min(dx, dy) / c;
	int out_cnt = 0, it = 0;
	//create data types for send and receiving data 
	create_left_types(imax_local, jmax_local); //left_send
	create_right_types(imax_local, jmax_local); //right_send
	create_top_types(imax_local, jmax_local); //top_send
	create_bottom_types(imax_local, jmax_local); //bottom_send
	//-------------------------------------------------------------------------------------------
	create_right_types_recv(imax_local, jmax_local); //left_receive
	create_left_types_recv(imax_local, jmax_local); //right_receive
	create_top_types_recv(imax_local, jmax_local); //top_receive
	create_bottom_types_recv(imax_local, jmax_local); //bottom_receive

	// while loop to do iteration through timesteps
	while (t < t_max)
	{
		MPI_Request* requests = nullptr;

		requests = new MPI_Request[4 * 2];
		// communicate for sending data to ghost cells
		int cnt = 0;
		MPI_Isend(&grid, 1, Datatype_top, top_neigh_id, tag_num, MPI_COMM_WORLD, &requests[cnt]);
		cnt++;
		MPI_Isend(&grid, 1, Datatype_right, right_neigh_id, tag_num, MPI_COMM_WORLD, &requests[cnt]);
		cnt++;
		MPI_Isend(&grid, 1, Datatype_left, left_neigh_id, tag_num, MPI_COMM_WORLD, &requests[cnt]);
		cnt++;
		MPI_Isend(&grid, 1, Datatype_bottom, bottom_neigh_id, tag_num, MPI_COMM_WORLD, &requests[cnt]);
		cnt++;
		//------------------------------------------------------------------------------------------------
			// communicate for recieiving data at ghost cells
		MPI_Irecv(&grid, 1, Datatype_top_recv, top_neigh_id, tag_num, MPI_COMM_WORLD, &requests[cnt]);
		cnt++;
		MPI_Irecv(&grid, 1, Datatype_left_recv, left_neigh_id, tag_num, MPI_COMM_WORLD, &requests[cnt]);
		cnt++;
		MPI_Irecv(&grid, 1, Datatype_right_recv, right_neigh_id, tag_num, MPI_COMM_WORLD, &requests[cnt]);
		cnt++;
		MPI_Irecv(&grid, 1, Datatype_bottom_recv, bottom_neigh_id, tag_num, MPI_COMM_WORLD, &requests[cnt]);
		cnt++;

		// Wait for all non - blocking communications to complete.
		MPI_Waitall(cnt, requests, MPI_STATUSES_IGNORE);
		// call function to update for next time steps
		do_iteration();
		// writing the result to a file using grid_to_file function
		if (t_out <= t)
		{
			grid_to_file(out_cnt);
			out_cnt++;
			t_out += dt_out;
		}
		it++;
	}
	//  Record the final time.
	MPI_Barrier(MPI_COMM_WORLD);
	t2 = MPI_Wtime();
	//	Printing the elapsed time
	if (id == 0)
		printf("Elapsed time in seconds is %f\n", t2 - t1);
	//  Free created data types (for send and receive data to neighbors)
	MPI_Type_free(&Datatype_left);
	MPI_Type_free(&Datatype_right);
	MPI_Type_free(&Datatype_top);
	MPI_Type_free(&Datatype_bottom);
	MPI_Type_free(&Datatype_top_recv);
	MPI_Type_free(&Datatype_bottom_recv);
	MPI_Type_free(&Datatype_left_recv);
	MPI_Type_free(&Datatype_right_recv);

	//  Delete the dynamic allocated arrays
	delete[] grid;
	delete[] new_grid;
	delete[] old_grid;
	delete[] row_start;
	delete[] row_final;
	delete[] column_start;
	delete[] column_final;
	delete[] process_chunk;
	delete[] num_row;
	delete[] num_column;

	MPI_Finalize();

	return 0;
}