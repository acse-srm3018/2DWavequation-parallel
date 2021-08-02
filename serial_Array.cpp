#define _USE_MATH_DEFINES

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>

using namespace std;

//Note that this is a very simple serial implementation with a fixed grid and Neumann boundaries at the edges
//vector<vector<double>> grid, new_grid, old_grid;
int imax = 1000, jmax = 1000;
int* grid = new int[imax * jmax];
int* new_grid = new int[imax * jmax];
int* old_grid = new int[imax * jmax];
double t_max = 30.0;
double t, t_out = 0.0, dt_out = 0.04, dt;
double y_max = 10.0, x_max = 10.0, dx, dy;
double c = 1;

void grid_to_file(int out)
{
	stringstream fname;
	fstream f1;
	fname << "./out/output" << "_" << out << ".dat";
	f1.open(fname.str().c_str(), ios_base::out);
	for (int i = 0; i < imax; i++)
	{
		for (int j = 0; j < jmax; j++)
			f1 << grid[i * jmax + j] << "\t";
		f1 << endl;
	}
	f1.close();
}

void do_iteration(void)
{
	for (int i = 1; i < imax - 1; i++)
		for (int j = 1; j < jmax - 1; j++)
			new_grid[j + i * jmax] = pow(dt * c, 2.0) * ((grid[j + (i+1)*jmax] - 2.0 * grid[j + i *jmax] + grid[j + (i-1) * jmax]) / pow(dx, 2.0) + (grid[(j + 1) *jmax + i] - 2.0 * grid[j + i * jmax] + grid[(j - 1) + i * jmax]) / pow(dy, 2.0)) + 2.0 * grid[j  + jmax * i] - old_grid[j + i * jmax];

	for (int i = 0; i < imax; i++)
	{
		new_grid[i * jmax] = new_grid[1 + i * jmax];
		new_grid[(jmax - 1) + i * jmax] = new_grid[(jmax - 2) + i * jmax];
	}

	for (int j = 0; j < jmax; j++)
	{
		new_grid[j] = new_grid[j + jmax];
		new_grid[(imax-1) * imax + j] = new_grid[(imax-2) * imax + j];
	}

	t += dt;

	for (int i = 0; i < imax; i++)
	{
		for (int j = 0; j < jmax; j++) 
		{
			old_grid[j + i * imax] = new_grid[j + i * imax];
			old_grid[j + i * imax] = grid[j + i * imax];
		}
	}
}

int main(int argc, char *argv[])
{
// Record start time
auto start = std::chrono::high_resolution_clock::now();
	dx = x_max / ((double)imax - 1);
	dy = y_max / ((double)imax - 1);

	t = 0.0;

	dt = 0.1 * min(dx, dy) / c;

	int out_cnt = 0, it = 0;

	grid_to_file(out_cnt);
	out_cnt++;
	t_out += dt_out;

	//sets half sinusoidal intitial disturbance - this is brute force - it can be done more elegantly
	double r_splash = 1.0;
	double x_splash = 3.0;
	double y_splash = 3.0;
	
	for (int i = 1; i < imax - 1; i++)
		for (int j = 1; j < jmax - 1; j++)
		{
			double x = dx * i;
			double y = dy * j;

			double dist = sqrt(pow(x - x_splash, 2.0) + pow(y - y_splash, 2.0));

			if (dist < r_splash)
			{
				double h = 5.0*(cos(dist / r_splash * M_PI) + 1.0);

				grid[j + i * imax] = h;
				old_grid[j + i * imax] = h;
			}
		}

	while (t < t_max)
	{
		do_iteration();

		if (t_out <= t)
		{
			//cout << "output: " << out_cnt << "\tt: " << t << "\titeration: " << it << endl;
			grid_to_file(out_cnt);
			out_cnt++;
			t_out += dt_out;
		}

		it++;
	}
// Record end time
auto finish = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> elapsed = finish - start;
std::cout << "Elapsed time: " << elapsed.count() << " s\n";

	return 0;
    delete[] grid;
	delete[] new_grid;
	delete[] old_grid;
}
