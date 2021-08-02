#define _USE_MATH_DEFINES

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <cmath>

using namespace std;

//Note that this is a very simple serial implementation with a fixed grid and Neumann boundaries at the edges
vector<vector<double>> grid, new_grid, old_grid;
int imax = 100, jmax = 100;
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
			f1 << grid[i][j] << "\t";
		f1 << endl;
	}
	f1.close();
}

void do_iteration(void)
{
	for (int i = 1; i < imax - 1; i++)
		for (int j = 1; j < jmax - 1; j++)
			new_grid[i][j] = pow(dt * c, 2.0) * ((grid[i + 1][j] - 2.0 * grid[i][j] + grid[i - 1][j]) / pow(dx, 2.0) + (grid[i][j + 1] - 2.0 * grid[i][j] + grid[i][j - 1]) / pow(dy, 2.0)) + 2.0 * grid[i][j] - old_grid[i][j];

	for (int i = 0; i < imax; i++)
	{
		new_grid[i][0] = new_grid[i][1];
		new_grid[i][jmax - 1] = new_grid[i][jmax - 2];
	}

	for (int j = 0; j < jmax; j++)
	{
		new_grid[0][j] = new_grid[1][j];
		new_grid[imax-1][j] = new_grid[imax-2][j];
	}

	t += dt;

	old_grid.swap(new_grid);
	old_grid.swap(grid);
}

int main(int argc, char *argv[])
{
	old_grid.resize(imax, vector<double>(jmax));
	grid.resize(imax, vector<double>(jmax));
	new_grid.resize(imax, vector<double>(jmax));

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

				grid[i][j] = h;
				old_grid[i][j] = h;
			}
		}

	while (t < t_max)
	{
		do_iteration();

		if (t_out <= t)
		{
			cout << "output: " << out_cnt << "\tt: " << t << "\titeration: " << it << endl;
			grid_to_file(out_cnt);
			out_cnt++;
			t_out += dt_out;
		}

		it++;
	}

	return 0;
}
