#include <iostream>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

__global__ void f(double *t, double *x, double *res) {
	int idx = threadIdx.x;
	res[idx] = sin(2 * t[idx]) + x[idx];
}

int main()
{
	int quant = 100;
	double* t0 = new double[quant];
	double* x0 = new double[quant];
	int n = 1000;
	for (int i = 0; i < quant; i++) {
		t0[i] = rand() * 100;
		x0[i] = rand() * 100;
	}
	double* x1 = x0;
	double** k = new double*[4];
	for (int i = 0; i < 4; i++)
		k[i] = new double[quant];
	for (int i = 0; i < n; i++) {
		double *k1;
		cudaMalloc((void**) &k1, sizeof(int) * quant);
		f << < 1, quant >> > (t0, x0, k1);
		cudaMemcpy(&k[0], k1, sizeof(int) * quant, cudaMemcpyDeviceToHost);
		double *k2;
		cudaMalloc((void**)&k2, sizeof(int) * quant);
		f << < 1, quant >> > (t0, x0, k2);
		cudaMemcpy(&k[1], k2, sizeof(int) * quant, cudaMemcpyDeviceToHost);
		double *k3;
		cudaMalloc((void**) &k3, sizeof(int) * quant);
		f << <1, quant >> > (t0, x0, k3);
		cudaMemcpy(&k[2], k3, sizeof(int) * quant, cudaMemcpyDeviceToHost);
		double *k4;
		cudaMalloc((void**) &k4, sizeof(int) * quant);
		f << <1, quant >> > (t0, x0, k4);
		cudaMemcpy(&k[3], k4, sizeof(int) * quant, cudaMemcpyDeviceToHost);

		for (int j = 0; j < quant; j++) {
			x0[j] = x1[j] + (k[0][j] + 2 * k[1][j] + 2 * k[2][j] + k[3][j]) / 6;
		}
	}
	for (int i = 0; i < quant; i++) {
		cout << x0[i] << " ";
	}
	return 0;
}
