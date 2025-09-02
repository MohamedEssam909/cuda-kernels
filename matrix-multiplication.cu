
/*
pointer-to-pointer (float**) approach works and is nice for clarity, 
but in real-world CUDA programming, people almost always use flattened 1D arrays (float*).
Reason: extra indirection (row pointers) = slower global memory access.
Shared memory + flat arrays is the high-performance way.
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "iostream"
#include <stdio.h>
#include <ctime>
using namespace std;

__global__ void matrixMultiplication2D(float **A, float **B, float **C, int N) {
	int row_idx = threadIdx.y + blockIdx.y * blockDim.y;
	int column_idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (row_idx < N && column_idx < N) {
		float sum = 0;
		for (int k = 0;k < N;k++) {
			sum += A[row_idx][k] * B[k][column_idx];
		}
		C[row_idx][column_idx] = sum;
	}
}

int main() {
	constexpr int N = 4;  //guarantees N is a true compile-time constant
	int matrix_size = N * N;
	int matrix_size_in_bytes = matrix_size * sizeof(float);


	float h_A[N][N];
	float h_B[N][N];
	//float h_A[N][N] = {
	//	{1,2,3},
	//	{4,5,6},
	//	{7,8,9}
	//};
	//float h_B[N][N] = {
	//{9, 8, 7},
	//{6, 5, 4},
	//{3, 2, 1}
	//};
	
	float h_C[N][N];


	



	int value = 1;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			h_A[i][j] = value;
			h_B[i][j] = value;
			value++;
		}
	}
	
	float **d_A, **d_B, **d_C; //arrays of row pointers (float**).
	float *d_A_data, *d_B_data, *d_C_data; //contiguous blocks storing all numbers.

	cudaMalloc((void**)&d_A, N * sizeof(float*));
	cudaMalloc((void**)&d_B, N * sizeof(float*));
	cudaMalloc((void**)&d_C, N * sizeof(float*));

	cudaMalloc((void**)&d_A_data, matrix_size_in_bytes);
	cudaMalloc((void**)&d_B_data, matrix_size_in_bytes);
	cudaMalloc((void**)&d_C_data, matrix_size_in_bytes);


	cudaMemcpy(d_A_data, h_A, matrix_size_in_bytes, cudaMemcpyHostToDevice); 
	cudaMemcpy(d_B_data, h_B, matrix_size_in_bytes, cudaMemcpyHostToDevice);

	// CPU-side row pointers that point inside GPU contiguous blocks.
	float* h_A_ptrs[N]; 
	float* h_B_ptrs[N];
	float* h_C_ptrs[N];
	for (int i = 0; i < N; i++) {
		h_A_ptrs[i] = d_A_data + i * N;  // memory location + offset
		h_B_ptrs[i] = d_B_data + i * N;  // memory location + offset
		h_C_ptrs[i] = d_C_data + i * N;  // memory location + offset
	}

	cudaMemcpy(d_A, h_A_ptrs, N * sizeof(float*), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B_ptrs, N * sizeof(float*), cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C_ptrs, N * sizeof(float*), cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid( ( N + 15 ) / 16, (N + 15) / 16); // ensuring we cover all rows and columns(rounds up).


	clock_t start = clock();
	matrixMultiplication2D<<<blocksPerGrid, threadsPerBlock >>> (d_A, d_B, d_C, N);
	cudaDeviceSynchronize();
	clock_t end = clock();

	cout << "\nTime Taken : " << double(end - start) / CLOCKS_PER_SEC<< "\n";

	cudaMemcpy(h_C, d_C_data, matrix_size_in_bytes, cudaMemcpyDeviceToHost);


	
	cout << "Matrix A" << "\n";
	for (int i = 0;i < N;i++) {
		cout << "|\t";
		for (int j = 0; j < N; j++) {
			cout << h_A[i][j] << "\t";
		}
		cout << "|";
		cout << "\n";
	}

	cout << "\n";

	cout << "Matrix B" << "\n";
	for (int i = 0;i < N;i++) {
		cout << "|\t";
		for (int j = 0; j < N; j++) {
			cout << h_B[i][j] << "\t";
		}
		cout << "|";
		cout << "\n";
	}

	cout << "\n";

	cout << "Matrix C" << "\n";

	for (int i = 0;i < N;i++) {
		cout << "|\t";
		for (int j = 0; j < N; j++) {
			cout << h_C[i][j] << "\t";
		}
		cout << "|";
		cout << "\n";
	}
	
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaFree(d_A_data);
	cudaFree(d_B_data);
	cudaFree(d_C_data);
	return 0;



}