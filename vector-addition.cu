
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cuda_runtime.h>

#include <chrono>  // for CPU timing

__global__ void vectorAdd(const int* a, const int* b, int* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }

}

using namespace std;
int main() {
    const int N = 256;
    size_t size = N * sizeof(int);

    int* h_a = new int[N];
    int* h_b = new int[N];
    int* h_c = new int[N];

    for (int i = 0; i < N; i++) {
        h_a[i] = i + 2;
        h_b[i] = i * 2;
    }


    int* d_a, * d_b, * d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);


    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 64;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; //ceil(N/threadsPerBlock)

    vectorAdd<<< blocksPerGrid, threadsPerBlock >>> (d_a, d_b, d_c, N);


    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    cout << "Vector Addition Results (first 100):\n";

    for (int i = 0; i < 100; i++) {
        cout << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << "\n";
    }

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;

}