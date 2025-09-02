
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>


using namespace std;


__global__ void perceptron(const float* W, const float* x, float b, float* out, int n){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		atomicAdd(out, W[idx] * x[idx]); //thread-safe addition.
	}
}

int main() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    const int N = 1024;
    //float h_W[N] = { 1,2,3,4,5,6,7,8,9,10 };
    //float h_x[N] = { 1,2,3,4,5,6,7,8,9,10 };
    float b = 1.5f;
    float h_out = 0.0f;

    h_out = b;

    float h_W[N], h_x[N];

    for (int i = 0; i < N; i++) {
        h_W[i] = i;
        h_x[i] = i;
    }



    float* d_W, * d_x, * d_out;

    cudaMalloc(&d_W, N * sizeof(float));
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));

    cudaMemcpy(d_W, h_W, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, &h_out, sizeof(float), cudaMemcpyHostToDevice);


    // launch kernel
    int threads = 256;
    int blocks = (N + threads - 1) / threads; //ceil(blocksize/N)
    int REPEATS = 100;
    float milliseconds = 0;

    cudaEventRecord(start);
    for (int i = 0; i < REPEATS; i++) {
        h_out = b;
        cudaMemcpy(d_out, &h_out, sizeof(float), cudaMemcpyHostToDevice);
        perceptron <<<blocks, threads >>> (d_W, d_x, b, d_out, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);


    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Total Elapsed Time: %f ms\n\n\n", milliseconds / REPEATS);



    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);





    cout << "W vector [";

    for (int i = 0; i < N; i++) {
        cout << h_W[i]<<",";
    }
    cout << "]\n\n\n";


    cout << "x vector [";

    for (int i = 0; i < N; i++) {
        cout << h_x[i] << ",";
    }
    cout << "]\n\n\n";



    printf("Perceptron output: %f\n", h_out);

    cudaFree(d_W);
    cudaFree(d_x);
    cudaFree(d_out);

    return 0;
}