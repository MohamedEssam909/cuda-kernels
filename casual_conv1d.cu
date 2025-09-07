
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>



using namespace std;

template<int KERNEL_SIZE> //flexible kernel for different filter sizes without runtime overhead.
__global__ void casual_conv1d(
	const float* __restrict__ x,   // [batch, seq_len]  //tells the compiler that pointers don’t overlap → better memory optimization.
	const float* __restrict__ w,   // [KERNEL_SIZE]
	float* __restrict__ out,       // [batch, seq_len]
	int seq_len
) {
	extern __shared__ float shmem[];

	int batch = blockIdx.y;
	int tid = threadIdx.x;
	int g_start = blockIdx.x * blockDim.x;  // start index for this block
	int g_t = g_start + tid;            // global time step

	if (g_t >= seq_len) return;

	const float* x_batch = x + batch * seq_len;
	float* out_batch = out + batch * seq_len;


	// Load input into shared memory (with left padding for causality)

	int sh_idx = tid + (KERNEL_SIZE - 1); // leave K-1 slots at the left
	if (g_t < seq_len) {
		shmem[sh_idx] = x_batch[g_t];
	}
	else {
		shmem[sh_idx] = 0.0f;
	}

	if (tid < KERNEL_SIZE - 1) {
		int left_g = g_start + tid - (KERNEL_SIZE - 1);
		shmem[tid] = (left_g >= 0) ? x_batch[left_g] : 0.0f;
	}

	__syncthreads(); //all waits untill shared memory is fully written

	// Convolution
	float val = 0.0f;
	#pragma unroll //unrolls loop for speed since KERNEL_SIZE is known at compile time.
	for (int k = 0; k < KERNEL_SIZE; k++) {
		val += shmem[sh_idx - k] * w[k];
	}

	out_batch[g_t] = val;
}


int main() {
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	const int batch = 400, seq_len = 20, width = 3;

    float h_x[batch * seq_len];

    for (int i = 0; i < batch * seq_len; i++) {
        h_x[i] = i + 5 * 2 / 4;
    }

	float h_w[width] = { 0.3f,0.5f,0.1f };

	float h_out[batch * seq_len];
	
	float *d_x, *d_w, *d_out;

	cudaMalloc(&d_x, batch * seq_len * sizeof(float));
	cudaMalloc(&d_out, batch * seq_len * sizeof(float));
	cudaMalloc(&d_w, width * sizeof(float));

	cudaMemcpy(d_x, h_x, batch * seq_len * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_w, h_w, width * sizeof(float), cudaMemcpyHostToDevice);


	dim3 threads(128);
	dim3 grid((seq_len + threads.x - 1) / threads.x, batch); // grid.x= sequence length, grid.y= batch size

	size_t shmem_size = (threads.x + width - 1) * sizeof(float);

	cudaEventRecord(start);
	casual_conv1d<3> <<<grid, threads, shmem_size >>> (d_x, d_w, d_out, seq_len);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);



	cudaMemcpy(h_out, d_out, batch * seq_len * sizeof(float), cudaMemcpyDeviceToHost);


	cout << "Casual Convolution Result\n\n";

	for (int i = 0; i < batch; i++) {
		cout << "for batch " << i << "\n";
		for (int j = 0; j < seq_len;j++) {
			cout << h_out[j + i * seq_len]<<" ";
		}
		cout << "\n";
	}

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout << "\n\n\n Total Processing Time: " << milliseconds<< "ms";


}
