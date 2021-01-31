#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <opencv2/highgui.hpp>

#define BLOCK_SIZE (4)

using namespace cv;

__device__  __host__ void swap(unsigned char* a, unsigned char* b) {
	unsigned char temp = *a;
	*a = *b;
	*b = temp;
}

__device__ void sort(unsigned char arr[], int left, int right) {
	for (int i = left; i <= right; ++i)
		for (int j = i + 1; j <= right; ++j)
			if (arr[i] > arr[j])
				swap(&arr[i], &arr[j]);
}

__global__ void runMedianFilterSharedKernel(unsigned char* input, unsigned char* output, int img_width, int img_height)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int column = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ unsigned char shared_memory[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

	bool left_thread = (threadIdx.x == 0), right_thread = (threadIdx.x == BLOCK_SIZE - 1);
	bool top_thread = (threadIdx.y == 0), bot_thread = (threadIdx.y == BLOCK_SIZE - 1);

	if (left_thread)
		shared_memory[threadIdx.x][threadIdx.y + 1] = 0;
	else if (right_thread)
		shared_memory[threadIdx.x + 2][threadIdx.y + 1] = 0;
	if (top_thread) {
		shared_memory[threadIdx.x + 1][threadIdx.y] = 0;
		if (left_thread)
			shared_memory[threadIdx.x][threadIdx.y] = 0;
		else if (right_thread)
			shared_memory[threadIdx.x + 2][threadIdx.y] = 0;
	}
	else if (bot_thread) {
		shared_memory[threadIdx.x + 1][threadIdx.y + 2] = 0;
		if (right_thread)
			shared_memory[threadIdx.x + 2][threadIdx.y + 2] = 0;
		else if (left_thread)
			shared_memory[threadIdx.x][threadIdx.y + 2] = 0;
	}

	shared_memory[threadIdx.y + 1][threadIdx.x + 1] = input[row * img_width + column];

	if (left_thread && (column > 0))
		shared_memory[threadIdx.x][threadIdx.y + 1] = input[row * img_width + (column - 1)];
	else if (right_thread && (column < img_width - 1))
		shared_memory[threadIdx.x + 2][threadIdx.y + 1] = input[row * img_width + (column + 1)];
	if (top_thread && (row > 0)) {
		shared_memory[threadIdx.x + 1][threadIdx.y] = input[(row - 1) * img_width + column];
		if (left_thread)
			shared_memory[threadIdx.x][threadIdx.y] = input[(row - 1) * img_width + (column - 1)];
		else if (right_thread)
			shared_memory[threadIdx.x + 2][threadIdx.y] = input[(row - 1) * img_width + (column + 1)];
	}
	else if (bot_thread && (row < img_height - 1)) {
		shared_memory[threadIdx.x + 1][threadIdx.y + 2] = input[(row + 1) * img_width + column];
		if (right_thread)
			shared_memory[threadIdx.x + 2][threadIdx.y + 2] = input[(row + 1) * img_width + (column + 1)];
		else if (left_thread)
			shared_memory[threadIdx.x][threadIdx.y + 2] = input[(row + 1) * img_width + (column - 1)];
	}

	__syncthreads();

	unsigned char median_filter[9] = { shared_memory[threadIdx.x + 0][threadIdx.y + 0], shared_memory[threadIdx.x + 0][threadIdx.y + 1], shared_memory[threadIdx.x + 0][threadIdx.y + 2],
									   shared_memory[threadIdx.x + 1][threadIdx.y + 0], shared_memory[threadIdx.x + 1][threadIdx.y + 1], shared_memory[threadIdx.x + 1][threadIdx.y + 2],
									   shared_memory[threadIdx.x + 2][threadIdx.y + 0], shared_memory[threadIdx.x + 2][threadIdx.y + 1], shared_memory[threadIdx.x + 2][threadIdx.y + 2] };

	sort(median_filter, 0, 8);

	output[row * img_width + column] = median_filter[4];
}

float MedianFilter(Mat input, Mat& output) {
	int img_width = input.cols, img_height = input.rows, img_size = img_width * img_height * sizeof(unsigned char);
	unsigned char* device_input_img = NULL;
	unsigned char* device_output_img;

	cudaMalloc((void**)&device_input_img, img_size);
	cudaMalloc((void**)&device_output_img, img_size);

	cudaMemcpy(device_input_img, input.data, img_size, cudaMemcpyHostToDevice);

	dim3 grid_dim((int)ceil((float)img_width / (float)BLOCK_SIZE), (int)ceil((float)img_height / (float)BLOCK_SIZE));
	dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	runMedianFilterSharedKernel <<<grid_dim, block_dim>>> (device_input_img, device_output_img, img_width, img_height);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaMemcpy(output.data, device_output_img, img_size, cudaMemcpyDeviceToHost);

	cudaFree(device_input_img);
	cudaFree(device_output_img);

	float time = 0;
	cudaEventElapsedTime(&time, start, stop);
	return time / 1000;
}
