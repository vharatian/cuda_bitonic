
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define ASC 1
#define DESC 2

#define DEBUGING 0
#define NOT_DEBUGING 1

#define MAX_BLOCK_DIM 65000

#define DEFAULT_SIZE 50000000

int execution_mod = NOT_DEBUGING;

int check_arguments(int argc, char** argv);
int fix_size(int size);
void bubble_sort(int* arr, int start, int end, int direction);
int generate_array(int** arr, int size);
void print_array(int* arr, int size);
void fill(int* arr, int start, int end, int value);
int checK_sort(int* arr, int size, int direction);
void print_error_message();

int up_devide(int devidend, int divisor);

void bitonic_sort(int* arr, int size, int direction);
__global__ void bitonic_device(int* arr, int size, int frame_size, int half_frame_size, int frame_assigned_count, int thread_assigned_size, int directione)
{
	if (frame_assigned_count > 1)
	{
		int block_start_element = (blockIdx.y * gridDim.x + blockIdx.x) * frame_size * frame_assigned_count;

		if (block_start_element < size)
		{
			if (block_start_element > size - (frame_size*frame_assigned_count))
				block_start_element = size - (frame_size*frame_assigned_count);

			int frame_number = threadIdx.x / half_frame_size;
			int first = block_start_element + frame_number * frame_size + (threadIdx.x % half_frame_size);
			int second = first + half_frame_size;

			if ((directione == ASC && arr[first] > arr[second]) || (directione == DESC && arr[first] < arr[second]))
			{
				int temp = arr[first];
				arr[first] = arr[second];
				arr[second] = temp;
			}
		}
	}
	else
	{
		int block_start_element = (blockIdx.y * gridDim.x + blockIdx.x) * frame_size;

		if (block_start_element < size)
		{
			if (block_start_element > size - frame_size)
				block_start_element = size - frame_size;

			int first = block_start_element + threadIdx.x * thread_assigned_size;
			int second = first + half_frame_size;

			if (threadIdx.x == blockDim.x - 1)
				thread_assigned_size = half_frame_size - (threadIdx.x  *  thread_assigned_size);

			int i;
			for (i = 0; i<thread_assigned_size; i++)
			{
				if ((directione == ASC && arr[first + i] > arr[second + i]) || (directione == DESC && arr[first + i] < arr[second + i]))
				{
					int temp = arr[first + i];
					arr[first + i] = arr[second + i];
					arr[second + i] = temp;
				}
			}
		}
	}
}

int main(int argc, char** argv)
{
	int size = check_arguments(argc, argv);
	if (size <= 0)
	{
		print_error_message();
		return 1;
	}


	int* arr;
	int fixed_size = generate_array(&arr, size);

	if (execution_mod == DEBUGING)
	{
		printf("=========== source array ============\n");
		print_array(arr, size);
		printf("\n");
	}

	printf("=========== execution started ============\n");
	time_t start_time = time(NULL);
	bitonic_sort(arr, fixed_size, ASC);
	time_t end_time = time(NULL);
	printf("=========== execution finished ============\n");

	if (execution_mod == DEBUGING)
	{
		printf("=========== result array ============\n\n");
		print_array(arr, size);
	}

	printf("\nexecution time : %d s\n\n", end_time - start_time);
	printf("result : %s\n", (checK_sort(arr, size, ASC))?"success":"failure");
	
    return 0;
}

int check_arguments(int argc, char** argv)
{
	if (argc == 1)
	{
		printf("default size selected : %d\n", DEFAULT_SIZE);
		return DEFAULT_SIZE;
	}
	else if (argc == 2)
	{
		return atoi(argv[1]);
	}
	else if (argc == 3)
	{
		if (argv[1][0] != '-' && argv[1][1] != 'd')
		{
			return -1;
		}
		else
			execution_mod = DEBUGING;
		
		return atoi(argv[2]);
	}
	else
	{
		return - 1;
	}
}

int generate_array(int** arr_ptr, int size)
{
	int fixed_size = fix_size(size);
	(*arr_ptr) = (int*)malloc(fixed_size*sizeof(int));

	int* arr = *arr_ptr;

	/*srand(time(NULL));
	int max = 3 * size;
	int i;
	for (i = 0; i < size; i++)
		arr[i] = rand() % max;

	fill(arr, size, fixed_size, INT_MAX);

	bubble_sort(arr, 0, size/2, DESC);
	bubble_sort(arr, size / 2, size, ASC);*/

	int i;
	int number = 0;
	for (i = 0; i < (size/2); i++)
	{
		arr[i] = number;
		number += 5;
	}

	for (; i < size; i++)
	{
		arr[i] = number;
		number -= 3;
	}

	return fixed_size;
}

void print_array(int* arr, int size)
{
	int i;

	printf("[");
	for (i = 0; i < size; i++)
	{
		printf(" %d", arr[i]);
	}

	printf(" ]\n");
}

void bubble_sort(int* arr, int start, int end, int direction)
{
	int i, j, temp;
	for (i = start; i < end; i++)
	{
		for (j = start; j < end - 1; j++)
		{
			if ((direction == ASC && arr[j] > arr[j+1]) || (direction == DESC &&  arr[j] < arr[j+1]))
			{
				temp = arr[j];
				arr[j] = arr[j+1];
				arr[j+1] = temp;
			}
		}
	}
}

void bitonic_sort(int* arr, int size, int direction)
{
	int* device_arr;
	cudaError_t cuda_status;

	int selected_device = 0;
	struct cudaDeviceProp properties;
	cuda_status = cudaGetDeviceProperties(&properties, selected_device);
	if (cuda_status == cudaSuccess)
	{
		cuda_status = cudaSetDevice(selected_device);
		if (cuda_status == cudaSuccess)
		{
			cuda_status = cudaMalloc((void**)&device_arr, size * sizeof(int));
			if (cuda_status == cudaSuccess)
			{
				cuda_status = cudaMemcpy(device_arr, arr, size * sizeof(int), cudaMemcpyHostToDevice);
				if (cuda_status == cudaSuccess)
				{
					dim3 block_dim;
					dim3 thread_dim;

					int max_thread = properties.maxThreadsPerMultiProcessor / 6;
					int frame_size;
					int frame_assigned_count = 1;
					int thread_assigned_size;
					int block_x;

					for (frame_size = size; frame_size > 1; frame_size /= 2)
					{
						

						if (frame_size / 2 > max_thread)
						{
							thread_dim = max_thread;
							block_x = sqrt(1.0*(size / frame_size)) + 1;
							frame_assigned_count = 1;
							thread_assigned_size = (frame_size / (2 * thread_dim.x));
						}
						else
						{
							for (frame_assigned_count = 1; (frame_size / 2) * frame_assigned_count < max_thread && size / (frame_assigned_count * frame_size) > 200; frame_assigned_count++);
							//frame_assigned_count = 1;
							thread_dim = frame_assigned_count * (frame_size / 2);
							thread_assigned_size = 1;

							block_x = sqrt(1.0*(size / (frame_assigned_count * frame_size))) + 1;
						}
						block_dim = dim3(block_x, block_x);

						bitonic_device << <block_dim, thread_dim >> >(device_arr, size, frame_size, frame_size / 2, frame_assigned_count, thread_assigned_size, direction);

						if (execution_mod == DEBUGING)
						{
							printf("block_count : %d, frame_size : %d, farme_assinge_count : %d\n", block_dim.x*block_dim.y, frame_size, frame_assigned_count);
							cudaMemcpy(arr, device_arr, size*sizeof(int), cudaMemcpyDeviceToHost);
							print_array(arr, size);
						}

						cuda_status = cudaGetLastError();
						if (cuda_status != cudaSuccess)
						{
							fprintf(stderr, "faild to run kernel function, block count : %d, thread count : %d\n", block_dim.x, thread_dim.x);
							break;
						}
					}

					cuda_status = cudaMemcpy(arr, device_arr, size*sizeof(int), cudaMemcpyDeviceToHost);
					if (cuda_status != cudaSuccess)
						fprintf(stderr, "faild to copy arrry from device to host, array size : %d\n" + size * sizeof(int));
				}
				else
					fprintf(stderr, "faild to copy array from host to device, array size : %d\n", size * sizeof(int));
			}
			else
				fprintf(stderr, "failed to allocate memory on device, requested memoy : %d\n", size * sizeof(int));
		}
		else
			fprintf(stderr, "failed to select cuda capable device, device num : %d\n", selected_device);
	}
	else
		fprintf(stderr, "failed to read device properies, device num : %d\n", selected_device);
	
	

	cudaFree(device_arr);
}

int up_devide(int devidend, int divisor)
{
	int result = devidend / divisor;
	/*if (devidend % divisor != 0)
		result++;*/

	return result;
}

int checK_sort(int* arr, int size, int direction)
{
	int i;
	for (i = 0; i < size - 1; i++)
		if ((direction == ASC && arr[i] > arr[i + 1]) || (direction == DESC && arr[i] < arr[i + 1]))
			return false;

	return true;
}


int fix_size(int size)
{
	int result;
	for (result = 2; result < size; result *= 2);

	return result;
}

void fill(int* arr, int start, int end, int value)
{
	int i;
	for (i = start; i < end; i++)
		arr[i] = value;
}

void print_error_message()
{
	printf("wrong command format.\n betonic [option] [size] \n option : \n \t -d debug mod\n");
}
