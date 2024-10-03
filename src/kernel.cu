#include "kernel.cuh"


__global__ void WarmupAdd(int* src1, int* src2, int*dst, size_t pitch, size_t rows, size_t cols)
{
	unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int idx_r = tid / cols;
	unsigned int idx_c = tid % cols;

	if (idx_r < rows && idx_c < cols)
	{
		size_t offset = idx_r * pitch;
		int* ptr_s1 = (int*)((char*)src1 + offset);
		int* ptr_s2 = (int*)((char*)src2 + offset);
		int* ptr_d = (int*)((char*)dst + offset);
		ptr_d[idx_c] = ptr_s1[idx_c] + ptr_s2[idx_c];
	}
}


__global__ void WarmupMulti(int* src1, int* src2, int* dst, size_t pitch_src1, size_t pitch_src2, size_t pitch_dst, size_t M, size_t N, size_t S)
{
	unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int idx_r = tid / S;
	unsigned int idx_c = tid % S;

	if (idx_r < M && idx_c < S)
	{
		size_t offset_s1 = idx_r * pitch_src1;
		size_t offset_dst = idx_r * pitch_dst;
		int* ptr_s1 = (int*)((char*)src1 + offset_s1);
		int* ptr_d = (int*)((char*)dst + offset_dst);

		int tmp = 0;
		for (size_t i = 0; i < N; i++)
		{
			size_t offset_s2 = i * pitch_src2;
			int* ptr_s2 = (int*)((char*)src2 + offset_s2);

			tmp += ptr_s1[i] * ptr_s2[idx_c];
		}
		ptr_d[idx_c] = tmp;
	}
}