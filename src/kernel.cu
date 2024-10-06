#include "xMatrix.h"

template <typename T>
__global__ void WarmupMulti(T* src1, T* src2, T* dst,
	size_t pitch_src1, size_t pitch_src2, size_t pitch_dst,
	size_t M, size_t N, size_t S)
{
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int idx_r = tid / S;
	unsigned int idx_c = tid % S;

	if (idx_r < M && idx_c < S)
	{
		size_t offset_s1 = idx_r * pitch_src1;
		size_t offset_dst = idx_r * pitch_dst;
		T* ptr_s1 = (T*)((char*)src1 + offset_s1);
		T* ptr_d = (T*)((char*)dst + offset_dst);

		T tmp = 0;
		for (size_t i = 0; i < N; i++)
		{
			size_t offset_s2 = i * pitch_src2;
			T* ptr_s2 = (T*)((char*)src2 + offset_s2);

			tmp += ptr_s1[i] * ptr_s2[idx_c];
		}
		ptr_d[idx_c] = tmp;
	}
}


template <typename T>
void WarmupMultiWrap(T* src1, T* src2, T* dst,
	size_t pitch_src1, size_t pitch_src2, size_t pitch_dst,
	size_t M, size_t N, size_t S)
{
	rsize_t nsize = M * S;

	dim3 block(4);
	dim3 grid((nsize - 1) / block.x + 1);
	//TIMING("WarmupMulti")
	WarmupMulti << <grid, block >> > (src1, src2, dst,
		pitch_src1, pitch_src2, pitch_dst,
		M, N, S);
}


template xMatrix<float>;
//template xMatrix<int>;

template void multi17(const xMatrixf& A, const xMatrixf& B, xMatrixf& C);
