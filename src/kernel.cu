#include "xMatrix.h"

#pragma region Multiply

#define TILE_WIDTH 16	// dim3 block(TILE_HEIGHT, TILE_WIDTH);
#define TILE_HEIGHT 32

template <typename T>
// 每次都是从global显存中读取，延时很长
__global__ void NormalMulti(T* src1, T* src2, T* dst,
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

// 先将global显存中的数据读到shared内存中，之后每次都是从shared内存中读取
template<typename T>
__global__ void MultiKernelTile(T* src1, T* src2, T* dst,
	size_t pitch_src1, size_t pitch_src2, size_t pitch_dst,
	size_t M, size_t N, size_t S)
{
	__shared__ T Ads[TILE_WIDTH][TILE_WIDTH + 1];			// src1中的一个tile数据块,padding消除bank冲突
	__shared__ T Bds[TILE_WIDTH][TILE_WIDTH + 1];			// src2中的一个tile数据块
	unsigned int idx_r = blockIdx.y * blockDim.y + threadIdx.y;	// dst数据索引
	unsigned int idx_c = blockIdx.x * blockDim.x + threadIdx.x;	// 
	if (idx_r < M && idx_c < S)
	{
		size_t offset_s1 = idx_r * pitch_src1;
		size_t offset_dst = idx_r * pitch_dst;
		T* ptr_s1 = (T*)((char*)src1 + offset_s1);
		T* ptr_d = (T*)((char*)dst + offset_dst);
		T tmp = 0;
		size_t nTiles = N / TILE_WIDTH;			// 所有数据可以划分成多少个tile存储，这里N必须能被TILE_WIDTH整除
		for (size_t i = 0; i < nTiles; i++)
		{
			size_t offset_s2 = (i * TILE_WIDTH + threadIdx.y) * pitch_src2;
			T* ptr_s2 = (T*)((char*)src2 + offset_s2);

			Ads[threadIdx.y][threadIdx.x] = ptr_s1[i * TILE_WIDTH + threadIdx.x];
			Bds[threadIdx.y][threadIdx.x] = ptr_s2[idx_c];
			__syncthreads();

			for (size_t j = 0; j < TILE_WIDTH; j++)
			{
				tmp += Ads[threadIdx.x][j] * Bds[threadIdx.y][j];
			}
			__syncthreads();
		}
		ptr_d[idx_c] = tmp;
	}
}


template <typename T>
void NormalMultiWrap(T* src1, T* src2, T* dst,
	size_t pitch_src1, size_t pitch_src2, size_t pitch_dst,
	size_t M, size_t N, size_t S)
{
	rsize_t nsize = M * S;

	dim3 block(4);
	dim3 grid((nsize - 1) / block.x + 1);
	NormalMulti << <grid, block >> > (src1, src2, dst,
		pitch_src1, pitch_src2, pitch_dst,
		M, N, S);
}

template<typename T>
void MultiKernelTileWrap(T* src1, T* src2, T* dst,
	size_t pitch_src1, size_t pitch_src2, size_t pitch_dst,
	size_t M, size_t N, size_t S)
{
	dim3 block(TILE_WIDTH, TILE_WIDTH);
	dim3 grid((S - 1) / block.x + 1, (M - 1) / block.y + 1);
	MultiKernelTile << <grid, block >> > (src1, src2, dst,
		pitch_src1, pitch_src2, pitch_dst, M, N, S);
}

template xMatrix<float>;
//template xMatrix<int>;

template void multi17(const xMatrixf& A, const xMatrixf& B, xMatrixf& C);

#pragma endregion Multiply
