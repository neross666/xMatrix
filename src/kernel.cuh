﻿#pragma once
#include <cuda_runtime.h>

#define TILE_WIDTH 16	// dim3 block(TILE_HEIGHT, TILE_WIDTH);
#define TILE_HEIGHT 32

#pragma region 加法运算
// GPU预热，使用普通计算方式
__global__ void WarmupAdd(int* src1, int* src2, int*dst, size_t pitch, size_t rows, size_t cols);

// 使用普通计算方式，每个数据有且仅有一个线程处理，将线程排列理解为线性的
template<typename T>
__global__ void AddKernel(T* src1, T* src2, T*dst, size_t pitch, size_t rows, size_t cols)
{
	// grid 1D; block 1D;
	unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;

	// grid 2D; block 1D;
	//unsigned int bid = blockIdx.y * gridDim.x + blockIdx.x;
	//unsigned int tid = bid * blockDim.x + threadIdx.x;

	// grid 2D; block 2D;
	//unsigned int bid = blockIdx.y * gridDim.x + blockIdx.x;
	//unsigned int tid = bid * (blockDim.x * blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;

	// 求取该线程号对应数据所在二维矩阵中的行列索引
	unsigned int idx_r = tid / cols;
	unsigned int idx_c = tid % cols;
	if (idx_r < rows && idx_c < cols)
	{
		size_t offset = idx_r * pitch;
		T* ptr_s1 = (T*)((char*)src1 + offset);
		T* ptr_s2 = (T*)((char*)src2 + offset);
		T* ptr_d = (T*)((char*)dst + offset);
		ptr_d[idx_c] = ptr_s1[idx_c] + ptr_s2[idx_c];
	}
}

// 使用普通计算方式，每个数据有且仅有一个线程处理，将线程排列理解为二维的
template<typename T>
__global__ void AddKernelV2(T* src1, T* src2, T*dst, size_t pitch, size_t rows, size_t cols)
{
	// 线程y方向索引对应数据行方向索引，线程x方向索引对应数据列方向索引
	unsigned int idx_r = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int idx_c = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx_r < rows && idx_c < cols)
	{
		size_t offset = idx_r * pitch;
		T* ptr_s1 = (T*)((char*)src1 + offset);
		T* ptr_s2 = (T*)((char*)src2 + offset);
		T* ptr_d = (T*)((char*)dst + offset);
		ptr_d[idx_c] = ptr_s1[idx_c] + ptr_s2[idx_c];
	}
}

// 使用4重循环展开计算方式一，同一行的相邻4列数据被一个线程处理，将线程排列理解为线性的
template<typename T>
__global__ void Add4UnRollingKernelV1(T* src1, T* src2, T*dst, size_t pitch, size_t rows, size_t cols)
{
	// 	{
	// 		// grid 1D; block 1D; 数据使用一维连续存储方式，两个相邻行之间不存在间隙
	// 		unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
	// 		unsigned int idx = 4 * tid;
	// 		if (idx + 3 < size)
	// 		{
	// 			dst[idx] = src1[idx] + src2[idx];
	// 			dst[idx + 1] = src1[idx + 1] + src2[idx + 1];
	// 			dst[idx + 2] = src1[idx + 2] + src2[idx + 2];
	// 			dst[idx + 3] = src1[idx + 3] + src2[idx + 3];
	// 		}
	// 	}

		// 数据使用一维非连续存储方式，两个相邻行之间存在间隙
		// 求取该线程号对应数据所在二维矩阵中的行列索引
	unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int idx_r = tid / (cols / 4);
	unsigned int idx_c = 4 * (tid % (cols / 4));		// 列数为4的倍数	

	if (idx_r < rows && (idx_c + 3) < cols)
	{
		size_t offset = idx_r * pitch;
		T* ptr_s1 = (T*)((char*)src1 + offset);
		T* ptr_s2 = (T*)((char*)src2 + offset);
		T* ptr_d = (T*)((char*)dst + offset);
		ptr_d[idx_c] = ptr_s1[idx_c] + ptr_s2[idx_c];
		ptr_d[idx_c + 1] = ptr_s1[idx_c + 1] + ptr_s2[idx_c + 1];
		ptr_d[idx_c + 2] = ptr_s1[idx_c + 2] + ptr_s2[idx_c + 2];
		ptr_d[idx_c + 3] = ptr_s1[idx_c + 3] + ptr_s2[idx_c + 3];
	}
}

// 使用4重循环展开计算方式二，同一列的相邻4行数据被一个线程处理，将线程排列理解为线性的
template<typename T>
__global__ void Add4UnRollingKernelV2(T* src1, T* src2, T*dst, size_t pitch, size_t rows, size_t cols)
{
	// grid 1D; block 1D; 数据使用一维连续存储方式，两个相邻行之间不存在间隙
//	unsigned int tid = threadIdx.x + 4*blockIdx.x*blockDim.x;
// 	if (tid + 3 * blockDim.x < size)
// 	{
// 		dst[tid] = src1[tid] + src2[tid];
// 		dst[tid + blockDim.x] = src1[tid + blockDim.x] + src2[tid + blockDim.x];
// 		dst[tid + 2 * blockDim.x] = src1[tid + 2 * blockDim.x] + src2[tid + 2 * blockDim.x];
// 		dst[tid + 3 * blockDim.x] = src1[tid + 3 * blockDim.x] + src2[tid + 3 * blockDim.x];
// 	}

	// 数据使用一维非连续存储方式，两个相邻行之间存在间隙
	// 求取该线程号对应数据所在二维矩阵中的行列索引
	unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int idx_r = 4 * (tid / cols);
	unsigned int idx_c = tid % cols;
	if (idx_r + 3 < rows && idx_c < cols)
	{
		size_t offset = idx_r * pitch;
		T* ptr_s1 = (T*)((char*)src1 + offset);
		T* ptr_s2 = (T*)((char*)src2 + offset);
		T* ptr_d = (T*)((char*)dst + offset);
		ptr_d[idx_c] = ptr_s1[idx_c] + ptr_s2[idx_c];

		offset = (idx_r + 1) * pitch;
		ptr_s1 = (T*)((char*)src1 + offset);
		ptr_s2 = (T*)((char*)src2 + offset);
		ptr_d = (T*)((char*)dst + offset);
		ptr_d[idx_c] = ptr_s1[idx_c] + ptr_s2[idx_c];

		offset = (idx_r + 2) * pitch;
		ptr_s1 = (T*)((char*)src1 + offset);
		ptr_s2 = (T*)((char*)src2 + offset);
		ptr_d = (T*)((char*)dst + offset);
		ptr_d[idx_c] = ptr_s1[idx_c] + ptr_s2[idx_c];

		offset = (idx_r + 3) * pitch;
		ptr_s1 = (T*)((char*)src1 + offset);
		ptr_s2 = (T*)((char*)src2 + offset);
		ptr_d = (T*)((char*)dst + offset);
		ptr_d[idx_c] = ptr_s1[idx_c] + ptr_s2[idx_c];
	}
}

// 使用4重循环展开计算方式三，同一行的相邻4列数据被一个线程处理，将线程排列理解为二维的
template<typename T>
__global__ void Add4UnRollingKernelV3(T* src1, T* src2, T*dst, size_t pitch, size_t rows, size_t cols)
{
	// 线程y方向索引对应数据行方向索引，线程x方向索引对应4列数据起始索引
	unsigned int tid_y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int tid_x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int idx_r = tid_y;
	unsigned int idx_c = tid_x << 2;	// 列数为4的倍数

	if (idx_r < rows && (idx_c + 3) < cols)
	{
		size_t offset = idx_r * pitch;
		T* ptr_s1 = (T*)((char*)src1 + offset);
		T* ptr_s2 = (T*)((char*)src2 + offset);
		T* ptr_d = (T*)((char*)dst + offset);
		ptr_d[idx_c] = ptr_s1[idx_c] + ptr_s2[idx_c];
		ptr_d[idx_c + 1] = ptr_s1[idx_c + 1] + ptr_s2[idx_c + 1];
		ptr_d[idx_c + 2] = ptr_s1[idx_c + 2] + ptr_s2[idx_c + 2];
		ptr_d[idx_c + 3] = ptr_s1[idx_c + 3] + ptr_s2[idx_c + 3];
	}
}

// 使用4重循环展开计算方式四，同一行的相邻4列数据被一个线程处理，将线程排列理解为二维的
template<typename T>
__global__ void Add4UnRollingKernelV4(T* src1, T* src2, T*dst, size_t pitch, size_t rows, size_t cols)
{
	// 线程y方向索引对应数据行方向索引，线程x方向索引对应4列数据起始索引
	unsigned int tid_y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int tid_x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int idx_r = tid_y << 2;		// 行数为4的倍数
	unsigned int idx_c = tid_x;
	if (idx_r + 3 < rows && idx_c < cols)
	{
		size_t offset = idx_r * pitch;
		T* ptr_s1 = (T*)((char*)src1 + offset);
		T* ptr_s2 = (T*)((char*)src2 + offset);
		T* ptr_d = (T*)((char*)dst + offset);
		ptr_d[idx_c] = ptr_s1[idx_c] + ptr_s2[idx_c];

		offset = (idx_r + 1) * pitch;
		ptr_s1 = (T*)((char*)src1 + offset);
		ptr_s2 = (T*)((char*)src2 + offset);
		ptr_d = (T*)((char*)dst + offset);
		ptr_d[idx_c] = ptr_s1[idx_c] + ptr_s2[idx_c];

		offset = (idx_r + 2) * pitch;
		ptr_s1 = (T*)((char*)src1 + offset);
		ptr_s2 = (T*)((char*)src2 + offset);
		ptr_d = (T*)((char*)dst + offset);
		ptr_d[idx_c] = ptr_s1[idx_c] + ptr_s2[idx_c];

		offset = (idx_r + 3) * pitch;
		ptr_s1 = (T*)((char*)src1 + offset);
		ptr_s2 = (T*)((char*)src2 + offset);
		ptr_d = (T*)((char*)dst + offset);
		ptr_d[idx_c] = ptr_s1[idx_c] + ptr_s2[idx_c];
	}
}
#pragma endregion 加法运算


#pragma region 乘法运算
__global__ void WarmupMulti(int* src1, int* src2, int*dst,
	size_t pitch_src1, size_t pitch_src2, size_t pitch_dst,
	size_t M, size_t N, size_t S);

// 每次都是从global显存中读取，延时很长
template<typename T>
__global__ void MultiKernel(T* src1, T* src2, T*dst,
	size_t pitch_src1, size_t pitch_src2, size_t pitch_dst,
	size_t M, size_t N, size_t S)
{
	unsigned int idx_r = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int idx_c = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (idx_r < M && idx_c < S)
	{
		size_t offset_s1 = idx_r * pitch_src1;
		size_t offset_dst = idx_r * pitch_dst;
		T* ptr_s1 = (T*)((char*)src1 + offset_s1);
		T* ptr_d = (T*)((char*)dst + offset_dst);

		T tmp = 0;	// 使用寄存器暂存
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
__global__ void MultiKernelTile(T* src1, T* src2, T*dst,
	size_t pitch_src1, size_t pitch_src2, size_t pitch_dst,
	size_t M, size_t N, size_t S)
{
	__shared__ T Ads[TILE_WIDTH][TILE_WIDTH+1];			// src1中的一个tile数据块,padding消除bank冲突
	__shared__ T Bds[TILE_WIDTH][TILE_WIDTH+1];			// src2中的一个tile数据块
	unsigned int idx_r = blockIdx.y*blockDim.y + threadIdx.y;	// dst数据索引
	unsigned int idx_c = blockIdx.x*blockDim.x + threadIdx.x;	// 
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
			size_t offset_s2 = (i*TILE_WIDTH + threadIdx.y) * pitch_src2;
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
#pragma endregion 乘法运算


#pragma region 遍历
// nvprof -m gld_transactions_per_request,gst_transactions_per_request HpcLabs.exe
template<typename T>
__global__ void TraverseKernelRow(T* src, T* dst, size_t pitch, size_t rows, size_t cols)
{
	unsigned int idx_r = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int idx_c = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx_r < rows && idx_c < cols)
	{
		size_t offset = idx_r * pitch;
		T* ptr_s = (T*)((char*)src + offset);
		T* ptr_d = (T*)((char*)dst + offset);
		ptr_d[idx_c] = ptr_s[idx_c];		// 行主序读写
	}
}

template<typename T>
__global__ void TraverseKernelCol(T* src, T* dst, size_t pitch, size_t rows, size_t cols)
{
	unsigned int idx_r = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int idx_c = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (idx_r < rows && idx_c < cols)
	{
		size_t idx = idx_c * rows + idx_r;
		dst[idx] = src[idx];			// 列主序读写
	}
}

// bank_no = (addr/4)%32
// nvprof --metrics shared_load_transactions_per_request,shared_store_transactions_per_request
template<typename T>
__global__ void TraverseKernelSMEM(T* src, T* dst, size_t pitch, size_t rows, size_t cols)
{
	__shared__ T tile[TILE_WIDTH][TILE_WIDTH];
	unsigned int idx_r = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int idx_c = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx_r < rows && idx_c < cols)
	{
		size_t offset = idx_r * pitch;
		T* ptr_s = (T*)((char*)src + offset);
		T* ptr_d = (T*)((char*)dst + offset);

		tile[threadIdx.y][threadIdx.x] = ptr_s[idx_c];		// 行主序写
		__syncthreads();

		ptr_d[idx_c] = tile[threadIdx.y][threadIdx.x];		// 行主序写
	}
}

template<typename T>
__global__ void TraverseKernelSMEMRect(T* src, T* dst, size_t pitch, size_t rows, size_t cols)
{
	__shared__ T tile[TILE_WIDTH][TILE_HEIGHT];
	unsigned int idx_r = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int idx_c = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx_r < rows && idx_c < cols)
	{
		size_t offset = idx_r * pitch;
		T* ptr_s = (T*)((char*)src + offset);
		T* ptr_d = (T*)((char*)dst + offset);

		tile[threadIdx.y][threadIdx.x] = ptr_s[idx_c];		// 行主序写
		__syncthreads();

		unsigned int idx = threadIdx.y*blockDim.x + threadIdx.x;
		unsigned int irow = idx / blockDim.y;
		unsigned int icol = idx % blockDim.y;

		ptr_d[idx_c] = tile[icol][irow];					// 列主序读
	}
}
#pragma endregion 遍历


template<typename T>
__global__ void TransformationKernel(T* src, T* dst, float pos, float bias, size_t pitch, size_t rows, size_t cols)
{
	unsigned int idx_r = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int idx_c = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx_r < rows && idx_c < cols)
	{
		size_t offset = idx_r * pitch;
		T* ptr_d = (T*)((char*)dst + offset);
		T* ptr_s = (T*)((char*)src + offset);
		ptr_d[idx_c] = (T)(pos * ptr_s[idx_c] + bias);
	}
}

#pragma region 平滑
template<typename T>
__global__ void SmoothKernel(T* src, T* dst, float* coef, size_t pitch, size_t rows, size_t cols)
{
	unsigned int idx_r = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int idx_c = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx_r < rows && idx_c < cols)
	{
		float tmp = 0.0f;
		for (int i = -3; i <= 3; i++)
		{
			size_t rr = idx_r;
			if (idx_r + i < 0 || idx_r + i >= rows)
				rr = idx_r - i;
			size_t offset_s = rr * pitch;
			T* ptr_s = (T*)((char*)src + offset_s);

			float* ptr_coef = coef + (i + 3) * 7;
			for (int j = -3; j <= 3; j++)
			{
				size_t cc = idx_c;
				if (idx_c + j < 0 || idx_c + j >= cols)
					cc = idx_c - j;

				tmp += ptr_coef[j + 3] * ptr_s[cc];
			}
		}
		size_t offset_d = idx_r * pitch;
		T* ptr_d = (T*)((char*)dst + offset_d);
		ptr_d[idx_c] = (T)(tmp / 49);
	}
}

__constant__ float coef[49];
template<typename T>
__global__ void SmoothKernelCMEM(T* src, T* dst, size_t pitch, size_t rows, size_t cols)
{
	unsigned int idx_r = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int idx_c = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx_r < rows && idx_c < cols)
	{
		float tmp = 0.0f;
		for (int i = -3; i <= 3; i++)
		{
			size_t rr = idx_r;
			if (idx_r + i < 0 || idx_r + i >= rows)
				rr = idx_r - i;
			size_t offset_s = rr * pitch;
			T* ptr_s = (T*)((char*)src + offset_s);

			float* ptr_coef = coef + (i + 3) * 7;
			for (int j = -3; j <= 3; j++)
			{
				size_t cc = idx_c;
				if (idx_c + j < 0 || idx_c + j >= cols)
					cc = idx_c - j;
				tmp += ptr_coef[j + 3] * ptr_s[cc];
			}
		}
		size_t offset_d = idx_r * pitch;
		T* ptr_d = (T*)((char*)dst + offset_d);
		ptr_d[idx_c] = (T)(tmp/49);
	}
}

template<typename T>
__global__ void SmoothKernelSMEM(T* src, T* dst, size_t pitch, size_t rows, size_t cols)
{
	__shared__ T tile[TILE_WIDTH + 3][TILE_WIDTH + 3];
	unsigned int idx_r = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int idx_c = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx_r < rows && idx_c < cols)
	{
		size_t offset = idx_r * pitch;
		T* ptr_s = (T*)((char*)src + offset);
		T* ptr_d = (T*)((char*)dst + offset);

		tile[threadIdx.y][threadIdx.x] = ptr_s[idx_c];		// 行主序写
		__syncthreads();

		// 对当前块进行平滑
		float tmp = 0;
		for (int i = -3; i < 3; i++)
		{
			float* ptr_coef = coef + (i + 3) * 7;
			for (int j = -3; j < 3; j++)
			{
				tmp += ptr_coef[j + 3] * tile[threadIdx.y + i][threadIdx.x + j];
			}
		}
		ptr_d[idx_c] = (T)tmp;


		ptr_d[idx_c] = tile[threadIdx.y][threadIdx.x];		// 行主序写
	}
}
#pragma endregion 平滑

#pragma region 转置
template<typename T>
__global__ void TranspositionKernel(T* src, T* dst, size_t pitch_s, size_t pitch_d, size_t rows, size_t cols)
{
	unsigned int idx_x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int idx_y = blockIdx.y*blockDim.y + threadIdx.y;

	if (idx_y < rows && idx_x < cols)
	{
		size_t offset_s = idx_x * pitch_s;
		size_t offset_d = idx_y * pitch_d;
		T* ptr_s = (T*)((char*)src + offset_s);			   
		T* ptr_d = (T*)((char*)dst + offset_d);
		ptr_d[idx_x] = ptr_s[idx_y];
	}
}

// 从nvprof中观察到，无论是按行读取还是按列读取，全局内存的读取事务都是一样的，why？
template<typename T>
__global__ void TranspositionKernelSMEM(T* src, T* dst, size_t pitch_s, size_t pitch_d, size_t rows, size_t cols)
{
	__shared__ float tile[TILE_WIDTH][TILE_HEIGHT+2];
	unsigned int idx_x = blockDim.x*blockIdx.x + threadIdx.x;	// 全局x坐标
	unsigned int idx_y = blockDim.y*blockIdx.y + threadIdx.y;	// 全局y坐标
	size_t offset_s = idx_y * pitch_s;
	T* ptr_s = (T*)((char*)src + offset_s);
	   	
	unsigned int bidx, irow, icol;
	bidx = threadIdx.y*blockDim.x + threadIdx.x;			// 当前块的线程索引
	irow = bidx / blockDim.y;								// 转置后的tile的行索引，如果tile为方阵，则等于threadIdx.y
	icol = bidx % blockDim.y;								// 转置后的tile的列索引，如果tile为方阵，则等于threadIdx.x

	unsigned int ix_d = blockIdx.y*blockDim.y + icol;		// 转置后的全局x坐标
	unsigned int iy_d = blockIdx.x*blockDim.x + irow;		// 转置后的全局y坐标
	size_t offset_d = iy_d * pitch_d;
	T* ptr_d = (T*)((char*)dst + offset_d);

	if (iy_d < cols && ix_d < rows)
	{
		tile[threadIdx.y][threadIdx.x] = ptr_s[idx_x];		// 是用tile[threadIdx.y][threadIdx.x]，而非tile[threadIdx.x][threadIdx.y]，是为了消测bank写冲突
		__syncthreads();
		ptr_d[ix_d] = tile[icol][irow];						// 读出tile中的一列写入dst中的一行，实现转置

	}
}
#pragma endregion 转置


texture<int, cudaTextureType2D, cudaReadModeElementType> texRefA;
texture<int, cudaTextureType2D, cudaReadModeElementType> texRefB;
template<typename T>
__global__ void BlendKernel(T*dst, size_t pitch, size_t rows, size_t cols)
{
	unsigned int idx_x = blockDim.x*blockIdx.x + threadIdx.x;
	unsigned int idx_y = blockDim.y*blockIdx.y + threadIdx.y;
	if (idx_x < cols && idx_y < rows)
	{
		size_t offset = idx_y * pitch;
		T* ptr_dst = (T*)((char*)dst + offset);
		ptr_dst[idx_x] = (T)(0.5*tex2D(texRefA, idx_x, idx_y) + 0.5*tex2D(texRefB, idx_x, idx_y));
	}
}