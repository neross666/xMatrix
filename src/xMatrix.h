#pragma once
#include <memory>
#include <random>
#include <xmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>
#include <omp.h>
//#include <cblas.h>
#include <iostream>
#include <assert.h>
#include "util.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

template <class T>
void WarmupMultiWrap(T* src1, T* src2, T* dst,
	size_t pitch_src1, size_t pitch_src2, size_t pitch_dst,
	size_t M, size_t N, size_t S);



template <typename T>
class xMatrix
{
public:
	xMatrix(int rows, int cols, bool bRst = true) : m_rows(rows), m_cols(cols) {
		m_pitch = ((8 * sizeof(T) * cols + 31) >> 5) << 2;		// 4字节对齐后一行数据所占字节数
		m_pData = (T*)_aligned_malloc(m_pitch * rows, 4);		// 起始地址也要4字节对齐
		assert((unsigned long long)m_pData % 4 == 0);
		if (bRst)
			memset(m_pData, 0, rows * cols * sizeof(T));
	}
	~xMatrix() {
		_aligned_free(m_pData);
	}

	static std::unique_ptr<xMatrix> makeRandMat(int rows, int cols) {
		auto mat = std::make_unique<xMatrix<T>>(rows, cols);

		std::random_device rd;
		std::default_random_engine eng(rd());
		std::uniform_real_distribution<T> distr(0.01f, 1.0f);
		//m_pData = new float[row*col];

		mat->m_pData = (T*)_aligned_malloc(rows * cols * sizeof(float), 32);
		assert(mat->m_pData != nullptr);
		assert((unsigned long long)mat->m_pData % 32 == 0);

		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				mat->m_pData[i * cols + j] = distr(eng);
			}
		}

		return std::move(mat);
	}

	int m_rows;
	int m_cols;
	int m_pitch;
	T* m_pData;
};

using xMatrixf = xMatrix<float>;


template<typename T>
inline void print(const xMatrix<T>& m)
{
	for (int i = 0; i < m.m_rows; ++i)
	{
		for (int j = 0; j < m.m_cols; ++j)
		{
			std::cout << m.m_pData[i * m.m_cols + j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

template<typename T>
inline bool isEqual(const xMatrix<T>& A, const xMatrix<T>& B) {
	if (A.m_rows != B.m_rows || A.m_cols != B.m_cols)
		return false;

	for (int i = 0; i < A.m_rows; i++)
	{
		for (int j = 0; j < A.m_cols; j++)
		{
			T delta = abs(A.m_pData[i * A.m_cols + j] - B.m_pData[i * B.m_cols + j]);
			if (delta > 0.001/*FLT_EPSILON*/)
			{
				return false;
			}
		}
	}

	return true;
}

template<typename T>
inline void transpose(const xMatrix<T>& A, xMatrix<T>& B)
{
	if (A.m_rows != B.m_cols || A.m_cols != B.m_rows)
		return;

	for (int i = 0; i < B.m_rows; i++)
	{
		for (int j = 0; j < B.m_cols; j++)
		{
			B.m_pData[i * B.m_cols + j] = A.m_pData[j * A.m_cols + i];
		}
	}
}


template<typename T>
void addDot4kx1j(const T* rowA, const T* ptrB, T* rowC, int colsB, int k)
{
	T elemA0 = rowA[k];
	T elemA1 = rowA[k + 1];
	T elemA2 = rowA[k + 2];
	T elemA3 = rowA[k + 3];
	const T* rowB0 = ptrB + k * colsB;
	const T* rowB1 = ptrB + (k + 1) * colsB;
	const T* rowB2 = ptrB + (k + 2) * colsB;
	const T* rowB3 = ptrB + (k + 3) * colsB;

	float c_reg;
	for (int j = 0; j < colsB; j++)
	{
		c_reg = rowC[j];
		c_reg += elemA0 * rowB0[j];
		c_reg += elemA1 * rowB1[j];
		c_reg += elemA2 * rowB2[j];
		c_reg += elemA3 * rowB3[j];
		rowC[j] = c_reg;
	}
}

template<typename T>
void addDot4kx4j(const T* rowA, const T* ptrB, T* rowC, int colsB, int k)
{
	T elemA0 = rowA[k];
	T elemA1 = rowA[k + 1];
	T elemA2 = rowA[k + 2];
	T elemA3 = rowA[k + 3];
	const T* rowB0 = ptrB + k * colsB;
	const T* rowB1 = ptrB + (k + 1) * colsB;
	const T* rowB2 = ptrB + (k + 2) * colsB;
	const T* rowB3 = ptrB + (k + 3) * colsB;
	/*register*/ T c_reg;
	for (int j = 0; j < colsB; j += 4)
	{
		c_reg = rowC[j];
		c_reg += elemA0 * rowB0[j];
		c_reg += elemA1 * rowB1[j];
		c_reg += elemA2 * rowB2[j];
		c_reg += elemA3 * rowB3[j];
		rowC[j] = c_reg;

		c_reg = rowC[j + 1];
		c_reg += elemA0 * rowB0[j + 1];
		c_reg += elemA1 * rowB1[j + 1];
		c_reg += elemA2 * rowB2[j + 1];
		c_reg += elemA3 * rowB3[j + 1];
		rowC[j + 1] = c_reg;

		c_reg = rowC[j + 2];
		c_reg += elemA0 * rowB0[j + 2];
		c_reg += elemA1 * rowB1[j + 2];
		c_reg += elemA2 * rowB2[j + 2];
		c_reg += elemA3 * rowB3[j + 2];
		rowC[j + 2] = c_reg;

		c_reg = rowC[j + 3];
		c_reg += elemA0 * rowB0[j + 3];
		c_reg += elemA1 * rowB1[j + 3];
		c_reg += elemA2 * rowB2[j + 3];
		c_reg += elemA3 * rowB3[j + 3];
		rowC[j + 3] = c_reg;
	}
}

template<typename T>
void addDot4ix4k(const T* rowA, const T* ptrB, T* rowC, int colsA, int colsB, int k)
{
	const T* pelem = rowA + k;
	T elemA00 = pelem[0];
	T elemA01 = pelem[1];
	T elemA02 = pelem[2];
	T elemA03 = pelem[3];

	pelem += colsA;
	T elemA10 = pelem[0];
	T elemA11 = pelem[1];
	T elemA12 = pelem[2];
	T elemA13 = pelem[3];

	pelem += colsA;
	T elemA20 = pelem[0];
	T elemA21 = pelem[1];
	T elemA22 = pelem[2];
	T elemA23 = pelem[3];

	pelem += colsA;
	T elemA30 = pelem[0];
	T elemA31 = pelem[1];
	T elemA32 = pelem[2];
	T elemA33 = pelem[3];

	const T* rowB0 = ptrB + k * colsB;
	const T* rowB1 = ptrB + (k + 1) * colsB;
	const T* rowB2 = ptrB + (k + 2) * colsB;
	const T* rowB3 = ptrB + (k + 3) * colsB;

	T* rowC0 = rowC;
	T* rowC1 = rowC + colsB;
	T* rowC2 = rowC + 2 * colsB;
	T* rowC3 = rowC + 3 * colsB;

	T c_reg;
	for (int j = 0; j < colsB; j++)
	{
		T elemB0j = rowB0[j];
		T elemB1j = rowB1[j];
		T elemB2j = rowB2[j];
		T elemB3j = rowB3[j];

		c_reg = rowC0[j];
		c_reg += elemA00 * elemB0j;
		c_reg += elemA01 * elemB1j;
		c_reg += elemA02 * elemB2j;
		c_reg += elemA03 * elemB3j;
		rowC0[j] = c_reg;

		c_reg = rowC1[j];
		c_reg += elemA10 * elemB0j;
		c_reg += elemA11 * elemB1j;
		c_reg += elemA12 * elemB2j;
		c_reg += elemA13 * elemB3j;
		rowC1[j] = c_reg;

		c_reg = rowC2[j];
		c_reg += elemA20 * elemB0j;
		c_reg += elemA21 * elemB1j;
		c_reg += elemA22 * elemB2j;
		c_reg += elemA23 * elemB3j;
		rowC2[j] = c_reg;

		c_reg = rowC3[j];
		c_reg += elemA30 * elemB0j;
		c_reg += elemA31 * elemB1j;
		c_reg += elemA32 * elemB2j;
		c_reg += elemA33 * elemB3j;
		rowC3[j] = c_reg;
	}
}

template<typename T>
void addDot4ix4k(const T* rowA, const T* ptrB, T* rowC, int colsA, int colsB, int k, int sj, int block_j_size)
{
	const T* pelem = rowA + k;
	T elemA00 = pelem[0];
	T elemA01 = pelem[1];
	T elemA02 = pelem[2];
	T elemA03 = pelem[3];

	pelem += colsA;
	T elemA10 = pelem[0];
	T elemA11 = pelem[1];
	T elemA12 = pelem[2];
	T elemA13 = pelem[3];

	pelem += colsA;
	T elemA20 = pelem[0];
	T elemA21 = pelem[1];
	T elemA22 = pelem[2];
	T elemA23 = pelem[3];

	pelem += colsA;
	T elemA30 = pelem[0];
	T elemA31 = pelem[1];
	T elemA32 = pelem[2];
	T elemA33 = pelem[3];

	const T* rowB0 = ptrB + k * colsB;
	const T* rowB1 = ptrB + (k + 1) * colsB;
	const T* rowB2 = ptrB + (k + 2) * colsB;
	const T* rowB3 = ptrB + (k + 3) * colsB;

	T* rowC0 = rowC;
	T* rowC1 = rowC + colsB;
	T* rowC2 = rowC + 2 * colsB;
	T* rowC3 = rowC + 3 * colsB;

	T c_reg;
	int max_sj = std::min(sj + block_j_size, colsB);
	for (int j = sj; j < max_sj; j++)
	{
		T elemB0j = rowB0[j];
		T elemB1j = rowB1[j];
		T elemB2j = rowB2[j];
		T elemB3j = rowB3[j];

		c_reg = rowC0[j];
		c_reg += elemA00 * elemB0j;
		c_reg += elemA01 * elemB1j;
		c_reg += elemA02 * elemB2j;
		c_reg += elemA03 * elemB3j;
		rowC0[j] = c_reg;

		c_reg = rowC1[j];
		c_reg += elemA10 * elemB0j;
		c_reg += elemA11 * elemB1j;
		c_reg += elemA12 * elemB2j;
		c_reg += elemA13 * elemB3j;
		rowC1[j] = c_reg;

		c_reg = rowC2[j];
		c_reg += elemA20 * elemB0j;
		c_reg += elemA21 * elemB1j;
		c_reg += elemA22 * elemB2j;
		c_reg += elemA23 * elemB3j;
		rowC2[j] = c_reg;

		c_reg = rowC3[j];
		c_reg += elemA30 * elemB0j;
		c_reg += elemA31 * elemB1j;
		c_reg += elemA32 * elemB2j;
		c_reg += elemA33 * elemB3j;
		rowC3[j] = c_reg;
	}
}

template<typename T>
void addDot4ix4k_avx(const T* rowA, const T* ptrB, T* rowC, int colsA, int colsB, int k)
{
	assert(typeid(T) == typeid(float));

	const T* pelem = rowA + k;
	__m256 vA00 = _mm256_set1_ps(pelem[0]);
	__m256 vA01 = _mm256_set1_ps(pelem[1]);
	__m256 vA02 = _mm256_set1_ps(pelem[2]);
	__m256 vA03 = _mm256_set1_ps(pelem[3]);

	pelem += colsA;
	__m256 vA10 = _mm256_set1_ps(pelem[0]);
	__m256 vA11 = _mm256_set1_ps(pelem[1]);
	__m256 vA12 = _mm256_set1_ps(pelem[2]);
	__m256 vA13 = _mm256_set1_ps(pelem[3]);

	pelem += colsA;
	__m256 vA20 = _mm256_set1_ps(pelem[0]);
	__m256 vA21 = _mm256_set1_ps(pelem[1]);
	__m256 vA22 = _mm256_set1_ps(pelem[2]);
	__m256 vA23 = _mm256_set1_ps(pelem[3]);

	pelem += colsA;
	__m256 vA30 = _mm256_set1_ps(pelem[0]);
	__m256 vA31 = _mm256_set1_ps(pelem[1]);
	__m256 vA32 = _mm256_set1_ps(pelem[2]);
	__m256 vA33 = _mm256_set1_ps(pelem[3]);


	const T* rowB0 = ptrB + k * colsB;
	const T* rowB1 = ptrB + (k + 1) * colsB;
	const T* rowB2 = ptrB + (k + 2) * colsB;
	const T* rowB3 = ptrB + (k + 3) * colsB;

	T* rowC0 = rowC;
	T* rowC1 = rowC + colsB;
	T* rowC2 = rowC + 2 * colsB;
	T* rowC3 = rowC + 3 * colsB;

	__m256 vC;
	int jtail = colsB % 8;
	for (int j = 0; j < colsB - jtail; j += 8)
	{
		__m256 vB0 = _mm256_load_ps(rowB0 + j);
		__m256 vB1 = _mm256_load_ps(rowB1 + j);
		__m256 vB2 = _mm256_load_ps(rowB2 + j);
		__m256 vB3 = _mm256_load_ps(rowB3 + j);

		// 0>
		vC = _mm256_load_ps(rowC0 + j);
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA00, vB0));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA01, vB1));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA02, vB2));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA03, vB3));
		_mm256_store_ps(rowC0 + j, vC);

		// 1>
		vC = _mm256_load_ps(rowC1 + j);
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA10, vB0));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA11, vB1));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA12, vB2));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA13, vB3));
		_mm256_store_ps(rowC1 + j, vC);

		// 2>
		vC = _mm256_load_ps(rowC2 + j);
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA20, vB0));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA21, vB1));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA22, vB2));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA23, vB3));
		_mm256_store_ps(rowC2 + j, vC);

		// 3>
		vC = _mm256_load_ps(rowC3 + j);
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA30, vB0));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA31, vB1));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA32, vB2));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA33, vB3));
		_mm256_store_ps(rowC3 + j, vC);
	}
}

template<typename T>
void addDot4ix4k_avx(const T* rowA, const T* ptrB, T* rowC, int colsA, int colsB, int k, int sj, int block_size)
{
	assert(typeid(T) == typeid(float));

	const T* pelem = rowA + k;
	__m256 vA00 = _mm256_set1_ps(pelem[0]);
	__m256 vA01 = _mm256_set1_ps(pelem[1]);
	__m256 vA02 = _mm256_set1_ps(pelem[2]);
	__m256 vA03 = _mm256_set1_ps(pelem[3]);

	pelem += colsA;
	__m256 vA10 = _mm256_set1_ps(pelem[0]);
	__m256 vA11 = _mm256_set1_ps(pelem[1]);
	__m256 vA12 = _mm256_set1_ps(pelem[2]);
	__m256 vA13 = _mm256_set1_ps(pelem[3]);

	pelem += colsA;
	__m256 vA20 = _mm256_set1_ps(pelem[0]);
	__m256 vA21 = _mm256_set1_ps(pelem[1]);
	__m256 vA22 = _mm256_set1_ps(pelem[2]);
	__m256 vA23 = _mm256_set1_ps(pelem[3]);

	pelem += colsA;
	__m256 vA30 = _mm256_set1_ps(pelem[0]);
	__m256 vA31 = _mm256_set1_ps(pelem[1]);
	__m256 vA32 = _mm256_set1_ps(pelem[2]);
	__m256 vA33 = _mm256_set1_ps(pelem[3]);


	const T* rowB0 = ptrB + k * colsB;
	const T* rowB1 = ptrB + (k + 1) * colsB;
	const T* rowB2 = ptrB + (k + 2) * colsB;
	const T* rowB3 = ptrB + (k + 3) * colsB;

	T* rowC0 = rowC;
	T* rowC1 = rowC + colsB;
	T* rowC2 = rowC + 2 * colsB;
	T* rowC3 = rowC + 3 * colsB;

	__m256 vC;
	for (int j = sj; j < sj + block_size && j < colsB; j += 8)
	{
		__m256 vB0 = _mm256_load_ps(rowB0 + j);
		__m256 vB1 = _mm256_load_ps(rowB1 + j);
		__m256 vB2 = _mm256_load_ps(rowB2 + j);
		__m256 vB3 = _mm256_load_ps(rowB3 + j);

		// 0>
		vC = _mm256_load_ps(rowC0 + j);
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA00, vB0));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA01, vB1));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA02, vB2));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA03, vB3));
		_mm256_store_ps(rowC0 + j, vC);

		// 1>
		vC = _mm256_load_ps(rowC1 + j);
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA10, vB0));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA11, vB1));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA12, vB2));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA13, vB3));
		_mm256_store_ps(rowC1 + j, vC);

		// 2>
		vC = _mm256_load_ps(rowC2 + j);
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA20, vB0));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA21, vB1));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA22, vB2));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA23, vB3));
		_mm256_store_ps(rowC2 + j, vC);

		// 3>
		vC = _mm256_load_ps(rowC3 + j);
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA30, vB0));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA31, vB1));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA32, vB2));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA33, vB3));
		_mm256_store_ps(rowC3 + j, vC);
	}
}

template<typename T>
void addDot4ix4k_avx_packed(const T* rowA, const T* ptrB, T* rowC,
	int colsA, int colsB, int oldColsB,
	int k, int sj, int block_size)
{
	assert(typeid(T) == typeid(float));

	const T* pelem = rowA + k;
	__m256 vA00 = _mm256_set1_ps(pelem[0]);
	__m256 vA01 = _mm256_set1_ps(pelem[1]);
	__m256 vA02 = _mm256_set1_ps(pelem[2]);
	__m256 vA03 = _mm256_set1_ps(pelem[3]);

	pelem += colsA;
	__m256 vA10 = _mm256_set1_ps(pelem[0]);
	__m256 vA11 = _mm256_set1_ps(pelem[1]);
	__m256 vA12 = _mm256_set1_ps(pelem[2]);
	__m256 vA13 = _mm256_set1_ps(pelem[3]);

	pelem += colsA;
	__m256 vA20 = _mm256_set1_ps(pelem[0]);
	__m256 vA21 = _mm256_set1_ps(pelem[1]);
	__m256 vA22 = _mm256_set1_ps(pelem[2]);
	__m256 vA23 = _mm256_set1_ps(pelem[3]);

	pelem += colsA;
	__m256 vA30 = _mm256_set1_ps(pelem[0]);
	__m256 vA31 = _mm256_set1_ps(pelem[1]);
	__m256 vA32 = _mm256_set1_ps(pelem[2]);
	__m256 vA33 = _mm256_set1_ps(pelem[3]);


	const T* rowB0 = ptrB + k * colsB;
	const T* rowB1 = ptrB + (k + 1) * colsB;
	const T* rowB2 = ptrB + (k + 2) * colsB;
	const T* rowB3 = ptrB + (k + 3) * colsB;

	T* rowC0 = rowC;
	T* rowC1 = rowC + oldColsB;
	T* rowC2 = rowC + 2 * oldColsB;
	T* rowC3 = rowC + 3 * oldColsB;

	__m256 vC;
	for (int j = sj; j < sj + block_size && j < oldColsB; j += 8)
	{
		__m256 vB0 = _mm256_load_ps(rowB0 + (j - sj));
		__m256 vB1 = _mm256_load_ps(rowB1 + (j - sj));
		__m256 vB2 = _mm256_load_ps(rowB2 + (j - sj));
		__m256 vB3 = _mm256_load_ps(rowB3 + (j - sj));

		// 0>
		vC = _mm256_load_ps(rowC0 + j);
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA00, vB0));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA01, vB1));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA02, vB2));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA03, vB3));
		_mm256_store_ps(rowC0 + j, vC);

		// 1>
		vC = _mm256_load_ps(rowC1 + j);
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA10, vB0));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA11, vB1));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA12, vB2));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA13, vB3));
		_mm256_store_ps(rowC1 + j, vC);

		// 2>
		vC = _mm256_load_ps(rowC2 + j);
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA20, vB0));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA21, vB1));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA22, vB2));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA23, vB3));
		_mm256_store_ps(rowC2 + j, vC);

		// 3>
		vC = _mm256_load_ps(rowC3 + j);
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA30, vB0));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA31, vB1));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA32, vB2));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA33, vB3));
		_mm256_store_ps(rowC3 + j, vC);
	}
}

template<typename T>
void addDot4kx1j_avx(const T* rowA, const T* ptrB, T* rowC, int colsB, int k)
{
	assert(typeid(T) == typeid(float));

	__m256 vA = _mm256_set1_ps(rowA[k]);
	__m256 vA1 = _mm256_set1_ps(rowA[k + 1]);
	__m256 vA2 = _mm256_set1_ps(rowA[k + 2]);
	__m256 vA3 = _mm256_set1_ps(rowA[k + 3]);
	const T* rowB0 = ptrB + k * colsB;
	const T* rowB1 = ptrB + (k + 1) * colsB;
	const T* rowB2 = ptrB + (k + 2) * colsB;
	const T* rowB3 = ptrB + (k + 3) * colsB;
	int jtail = colsB % 8;
	for (int j = 0; j < colsB - jtail; j += 8)
	{
		__m256 vB = _mm256_load_ps(rowB0 + j);
		__m256 vB1 = _mm256_load_ps(rowB1 + j);
		__m256 vB2 = _mm256_load_ps(rowB2 + j);
		__m256 vB3 = _mm256_load_ps(rowB3 + j);
		__m256 vC = _mm256_load_ps(rowC + j);

		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA, vB));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA1, vB1));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA2, vB2));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA3, vB3));

		_mm256_store_ps(rowC + j, vC);
	}
}

template<typename T>
void addDot1kx4j(float elemA, const float* ptrB, float* ptrC, int j)
{
	// C(i, j) += A(i, k)* B(k, j);
	ptrC[j] += elemA * ptrB[j];
	ptrC[j + 1] += elemA * ptrB[j + 1];
	ptrC[j + 2] += elemA * ptrB[j + 2];
	ptrC[j + 3] += elemA * ptrB[j + 3];
}


template<typename T>
void subBlock4(int oldARow, int oldACol, int oldBCol, float* ptrA, float* ptrB, float* ptrC, const int& ARow, const int& ACol, const int& BCol)
{
	const int nARowBlock = 4;
	const int nAColBlock = 4;
	const int nBColBlock = 8;

	int packARow = ARow;
	int packACol = ACol;
	int packBCol = BCol;
	if (packARow % nARowBlock)
	{
		packARow += nARowBlock - packARow % nARowBlock;
	}

	if (packACol % nAColBlock)
	{
		packACol += nAColBlock - packACol % nAColBlock;
	}

	if (packBCol % nBColBlock)
	{
		packBCol += nBColBlock - packBCol % nBColBlock;
	}

	T* packA = new T[packARow * packACol]();	//默认初始化为0
	T* packB = new T[packACol * packBCol]();

	for (int i = 0; i < ARow; i += nARowBlock)
	{
		int nrealAow = i + nARowBlock > ARow ? ARow - i : nARowBlock;
		for (int m = i; m < i + nARowBlock && m < ARow; m++)
		{
			memcpy(packA + m * ACol, ptrA + m * oldACol, ACol * sizeof(T));
		}

		T* pATemp_LI = packA + i * ACol;
		T* pCTemp_LI = ptrC + i * oldBCol;

		for (int k = 0; k < ACol; k += nAColBlock)
		{
			if (0 == i)
			{
				for (int m = k; m < k + nAColBlock && m < ACol; m++)
				{
					memcpy(packB + m * BCol, ptrB + m * oldBCol, BCol * sizeof(T));
				}
			}

			//subBlock4addDot(oldARow, oldACol, oldBCol, pATemp_LI + k, packB + k * BCol, pCTemp_LI, ARow, ACol, BCol, nBColBlock);
			subBlock4addDot(oldARow, oldACol, oldBCol, pATemp_LI + k, packB + k * BCol, pCTemp_LI, nrealAow, ACol, BCol, nBColBlock);
		}
	}

	delete[] packA;
	delete[] packB;
}

template<typename T>
void subBlock4addDot(int oldARow, int oldACol, int oldBCol, float* A, float* B, float* C,
	const int& ARow, const int& ACol, const int& BCol, const int& BColBlock)
{
	assert(typeid(T) == typeid(float));

	const T* pelem = A;
	__m256 vA00 = _mm256_set1_ps(pelem[0]);
	__m256 vA01 = _mm256_set1_ps(pelem[1]);
	__m256 vA02 = _mm256_set1_ps(pelem[2]);
	__m256 vA03 = _mm256_set1_ps(pelem[3]);

	pelem += ACol;
	__m256 vA10 = _mm256_set1_ps(pelem[0]);
	__m256 vA11 = _mm256_set1_ps(pelem[1]);
	__m256 vA12 = _mm256_set1_ps(pelem[2]);
	__m256 vA13 = _mm256_set1_ps(pelem[3]);

	pelem += ACol;
	__m256 vA20 = _mm256_set1_ps(pelem[0]);
	__m256 vA21 = _mm256_set1_ps(pelem[1]);
	__m256 vA22 = _mm256_set1_ps(pelem[2]);
	__m256 vA23 = _mm256_set1_ps(pelem[3]);

	pelem += ACol;
	__m256 vA30 = _mm256_set1_ps(pelem[0]);
	__m256 vA31 = _mm256_set1_ps(pelem[1]);
	__m256 vA32 = _mm256_set1_ps(pelem[2]);
	__m256 vA33 = _mm256_set1_ps(pelem[3]);

	const T* rowB0 = B;
	const T* rowB1 = B + BCol;
	const T* rowB2 = B + BCol * 2;
	const T* rowB3 = B + BCol * 3;

	T* rowC0 = C;
	T* rowC1 = C + oldBCol;
	T* rowC2 = C + 2 * oldBCol;
	T* rowC3 = C + 3 * oldBCol;

	__m256 vC;
	int j = 0;
	for (; j + BColBlock <= BCol; j += BColBlock)
	{
		__m256 vB0 = _mm256_load_ps(rowB0 + j);
		__m256 vB1 = _mm256_load_ps(rowB1 + j);
		__m256 vB2 = _mm256_load_ps(rowB2 + j);
		__m256 vB3 = _mm256_load_ps(rowB3 + j);

		// 0>
		vC = _mm256_load_ps(rowC0 + j);
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA00, vB0));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA01, vB1));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA02, vB2));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA03, vB3));
		_mm256_store_ps(rowC0 + j, vC);

		if (ARow == 4)
		{
			// 1>
			vC = _mm256_load_ps(rowC1 + j);
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA10, vB0));
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA11, vB1));
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA12, vB2));
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA13, vB3));
			_mm256_store_ps(rowC1 + j, vC);

			// 2>
			vC = _mm256_load_ps(rowC2 + j);
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA20, vB0));
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA21, vB1));
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA22, vB2));
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA23, vB3));
			_mm256_store_ps(rowC2 + j, vC);

			// 3>
			vC = _mm256_load_ps(rowC3 + j);
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA30, vB0));
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA31, vB1));
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA32, vB2));
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA33, vB3));
			_mm256_store_ps(rowC3 + j, vC);

		}
		else if (ARow == 3)
		{
			// 1>
			vC = _mm256_load_ps(rowC1 + j);
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA10, vB0));
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA11, vB1));
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA12, vB2));
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA13, vB3));
			_mm256_store_ps(rowC1 + j, vC);

			// 2>
			vC = _mm256_load_ps(rowC2 + j);
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA20, vB0));
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA21, vB1));
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA22, vB2));
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA23, vB3));
			_mm256_store_ps(rowC2 + j, vC);

		}
		else if (ARow == 2)
		{
			// 1>
			vC = _mm256_load_ps(rowC1 + j);
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA10, vB0));
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA11, vB1));
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA12, vB2));
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA13, vB3));
			_mm256_store_ps(rowC1 + j, vC);

		}

	}

	if (j < BCol)
	{
		T* Ctempbuf0 = new T[BColBlock]();
		T* Ctempbuf1 = new T[BColBlock]();
		T* Ctempbuf2 = new T[BColBlock]();
		T* Ctempbuf3 = new T[BColBlock]();

		__m256 vB0 = _mm256_load_ps(rowB0 + j);
		__m256 vB1 = _mm256_load_ps(rowB1 + j);
		__m256 vB2 = _mm256_load_ps(rowB2 + j);
		__m256 vB3 = _mm256_load_ps(rowB3 + j);


		// 0>
		memcpy(Ctempbuf0, rowC0 + j, (BCol - j) * sizeof(float));
		vC = _mm256_load_ps(Ctempbuf0);
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA00, vB0));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA01, vB1));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA02, vB2));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA03, vB3));
		_mm256_store_ps(Ctempbuf0, vC);
		memcpy(rowC0 + j, Ctempbuf0, (BCol - j) * sizeof(float));


		if (ARow == 4)
		{
			// 1>
			memcpy(Ctempbuf1, rowC1 + j, (BCol - j) * sizeof(T));
			vC = _mm256_load_ps(Ctempbuf1);
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA10, vB0));
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA11, vB1));
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA12, vB2));
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA13, vB3));
			_mm256_store_ps(Ctempbuf1, vC);
			memcpy(rowC1 + j, Ctempbuf1, (BCol - j) * sizeof(T));

			// 2>
			memcpy(Ctempbuf2, rowC2 + j, (BCol - j) * sizeof(T));
			vC = _mm256_load_ps(Ctempbuf2);
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA20, vB0));
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA21, vB1));
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA22, vB2));
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA23, vB3));
			_mm256_store_ps(Ctempbuf2, vC);
			memcpy(rowC2 + j, Ctempbuf2, (BCol - j) * sizeof(T));

			// 3>
			memcpy(Ctempbuf3, rowC3 + j, (BCol - j) * sizeof(T));
			vC = _mm256_load_ps(Ctempbuf3);
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA30, vB0));
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA31, vB1));
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA32, vB2));
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA33, vB3));
			_mm256_store_ps(Ctempbuf3, vC);
			memcpy(rowC3 + j, Ctempbuf3, (BCol - j) * sizeof(T));
		}
		else if (ARow == 3)
		{
			// 1>
			memcpy(Ctempbuf1, rowC1 + j, (BCol - j) * sizeof(T));
			vC = _mm256_load_ps(Ctempbuf1);
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA10, vB0));
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA11, vB1));
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA12, vB2));
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA13, vB3));
			_mm256_store_ps(Ctempbuf1, vC);
			memcpy(rowC1 + j, Ctempbuf1, (BCol - j) * sizeof(T));

			// 2>
			memcpy(Ctempbuf2, rowC2 + j, (BCol - j) * sizeof(T));
			vC = _mm256_load_ps(Ctempbuf2);
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA20, vB0));
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA21, vB1));
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA22, vB2));
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA23, vB3));
			_mm256_store_ps(Ctempbuf2, vC);
			memcpy(rowC2 + j, Ctempbuf2, (BCol - j) * sizeof(T));
		}
		else if (ARow == 2)
		{
			// 1>
			memcpy(Ctempbuf1, rowC1 + j, (BCol - j) * sizeof(T));
			vC = _mm256_load_ps(Ctempbuf1);
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA10, vB0));
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA11, vB1));
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA12, vB2));
			vC = _mm256_add_ps(vC, _mm256_mul_ps(vA13, vB3));
			_mm256_store_ps(Ctempbuf1, vC);
			memcpy(rowC1 + j, Ctempbuf1, (BCol - j) * sizeof(T));
		}

		delete[] Ctempbuf0;
		delete[] Ctempbuf1;
		delete[] Ctempbuf2;
		delete[] Ctempbuf3;
	}
}

template<typename T>
void subBlock3(int oldARow, int oldACol, int oldBCol, T* ptrA, T* ptrB, T* ptrC, const int& ARow, const int& ACol, const int& BCol)
{
	T* packA = new T[ARow * ACol];
	T* packB = new T[ACol * BCol];

	for (int i = 0; i < ARow; i += 4)
	{
		memcpy(packA + i * ACol, ptrA + i * oldACol, ACol * sizeof(T));
		memcpy(packA + (i + 1) * ACol, ptrA + (i + 1) * oldACol, ACol * sizeof(T));
		memcpy(packA + (i + 2) * ACol, ptrA + (i + 2) * oldACol, ACol * sizeof(T));
		memcpy(packA + (i + 3) * ACol, ptrA + (i + 3) * oldACol, ACol * sizeof(T));

		T* pATemp_LI = packA + i * ACol;
		T* pCTemp_LI = ptrC + i * oldBCol;

		for (int k = 0; k < ACol; k += 4)
		{
			if (0 == i)
			{
				memcpy(packB + k * BCol, ptrB + k * oldBCol, BCol * sizeof(T));
				memcpy(packB + (k + 1) * BCol, ptrB + (k + 1) * oldBCol, BCol * sizeof(T));
				memcpy(packB + (k + 2) * BCol, ptrB + (k + 2) * oldBCol, BCol * sizeof(T));
				memcpy(packB + (k + 3) * BCol, ptrB + (k + 3) * oldBCol, BCol * sizeof(T));
			}

			subBlock3addDot(oldARow, oldACol, oldBCol, pATemp_LI + k, packB + k * BCol, pCTemp_LI, ARow, ACol, BCol);
		}
	}

	delete[] packA;
	delete[] packB;
}

template<typename T>
void subBlock3addDot(int oldARow, int oldACol, int oldBCol, T* A, T* B, T* C, const int& ARow, const int& ACol, const int& BCol)
{
	assert(typeid(T) == typeid(float));

	const T* pelem = A;
	__m256 vA00 = _mm256_set1_ps(pelem[0]);
	__m256 vA01 = _mm256_set1_ps(pelem[1]);
	__m256 vA02 = _mm256_set1_ps(pelem[2]);
	__m256 vA03 = _mm256_set1_ps(pelem[3]);

	pelem += ACol;
	__m256 vA10 = _mm256_set1_ps(pelem[0]);
	__m256 vA11 = _mm256_set1_ps(pelem[1]);
	__m256 vA12 = _mm256_set1_ps(pelem[2]);
	__m256 vA13 = _mm256_set1_ps(pelem[3]);

	pelem += ACol;
	__m256 vA20 = _mm256_set1_ps(pelem[0]);
	__m256 vA21 = _mm256_set1_ps(pelem[1]);
	__m256 vA22 = _mm256_set1_ps(pelem[2]);
	__m256 vA23 = _mm256_set1_ps(pelem[3]);

	pelem += ACol;
	__m256 vA30 = _mm256_set1_ps(pelem[0]);
	__m256 vA31 = _mm256_set1_ps(pelem[1]);
	__m256 vA32 = _mm256_set1_ps(pelem[2]);
	__m256 vA33 = _mm256_set1_ps(pelem[3]);

	const T* rowB0 = B;
	const T* rowB1 = B + BCol;
	const T* rowB2 = B + BCol * 2;
	const T* rowB3 = B + BCol * 3;

	T* rowC0 = C;
	T* rowC1 = C + oldBCol;
	T* rowC2 = C + 2 * oldBCol;
	T* rowC3 = C + 3 * oldBCol;

	__m256 vC;
	for (int j = 0; j < BCol; j += 8)
	{
		__m256 vB0 = _mm256_load_ps(rowB0 + j);
		__m256 vB1 = _mm256_load_ps(rowB1 + j);
		__m256 vB2 = _mm256_load_ps(rowB2 + j);
		__m256 vB3 = _mm256_load_ps(rowB3 + j);

		// 0>
		vC = _mm256_load_ps(rowC0 + j);
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA00, vB0));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA01, vB1));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA02, vB2));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA03, vB3));
		_mm256_store_ps(rowC0 + j, vC);

		// 1>
		vC = _mm256_load_ps(rowC1 + j);
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA10, vB0));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA11, vB1));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA12, vB2));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA13, vB3));
		_mm256_store_ps(rowC1 + j, vC);

		// 2>
		vC = _mm256_load_ps(rowC2 + j);
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA20, vB0));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA21, vB1));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA22, vB2));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA23, vB3));
		_mm256_store_ps(rowC2 + j, vC);

		// 3>
		vC = _mm256_load_ps(rowC3 + j);
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA30, vB0));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA31, vB1));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA32, vB2));
		vC = _mm256_add_ps(vC, _mm256_mul_ps(vA33, vB3));
		_mm256_store_ps(rowC3 + j, vC);
	}
}

template<typename T>
void subBlock(const xMatrix<T>& A, const xMatrix<T>& B, xMatrix<T>& C,
	int si, int sj, int sk, const int block_size)
{
	const T* ptrA = A.m_pData;
	const T* ptrB = B.m_pData;
	T* ptrC = C.m_pData;
	// 	for (int i = si; i < si+block_size && i < A.m_row; i++)
	// 	{
	// 		for (int j = sj; j < sj+block_size && j < B.m_col; j++)
	// 		{
	// 			for (int k = sk; k < sk+block_size && k < A.m_col; k++)
	// 			{
	// 				// C(i, j) += A(i, k)* B(k, j);
	// 				ptrC[i*C.m_col + j] += ptrA[i*A.m_col + k] * ptrB[k*B.m_col + j];
	// 			}
	// 		}
	// 	}


	for (int i = si; i < si + block_size && i < A.m_rows; i += 4)
	{
		T* rowC = ptrC + i * C.m_cols;
		const T* rowA = ptrA + i * A.m_cols;
		for (int k = sk; k < sk + block_size && k < A.m_cols; k += 4)
		{
			addDot4ix4k_avx(rowA, ptrB, rowC, A.m_cols, B.m_cols, k, sj, block_size);
		}
	}

}

template<typename T>
void subBlock2(const xMatrix<T>& A, const xMatrix<T>& B, xMatrix<T>& C,
	int si, int sj, int sk,
	const int block_i_size, const int block_j_size, const int block_k_size)
{
	const T* ptrA = A.m_pData;
	const T* ptrB = B.m_pData;
	T* ptrC = C.m_pData;

	T* packedA = new T[block_i_size * block_k_size];
	T* packedB = new T[block_k_size * block_j_size];

	for (int i = si; i < si + block_i_size && i < A.m_rows; i += 4)
	{
		memcpy(packedA + (i - si) * block_k_size, ptrA + i * A.m_cols + sk, block_k_size * sizeof(T));
		memcpy(packedA + (i - si + 1) * block_k_size, ptrA + (i + 1) * A.m_cols + sk, block_k_size * sizeof(T));
		memcpy(packedA + (i - si + 2) * block_k_size, ptrA + (i + 2) * A.m_cols + sk, block_k_size * sizeof(T));
		memcpy(packedA + (i - si + 3) * block_k_size, ptrA + (i + 3) * A.m_cols + sk, block_k_size * sizeof(T));

		T* rowC = ptrC + i * C.m_cols;
		const T* rowA = packedA + (i - si) * block_k_size;

		for (int k = sk; k < sk + block_k_size && k < A.m_cols; k += 4)
		{
			if (i == si)
			{
				memcpy(packedB + (k - sk) * block_j_size, ptrB + k * B.m_cols + sj, block_j_size * sizeof(T));
				memcpy(packedB + (k - sk + 1) * block_j_size, ptrB + (k + 1) * B.m_cols + sj, block_j_size * sizeof(T));
				memcpy(packedB + (k - sk + 2) * block_j_size, ptrB + (k + 2) * B.m_cols + sj, block_j_size * sizeof(T));
				memcpy(packedB + (k - sk + 3) * block_j_size, ptrB + (k + 3) * B.m_cols + sj, block_j_size * sizeof(T));
			}

			addDot4ix4k_avx_packed(rowA, packedB, rowC, block_k_size, block_j_size, B.m_cols, k - sk, sj, block_j_size);
		}
	}

	delete[] packedA;
	delete[] packedB;
}


template<typename T>
void multi1(const xMatrix<T>& A, const xMatrix<T>& B, xMatrix<T>& C)
{
	const T* ptrA = A.m_pData;
	const T* ptrB = B.m_pData;
	T* ptrC = C.m_pData;
	for (int i = 0; i < A.m_rows; i++)
	{
		for (int j = 0; j < B.m_cols; j++)
		{
			for (int k = 0; k < A.m_cols; k++)
			{
				// C(i, j) += A(i, k)* B(k, j);
				ptrC[i * C.m_cols + j] += ptrA[i * A.m_cols + k] * ptrB[k * B.m_cols + j];
			}
		}
	}
}

template<typename T>
void multi2(const xMatrix<T>& A, const xMatrix<T>& B, xMatrix<T>& C)
{
	const T* ptrA = A.m_pData;
	const T* ptrB = B.m_pData;
	T* ptrC = C.m_pData;
	for (int i = 0; i < A.m_rows; i++)
	{
		for (int k = 0; k < A.m_cols; k++)
		{
			float elemA = ptrA[i * A.m_cols + k];
			for (int j = 0; j < B.m_cols; j += 4)		// k*B.m_col和i*C.m_col的计算重复了B.m_col次
			{
				// C(i, j) += A(i, k)* B(k, j);
				ptrC[i * C.m_cols + j] += elemA * ptrB[k * B.m_cols + j];
				ptrC[i * C.m_cols + j + 1] += elemA * ptrB[k * B.m_cols + j + 1];
				ptrC[i * C.m_cols + j + 2] += elemA * ptrB[k * B.m_cols + j + 2];
				ptrC[i * C.m_cols + j + 3] += elemA * ptrB[k * B.m_cols + j + 3];
			}
		}
	}
}

template<typename T>
void subBlockX(const xMatrix<T>& A, const xMatrix<T>& B, xMatrix<T>& C,
	int si, int sj, int sk,
	const int block_i_size, const int block_j_size, const int block_k_size)
{
	const T* ptrA = A.m_pData;
	const T* ptrB = B.m_pData;
	T* ptrC = C.m_pData;
	int max_si = std::min(si + block_i_size, A.m_rows);
	int max_sj = std::min(sj + block_j_size, B.m_cols);
	int max_sk = std::min(sk + block_k_size, A.m_cols);
	for (int i = si; i < max_si; i++)
	{
		T* rowC = ptrC + i * C.m_cols;
		const T* rowA = ptrA + i * A.m_cols;
		for (int k = sk; k < max_sk; k++)
		{
			T elemA = rowA[k];
			for (int j = sj; j < max_sj; j++)
			{
				// C(i, j) += A(i, k)* B(k, j);
				rowC[j] += elemA * ptrB[k * B.m_cols + j];
			}
		}
	}
}

template<typename T>
void subBlockI(const xMatrix<T>& A, const xMatrix<T>& B, xMatrix<T>& C,
	int si, int sj, int sk,
	const int block_i_size, const int block_j_size, const int block_k_size)
{
	const T* ptrA = A.m_pData;
	const T* ptrB = B.m_pData;
	T* ptrC = C.m_pData;
	int max_si = std::min(si + block_i_size, A.m_rows);
	int max_sj = std::min(sj + block_j_size, B.m_cols);
	int max_sk = std::min(sk + block_k_size, A.m_cols);
	int jtail = max_sj % 8;
	int ktail = max_sk % 4;
	for (int i = si; i < max_si; i++)
	{
		T* rowC = ptrC + i * C.m_cols;
		const T* rowA = ptrA + i * A.m_cols;
		for (int k = sk; k < max_sk - ktail; k += 4)
		{
			addDot4kx1j_avx(rowA, ptrB, rowC, B.m_cols, k);
		}
	}


	// 执行顺序必须是j、k、i，对应上面的i、j、k。否则会因为浮点数加法的执行顺序不一致导致结果不一样
	if (jtail != 0)
	{
		subBlockX(A, B, C,
			si, max_sj - jtail, sk,
			max_si, jtail, max_sk - ktail);
	}
	if (ktail != 0)
	{
		subBlockX(A, B, C,
			si, sj, max_sk - ktail,
			max_si, max_sj, ktail);
	}
}

template<typename T>
void subBlockJ(const xMatrix<T>& A, const xMatrix<T>& B, xMatrix<T>& C,
	int si, int sj, int sk,
	const int block_i_size, const int block_j_size, const int block_k_size)
{
	const T* ptrA = A.m_pData;
	const T* ptrB = B.m_pData;
	T* ptrC = C.m_pData;
	int max_si = std::min(si + block_i_size, A.m_rows);
	int max_sj = std::min(sj + block_j_size, B.m_cols);
	int max_sk = std::min(sk + block_k_size, A.m_cols);
	int itail = max_si % 4;
	int ktail = max_sk % 4;
	for (int i = si; i < max_si - itail; i += 4)
	{
		T* rowC = ptrC + i * C.m_cols;
		const T* rowA = ptrA + i * A.m_cols;
		for (int k = sk; k < max_sk - ktail; k += 4)
		{
			addDot4ix4k(rowA, ptrB, rowC, A.m_cols, B.m_cols, k, sj, block_j_size);
		}
	}

	// 执行顺序必须是j、k、i，对应上面的i、j、k。否则会因为浮点数加法的执行顺序不一致导致结果不一样	
	if (ktail != 0)
	{
		subBlockX(A, B, C,
			si, sj, max_sk - ktail,
			max_si - itail, max_sj, ktail);
	}
	if (itail != 0)
	{
		subBlockX(A, B, C,
			max_si - itail, sj, sk,
			itail, max_sj, max_sk);
	}
}

template<typename T>
void subBlockK(const xMatrix<T>& A, const xMatrix<T>& B, xMatrix<T>& C,
	int si, int sj, int sk,
	const int block_i_size, const int block_j_size, const int block_k_size)
{
	const T* ptrA = A.m_pData;
	const T* ptrB = B.m_pData;
	T* ptrC = C.m_pData;
	int max_si = std::min(si + block_i_size, A.m_rows);
	int max_sj = std::min(sj + block_j_size, B.m_cols);
	int max_sk = std::min(sk + block_k_size, A.m_cols);
	for (int i = si; i < max_si; i++)
	{
		const T* rowA = ptrA + i * A.m_cols;
		T* rowC = ptrC + i * C.m_cols;
		for (int j = sj; j < max_sj; j++)
		{
			T sum = rowC[j];
			for (int k = sk; k < max_sk; k++)
			{
				// C(i, j) += A(i, k)* B(k, j);
				sum += rowA[k] * ptrB[k * B.m_cols + j];
			}
			ptrC[i * C.m_cols + j] = sum;
		}
	}
}

template<typename T>
void multi3(const xMatrix<T>& A, const xMatrix<T>& B, xMatrix<T>& C)
{
	const T* ptrA = A.m_pData;
	const T* ptrB = B.m_pData;
	T* ptrC = C.m_pData;
	for (int i = 0; i < A.m_rows; i++)
	{
		T* rowC = ptrC + i * C.m_cols;			// 每行C重复A.m_col次访问
		const T* rowA = ptrA + i * A.m_cols;
		for (int k = 0; k < A.m_cols; k++)
		{
			T elemA = rowA[k];
			const T* rowB = ptrB + k * B.m_cols;
			for (int j = 0; j < B.m_cols; j += 4)
			{
				addDot1kx4j(elemA, rowB, rowC, j);
			}
		}
	}
}

template<typename T>
void multi4(const xMatrix<T>& A, const xMatrix<T>& B, xMatrix<T>& C)
{
	const T* ptrA = A.m_pData;
	const T* ptrB = B.m_pData;
	T* ptrC = C.m_pData;
	for (int i = 0; i < A.m_rows; i++)
	{
		T* rowC = ptrC + i * C.m_cols;			// 每行C重复A.m_col次访问
		const T* rowA = ptrA + i * A.m_cols;
		for (int k = 0; k < A.m_cols; k += 4)
		{
			addDot4kx1j(rowA, ptrB, rowC, B.m_cols, k);
		}
	}
}

template<typename T>
void multi5(const xMatrix<T>& A, const xMatrix<T>& B, xMatrix<T>& C)
{
	const T* ptrA = A.m_pData;
	const T* ptrB = B.m_pData;
	T* ptrC = C.m_pData;
	for (int i = 0; i < A.m_rows; i++)
	{
		T* rowC = ptrC + i * C.m_cols;			// 每行C重复A.m_col次访问
		const T* rowA = ptrA + i * A.m_cols;
		for (int k = 0; k < A.m_cols; k += 4)
		{
			addDot4kx4j(rowA, ptrB, rowC, B.m_cols, k);
		}
	}
}

template<typename T>
void multi6(const xMatrix<T>& A, const xMatrix<T>& B, xMatrix<T>& C)
{
	const float* ptrA = A.m_pData;
	const float* ptrB = B.m_pData;
	float* ptrC = C.m_pData;
	for (int i = 0; i < A.m_rows; i++)
	{
		float* rowC = ptrC + i * C.m_cols;			// 每行C重复A.m_col次访问
		const float* rowA = ptrA + i * A.m_cols;
		for (int k = 0; k < A.m_cols; k++)
		{
			__declspec(align(32)) __m256 vA = _mm256_set1_ps(rowA[k]);
			const float* rowB = ptrB + k * B.m_cols;
			for (int j = 0; j < B.m_cols; j += 8)
			{
				__declspec(align(32)) __m256 vB = _mm256_load_ps(rowB + j);
				__declspec(align(32)) __m256 vC = _mm256_load_ps(rowC + j);
				vC = _mm256_add_ps(vC, _mm256_mul_ps(vA, vB));
				_mm256_store_ps(rowC + j, vC);
			}
		}
	}
}

template<typename T>
void multi7(const xMatrix<T>& A, const xMatrix<T>& B, xMatrix<T>& C)
{
	const T* ptrA = A.m_pData;
	const T* ptrB = B.m_pData;
	float* ptrC = C.m_pData;
	for (int i = 0; i < A.m_rows; i++)
	{
		T* rowC = ptrC + i * C.m_cols;			// 每行C重复A.m_col次访问
		const T* rowA = ptrA + i * A.m_cols;
		for (int k = 0; k < A.m_cols; k += 4)
		{
			addDot4kx1j_avx(rowA, ptrB, rowC, B.m_cols, k);
		}
	}
}

template<typename T>
void multi8(const xMatrix<T>& A, const xMatrix<T>& B, xMatrix<T>& C)
{
	const T* ptrA = A.m_pData;
	const T* ptrB = B.m_pData;
	T* ptrC = C.m_pData;
	for (int i = 0; i < A.m_rows; i += 4)
	{
		T* rowC = ptrC + i * C.m_cols;			// 每行C重复A.m_col次访问
		const T* rowA = ptrA + i * A.m_cols;
		for (int k = 0; k < A.m_cols; k += 4)
		{
			addDot4ix4k(rowA, ptrB, rowC, A.m_cols, B.m_cols, k);
		}
	}
}

template<typename T>
void multi9(const xMatrix<T>& A, const xMatrix<T>& B, xMatrix<T>& C)
{
	//A矩阵不是4n * 4n B不是4n * 8n的情况
	int NewARow = A.m_rows;
	int NewACol = A.m_cols;
	int NewBCol = B.m_cols;

	bool isCreateA = false;
	bool isCreateB = false;

	if (A.m_rows & 0x03)
	{
		isCreateA = true;
		NewARow = (A.m_rows & ((-1) << 2)) + 4;
	}

	if (A.m_cols & 0x03)
	{
		isCreateA = true;
		isCreateB = true;
		NewACol = (A.m_cols & ((-1) << 2)) + 4;
	}

	if (B.m_cols & 0x07)
	{
		isCreateB = true;
		NewBCol = (B.m_cols & ((-1) << 3)) + 8;
	}

	const xMatrix<T>* NewA = &A;
	const xMatrix<T>* NewB = &B;
	xMatrix<T>* NewC = &C;
	if (isCreateA)
	{
		NewA = new xMatrix<T>(NewARow, NewACol, false);
		for (int i = 0; i < A.m_rows; ++i)
		{
			memcpy(NewA->m_pData + i * NewACol, A.m_pData + i * A.m_cols, A.m_cols * sizeof(T));
		}
	}

	if (isCreateB)
	{
		NewB = new xMatrix<T>(NewACol, NewBCol, false);
		for (int i = 0; i < B.m_rows; ++i)
		{
			memcpy(NewB->m_pData + i * NewBCol, B.m_pData + i * B.m_cols, B.m_cols * sizeof(T));
		}
	}

	if (isCreateA || isCreateB)
	{
		NewC = new xMatrix<T>(NewARow, NewBCol, false);
	}

	const T* ptrA = NewA->m_pData;
	const T* ptrB = NewB->m_pData;
	T* ptrC = NewC->m_pData;
	for (int i = 0; i < NewARow; i += 4)
	{
		T* rowC = ptrC + i * NewBCol;			// 每行C重复A.m_col次访问
		const T* rowA = ptrA + i * NewACol;
		for (int k = 0; k < NewACol; k += 4)
		{
			addDot4ix4k_avx(rowA, ptrB, rowC, NewACol, NewBCol, k);
		}
	}

	if (isCreateA)
	{
		delete NewA;
	}

	if (isCreateB)
	{
		delete NewB;
	}

	if (isCreateA || isCreateB)
	{
		//结果复制到原矩阵中去
		for (int i = 0; i < C.m_rows; ++i)
		{
			memcpy(C.m_pData + i * C.m_cols, NewC->m_pData + i * NewBCol, C.m_cols * sizeof(T));
		}

		delete NewC;
	}
}

template<typename T>
void multi10(const xMatrix<T>& A, const xMatrix<T>& B, xMatrix<T>& C)
{
	const int BLK_SIZE = 512;
#pragma omp parallel for num_threads(8)
	for (int i = 0; i < A.m_rows; i += BLK_SIZE)
	{
		for (int j = 0; j < B.m_cols; j += BLK_SIZE)
		{
			for (int k = 0; k < A.m_cols; k += BLK_SIZE)
			{
				subBlock(A, B, C, i, j, k, BLK_SIZE);
			}
		}
	}
}

// 一般尺寸矩阵
template<typename T>
void multi11(const xMatrix<T>& A, const xMatrix<T>& B, xMatrix<T>& C)
{
	// 	const T* ptrA = A.m_pData;
	// 	const T* ptrB = B.m_pData;
	// 	T* ptrC = C.m_pData;
	// #pragma omp parallel for num_threads(8)
	// 	for (int i = 0; i < A.m_row; i += 4)
	// 	{
	// 		T* rowC = ptrC + i * C.m_col;			// 每行C重复A.m_col次访问
	// 		const T* rowA = ptrA + i * A.m_col;
	// 		for (int k = 0; k < A.m_col; k += 4)
	// 		{
	// 			addDot4ix4k_avx(rowA, ptrB, rowC, A.m_col, B.m_col, k);
	// 		}
	// 	}

	const T* ptrA = A.m_pData;
	const T* ptrB = B.m_pData;
	T* ptrC = C.m_pData;
	int itail = A.m_rows % 4;
	int jtail = B.m_cols % 8;
	int ktail = A.m_cols % 4;
#pragma omp parallel for num_threads(8)
	for (int i = 0; i < A.m_rows - itail; i += 4)
	{
		T* rowC = ptrC + i * C.m_cols;			// 每行C重复A.m_col次访问
		const T* rowA = ptrA + i * A.m_cols;
		for (int k = 0; k < A.m_cols - ktail; k += 4)
		{
			addDot4ix4k_avx(rowA, ptrB, rowC, A.m_cols, B.m_cols, k);
		}
	}

	// 执行顺序必须是j、k、i，对应上面的i、j、k。否则会因为浮点数加法的执行顺序不一致导致结果不一样
	if (jtail != 0)
	{
		subBlockJ(A, B, C,
			0, B.m_cols - jtail, 0,
			A.m_rows - itail, jtail, A.m_cols - ktail);
	}
	if (ktail != 0)
	{
		subBlockK(A, B, C,
			0, 0, A.m_cols - ktail,
			A.m_rows - itail, B.m_cols, ktail);
	}
	if (itail != 0)
	{
		subBlockI(A, B, C,
			A.m_rows - itail, 0, 0,
			itail, B.m_cols, A.m_cols);
	}
}

template<typename T>
void multi12(const xMatrix<T>& A, const xMatrix<T>& B, xMatrix<T>& C)
{
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		A.m_rows, B.m_cols, A.m_cols, 1.0f, A.m_pData, A.m_cols, B.m_pData, B.m_cols, 0, C.m_pData, C.m_cols);
}

template<typename T>
void multi13(const xMatrix<T>& A, const xMatrix<T>& B, xMatrix<T>& C)
{
	const int BLK_I_SIZE = 512;
	const int BLK_J_SIZE = 512;
	const int BLK_K_SIZE = 512;
	//#pragma omp parallel for num_threads(8)
	for (int k = 0; k < A.m_cols; k += BLK_K_SIZE)
		//for (int i = 0; i < A.m_row; i += BLK_I_SIZE)
	{
		for (int j = 0; j < B.m_cols; j += BLK_J_SIZE)
		{
			for (int i = 0; i < A.m_rows; i += BLK_I_SIZE)
				//for (int k = 0; k < A.m_col; k += BLK_K_SIZE)
			{
				subBlock2(A, B, C, i, j, k, BLK_I_SIZE, BLK_J_SIZE, BLK_K_SIZE);
			}
		}
	}
}

//block + packed 
template<typename T>
void multi14(const xMatrix<T>& A, const xMatrix<T>& B, xMatrix<T>& C)
{
	const T* ptrA = A.m_pData;
	const T* ptrB = B.m_pData;
	T* ptrC = C.m_pData;

	int kBlocking = 512;
	int jBlocking = 512;
	for (int k = 0; k < A.m_cols; k += kBlocking)
	{
		int kb = std::min(A.m_cols - k, kBlocking);

		const T* pATemp = ptrA + k;
		const T* pBTemp = ptrB + k * B.m_cols;
		for (int j = 0; j < B.m_cols; j += jBlocking)
		{
			int jb = std::min(B.m_cols - j, jBlocking);
			subBlock3(A.m_rows, A.m_cols, B.m_cols,
				pATemp,
				pBTemp + j,
				ptrC + j,
				A.m_rows, kb, jb);
		}
	}

}

//处理A矩阵不是4n * 4n B不是4n * 8n的情况，在内部分别处理，耗时太久
template<typename T>
void multi15(const xMatrix<T>& A, const xMatrix<T>& B, xMatrix<T>& C)
{
	const T* ptrA = A.m_pData;
	const T* ptrB = B.m_pData;
	T* ptrC = C.m_pData;

	int kBlocking = 512;
	int jBlocking = 512;
	for (int k = 0; k < A.m_cols; k += kBlocking)
	{
		int kb = std::min(A.m_cols - k, kBlocking);

		const T* pATemp = ptrA + k;
		const T* pBTemp = ptrB + k * B.m_cols;
		for (int j = 0; j < B.m_cols; j += jBlocking)
		{
			int jb = std::min(B.m_cols - j, jBlocking);
			subBlock4(A.m_rows, A.m_cols, B.m_cols,
				pATemp,
				pBTemp + j,
				ptrC + j,
				A.m_rows, kb, jb);
		}
	}
}

//处理A矩阵不是4n * 4n B不是4n * 8n的情况，先把他们变成标准的，然后再运算,最终版本
template<typename T>
void multi16(const xMatrix<T>& A, const xMatrix<T>& B, xMatrix<T>& C)
{
	//A矩阵不是4n * 4n B不是4n * 8n的情况
	const T* ptrA;
	const T* ptrB;
	T* ptrC;

	int NewARow = A.m_rows;
	int NewACol = A.m_cols;
	int NewBCol = B.m_cols;

	bool isCreateA = false;
	bool isCreateB = false;

	if (A.m_rows & 0x03)
	{
		isCreateA = true;
		NewARow = (A.m_rows & ((-1) << 2)) + 4;
	}

	if (A.m_cols & 0x03)
	{
		isCreateA = true;
		isCreateB = true;
		NewACol = (A.m_cols & ((-1) << 2)) + 4;
	}

	if (B.m_cols & 0x07)
	{
		isCreateB = true;
		NewBCol = (B.m_cols & ((-1) << 3)) + 8;
	}

	const xMatrix<T>* NewA;
	const xMatrix<T>* NewB;
	xMatrix<T>* NewC;
	if (isCreateA)
	{
		NewA = new xMatrix<T>(NewARow, NewACol, false);

		for (int i = 0; i < A.m_rows; ++i)
		{
			memcpy(NewA->m_pData + i * NewACol, A.m_pData + i * A.m_cols, A.m_cols * sizeof(T));
		}
	}
	else
	{
		NewA = &A;
	}

	if (isCreateB)
	{
		NewB = new xMatrix<T>(NewACol, NewBCol, false);

		for (int i = 0; i < B.m_rows; ++i)
		{
			memcpy(NewB->m_pData + i * NewBCol, B.m_pData + i * B.m_cols, B.m_cols * sizeof(T));
		}
	}
	else
	{
		NewB = &B;
	}


	if (isCreateA || isCreateB)
	{
		NewC = new xMatrix<T>(NewARow, NewBCol, false);
	}
	else
	{
		NewC = &C;
	}

	ptrA = NewA->m_pData;
	ptrB = NewB->m_pData;
	ptrC = NewC->m_pData;

	int kBlocking = 512;
	int jBlocking = 512;
	for (int k = 0; k < NewACol; k += kBlocking)
	{
		int kb = std::min(NewACol - k, kBlocking);

		const T* pATemp = ptrA + k;
		const T* pBTemp = ptrB + k * NewBCol;
		for (int j = 0; j < NewBCol; j += jBlocking)
		{
			int jb = std::min(NewBCol - j, jBlocking);
			subBlock3(NewARow, NewACol, NewBCol,
				pATemp,
				pBTemp + j,
				ptrC + j,
				NewARow, kb, jb);
		}
	}

	if (isCreateA)
	{
		delete NewA;
	}

	if (isCreateB)
	{
		delete NewB;
	}

	if (isCreateA || isCreateB)
	{
		//结果复制到原矩阵中去
		for (int i = 0; i < C.m_rows; ++i)
		{
			memcpy(C.m_pData + i * C.m_cols, NewC->m_pData + i * NewBCol, C.m_cols * sizeof(T));
		}

		delete NewC;
	}
}

// cuda
template<typename T>
void multi17(const xMatrix<T>& A, const xMatrix<T>& B, xMatrix<T>& C)
{
	initDevice(0);

	T* pa = A.m_pData;
	T* pb = B.m_pData;
	T* pc = C.m_pData;
	size_t nsize = C.m_rows * C.m_cols;

	T* pb_d = nullptr;
	T* pa_d = nullptr;
	T* pc_d = nullptr;
	size_t pitch_a = 0;
	size_t pitch_b = 0;
	size_t pitch_c = 0;
	CHECK(cudaMallocPitch(&pa_d, &pitch_a, A.m_cols * sizeof(T), A.m_rows));
	CHECK(cudaMallocPitch(&pb_d, &pitch_b, B.m_cols * sizeof(T), B.m_rows));
	CHECK(cudaMallocPitch(&pc_d, &pitch_c, C.m_cols * sizeof(T), C.m_rows));
	CHECK(cudaMemcpy2D(pa_d, pitch_a, pa, A.m_pitch, A.m_cols * sizeof(T), A.m_rows, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy2D(pb_d, pitch_b, pb, B.m_pitch, B.m_cols * sizeof(T), B.m_rows, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());


	{
		// 		dim3 block(1024);
		// 		dim3 grid((nsize - 1) / block.x + 1);
		// 		//TIMING("WarmupMulti")
		// 		WarmupMulti << <grid, block >> > (pa_d, pb_d, pc_d,
		// 			pitch_a, pitch_b, pitch_c, A.m_rows, A.m_cols, B.m_cols);

		WarmupMultiWrap<T>(pa_d, pb_d, pc_d,
			pitch_a, pitch_b, pitch_c,
			A.m_rows, A.m_cols, B.m_cols);
		CHECK(cudaGetLastError());
		CHECK(cudaDeviceSynchronize());
	}
	// 	{
	// 		dim3 block(32, 16);
	// 		dim3 grid((C.m_cols - 1) / block.x + 1, (C.m_rows - 1) / block.y + 1);
	// 		//TIMING("MultiKernel")
	// 		MultiKernel << <grid, block >> > (pa_d, pb_d, pc_d,
	// 			pitch_a, pitch_b, pitch_c, A.m_rows, A.m_cols, B.m_cols);
	// 		CHECK(cudaGetLastError());
	// 		CHECK(cudaDeviceSynchronize());
	// 	}
	// 	{
	// 		dim3 block(TILE_WIDTH, TILE_WIDTH);
	// 		dim3 grid((C.m_cols - 1) / block.x + 1, (C.m_rows - 1) / block.y + 1);
	// 		//TIMING("MultiKernelTile")
	// 		MultiKernelTile << <grid, block >> > (pa_d, pb_d, pc_d,
	// 			pitch_a, pitch_b, pitch_c, A.m_rows, A.m_cols, B.m_cols);
	// 		CHECK(cudaGetLastError());
	// 		CHECK(cudaDeviceSynchronize());
	// 	}

	CHECK(cudaMemcpy2D(pc, C.m_pitch, pc_d, pitch_c, C.m_cols * sizeof(T), C.m_rows, cudaMemcpyDeviceToHost));
	CHECK(cudaDeviceSynchronize());

	cudaFree(pb_d);
	cudaFree(pa_d);
	cudaFree(pc_d);
	cudaDeviceReset();
}

