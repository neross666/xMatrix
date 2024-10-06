#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#pragma region 乘法运算


void WarmupMultiWrap(float* src1, float* src2, float* dst,
	size_t pitch_src1, size_t pitch_src2, size_t pitch_dst,
	size_t M, size_t N, size_t S);

#pragma endregion 乘法运算
