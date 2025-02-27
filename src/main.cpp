#include "xMatrix.h"
#include <spdlog/spdlog.h>


constexpr int M = 1024;
constexpr int N = 1024;
constexpr int K = 1024;

int main()
{
	auto A = xMatrixf::makeRandMat(M, N);
	auto B = xMatrixf::makeRandMat(N, K);
	xMatrixf C1(M, K);
	xMatrixf C2(M, K);
	xMatrixf C3(M, K);
	spdlog::info("matrix multiply A[{}][{}] x B[{}][{}]", M, N, N, K);

	spdlog::info("multi1 begin...");
	multi1(*A, *B, C1);
	spdlog::info("multi1 done.");

	spdlog::info("multi17 begin...");
	multi17(*A, *B, C2);
	spdlog::info("multi17 done.");
	assert(isEqual(C1, C2));

	spdlog::info("multi18 begin...");
	multi18(*A, *B, C3);
	spdlog::info("multi18 done.");
	assert(isEqual(C1, C3));

	return 0;
}