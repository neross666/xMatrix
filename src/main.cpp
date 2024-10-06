#include "xMatrix.h"
#include <spdlog/spdlog.h>


#define M 4
#define N 4
#define K 4

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

	spdlog::info("multi2 begin...");
	multi2(*A, *B, C2);
	spdlog::info("multi2 done.");

	spdlog::info("multi17 begin...");
	multi17(*A, *B, C3);
	spdlog::info("multi17 done.");

	print(C1);
	print(C3);
	assert(isEqual(C1, C3));

	print(*A);
	print(*B);
	print(C1);

	return 0;
}