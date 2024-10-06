#include "util.h"
#include <cuda_runtime.h>
#include <stdio.h>


void initDevice(int devNum)
{
	int dev = devNum;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("Using device %d: %s\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));
}