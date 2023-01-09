#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#pragma once

#define check(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

class CudaTiming
{
private:
	cudaEvent_t start, stop;
public:
	CudaTiming();

	void Start();
	void Stop();
	void PrintTime(const char* message);
	float GetTime();
	~CudaTiming();
};