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
	void GetTime(const char* message);

	~CudaTiming();
};

CudaTiming::CudaTiming()
{
	check(cudaEventCreate(&start));
	check(cudaEventCreate(&stop));
	check(cudaEventRecord(start, 0));
}

CudaTiming::~CudaTiming()
{
	check(cudaEventDestroy(start));
	check(cudaEventDestroy(stop));
}

void CudaTiming::Start()
{
	check(cudaEventCreate(&start));
	check(cudaEventCreate(&stop));
	check(cudaEventRecord(start, 0));
}

void CudaTiming::Stop()
{
	check(cudaEventRecord(stop, 0));
	check(cudaEventSynchronize(stop));
}

void CudaTiming::GetTime(const char* message)
{
	float elapsedTime;
	check(cudaEventElapsedTime(&elapsedTime,
		start, stop));
	std::cout << message << ": " << elapsedTime << " ms\n";
}