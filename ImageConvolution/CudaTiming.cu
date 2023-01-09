#include "CudaTiming.cuh"

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

void CudaTiming::PrintTime(const char* message)
{
	float elapsedTime;
	check(cudaEventElapsedTime(&elapsedTime,
		start, stop));
	std::cout << message << ": " << elapsedTime << " ms\n";
}

float CudaTiming::GetTime()
{
	float elapsedTime;
	check(cudaEventElapsedTime(&elapsedTime,
		start, stop));

	return elapsedTime;
}