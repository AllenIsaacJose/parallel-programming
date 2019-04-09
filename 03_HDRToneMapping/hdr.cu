%%cuda --name student_func.cu

/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.


  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"

#include <stdio.h>
#include <float.h>
#include <limits.h>

__device__ float _min(float a, float b) {
	return a < b ? a : b;
}

__device__ float _max(float a, float b) {
	return a > b ? a : b;
}

__global__
void findMinMaxLogLumPerBlock(const float* const d_logLuminance,
		const size_t numRows, const size_t numCols,
		float* d_minLogLum, float* d_maxLogLum)
{
  unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (i >= numCols || j >= numRows)
    return;

  unsigned int g_oneDOffset = j * numCols + i;
  unsigned int s_oneDOffset = threadIdx.y * blockDim.x + threadIdx.x;
  const unsigned int threadsPerBlock = blockDim.x * blockDim.y;

  extern __shared__ float s_minMaxLogLum[];

  s_minMaxLogLum[s_oneDOffset] = d_logLuminance[g_oneDOffset];
  s_minMaxLogLum[threadsPerBlock + s_oneDOffset] = d_logLuminance[g_oneDOffset];
  __syncthreads();

  for (size_t it = threadsPerBlock / 2; it > 0; it >>= 1)
  {
    if (s_oneDOffset < it)
      s_minMaxLogLum[s_oneDOffset] = min(s_minMaxLogLum[s_oneDOffset],
                                             s_minMaxLogLum[s_oneDOffset + it]);
    __syncthreads();
  }

  if(s_oneDOffset == 0)
    d_minLogLum[blockIdx.y * gridDim.x + blockIdx.x] = s_minMaxLogLum[0];
  __syncthreads();

  for (size_t it = threadsPerBlock / 2; it > 0; it >>= 1)
  {
    if (s_oneDOffset < it)
      s_minMaxLogLum[threadsPerBlock + s_oneDOffset] =
                           max(s_minMaxLogLum[threadsPerBlock + s_oneDOffset],
                           s_minMaxLogLum[threadsPerBlock + s_oneDOffset + it]);
    __syncthreads();
  }

  if(s_oneDOffset == 0)
    d_maxLogLum[blockIdx.y * gridDim.x + blockIdx.x] =
                                                s_minMaxLogLum[threadsPerBlock];
}

__global__
void reduceMinMaxLumPerBlock(float* const d_minLogLumArray,
                             float* const d_maxLogLumArray,
			     const size_t numRows,
			     const size_t numCols,
                             float* d_minLogLum,
                             float* d_maxLogLum)
{
  unsigned int i = threadIdx.x;

  if (i >= (numCols * numRows))
    return;

  const unsigned int blocksPerGrid = numRows * numCols;

  extern __shared__ float s_minMaxLogLumArray[];

  s_minMaxLogLumArray[i] = d_minLogLumArray[i];
  s_minMaxLogLumArray[i + blocksPerGrid] = d_maxLogLumArray[i];
  __syncthreads();

  for (size_t it = blocksPerGrid / 2; it > 0; it >>= 1)
  {
    if (i < it)
      s_minMaxLogLumArray[i] = min(s_minMaxLogLumArray[i],
                                                   s_minMaxLogLumArray[i + it]);
    __syncthreads();
  }

  if(i == 0)
    *d_minLogLum = s_minMaxLogLumArray[0];

  __syncthreads();

  for (size_t it = blocksPerGrid / 2; it > 0; it >>= 1)
  {
    if (i < it)
      s_minMaxLogLumArray[i + blocksPerGrid] =
                                   max(s_minMaxLogLumArray[i + blocksPerGrid],
                                   s_minMaxLogLumArray[i + blocksPerGrid + it]);
    __syncthreads();
  }

  if(i == 0)
    *d_maxLogLum = s_minMaxLogLumArray[blocksPerGrid];

}

__global__
void calculateHisto(const float* const d_logLuminance,
                    const size_t numRows,
                    const size_t numCols,
                    const size_t numBins,
                    float* d_minLogLum,
                    float* d_rangeLogLum,
                    unsigned int* d_histo)
{
  unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (i >= numCols || j >= numRows)
    return;

  unsigned int g_oneDOffset = j * numCols + i;

  unsigned int binNum = min(static_cast<unsigned int>(numBins - 1),
                        static_cast<unsigned int>(((d_logLuminance[g_oneDOffset]
                              - (*d_minLogLum)) / (*d_rangeLogLum)) * numBins));

  atomicAdd(&(d_histo[binNum]), 1);

}

__global__
void hellisAndSteeleCDF(unsigned int* d_histo, const size_t numBins,
                        unsigned int* d_cdf)
{
  extern __shared__ unsigned int temp[];
	unsigned int g_oneDOffset = (blockIdx.x * blockDim.x) + threadIdx.x;
  
  if (g_oneDOffset >= numBins)
    return;
	
  unsigned int pout = 0,pin=1;
	
  if(g_oneDOffset != 0)
    temp[g_oneDOffset] = d_histo[g_oneDOffset-1]; //exclusive scan
	else
    temp[g_oneDOffset] = 0;
  
  __syncthreads();

	for (size_t off = 1; off < numBins; off <<= 1) {
		pout = 1 - pout;
		pin = 1 - pout;
		if (g_oneDOffset >= off)
      temp[numBins * pout + g_oneDOffset] = temp[numBins * pin + g_oneDOffset]
                                     + temp[numBins * pin + g_oneDOffset - off];
		else
      temp[numBins * pout + g_oneDOffset] = temp[numBins * pin + g_oneDOffset];
		__syncthreads();
	}
	d_cdf[g_oneDOffset] = temp[pout * numBins + g_oneDOffset];

}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

  // Calculate min and max logLum per block and copy it back to global memory
  float* d_minLogLumPtr = nullptr;
  float* d_maxLogLumPtr = nullptr;

  // Number of threads per block (32 * 32)
  const unsigned int threads = 32;

  // Number of blocks per grid
  unsigned int blocksX = (numCols + threads - 1) / threads;
  unsigned int blocksY = (numRows + threads - 1) / threads;

  // Allocate memory for min and max logLum
  checkCudaErrors(cudaMalloc(&d_minLogLumPtr,
                                            sizeof(float) * blocksX * blocksY));
  checkCudaErrors(cudaMalloc(&d_maxLogLumPtr,
                                            sizeof(float) * blocksX * blocksY));
  checkCudaErrors(cudaMemset(d_minLogLumPtr, 0,
                                            sizeof(float) * blocksX * blocksY));
  checkCudaErrors(cudaMemset(d_maxLogLumPtr, 0,
                                            sizeof(float) * blocksX * blocksY));

  dim3 threadsPerBlock(threads, threads, 1);
  dim3 blocksPerGrid(blocksX, blocksY, 1);

  const unsigned int numThreadsPerBlock = threads * threads;
  findMinMaxLogLumPerBlock<<<blocksPerGrid, threadsPerBlock, 2 *
                           numThreadsPerBlock * sizeof(float)>>>(d_logLuminance,
                              numRows, numCols, d_minLogLumPtr, d_maxLogLumPtr);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  float* d_minLogLum = nullptr;
  float* d_maxLogLum = nullptr;
  checkCudaErrors(cudaMalloc(&d_minLogLum, sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_maxLogLum, sizeof(float)));
  checkCudaErrors(cudaMemset(d_minLogLum, 0, sizeof(float)));
  checkCudaErrors(cudaMemset(d_maxLogLum, 0, sizeof(float)));

  const unsigned int numblocksPerGrid = blocksY * blocksX;
  reduceMinMaxLumPerBlock<<<1, blocksX * blocksY, 2 * numblocksPerGrid *
                               sizeof(float)>>>(d_minLogLumPtr, d_maxLogLumPtr,
                                    blocksY, blocksX, d_minLogLum, d_maxLogLum);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaMemcpy(&min_logLum, d_minLogLum, sizeof(float),
                                                       cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(&max_logLum, d_maxLogLum, sizeof(float),
                                                       cudaMemcpyDeviceToHost));
  
  float range_logLum = max_logLum - min_logLum;
  float* d_rangeLogLum = nullptr;
  unsigned int* d_histo = nullptr;
  checkCudaErrors(cudaMalloc(&d_rangeLogLum, sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_histo, numBins * sizeof(unsigned int)));
  checkCudaErrors(cudaMemcpy(d_rangeLogLum, &range_logLum, sizeof(float),
                                                       cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemset(d_histo, 0, numBins * sizeof(unsigned int)));
  calculateHisto<<<blocksPerGrid, threadsPerBlock>>>(d_logLuminance, numRows,
                        numCols, numBins, d_minLogLum, d_rangeLogLum, d_histo);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaMemset(d_cdf, 0, numBins * sizeof(unsigned int)));
  unsigned int threadsPerBlockCDF = threads * threads;
  unsigned int blocksPerGridCDF = (numBins + ((threads * threads) - 1)) / 
                                                            (threads * threads);
  hellisAndSteeleCDF<<<blocksPerGridCDF, threadsPerBlockCDF, 2 * numBins *
                               sizeof(unsigned int)>>>(d_histo, numBins, d_cdf);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // Free allocated memory
  checkCudaErrors(cudaFree(d_minLogLumPtr));
  checkCudaErrors(cudaFree(d_maxLogLumPtr));
  checkCudaErrors(cudaFree(d_minLogLum));
  checkCudaErrors(cudaFree(d_maxLogLum));
  checkCudaErrors(cudaFree(d_rangeLogLum));
  checkCudaErrors(cudaFree(d_histo));
}
