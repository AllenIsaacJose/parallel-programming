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


#include "reference_calc.cpp"
#include "utils.h"

__global__
void findMinMaxLogLumPerBlock(const float* const d_logLuminance,
		const size_t numRows, const size_t numCols, 
		float d_minLogLum, float d_maxLogLum)
{
  unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (i >= numCols || j >= numRows)
    return;

  unsigned int g_oneDOffset = j * numCols + i;
  unsigned int s_oneDOffset = threadIdx.y * blockDim.x + threadIdx.x;
  unsigned int threadsPerBlock = blockDim.x * blockDim.y;
  __shared__ float s_minLogLum[threadsPerBlock];
  __shared__ float s_maxLogLum[threadsPerBlock];

  s_minLogLum[s_oneDOffset] = d_logLuminance[g_oneDOffset];
  s_maxLogLum[s_oneDOffset] = d_logLuminance[g_oneDOffset];
  __syncthreads();

  for (size_t it = threadsPerBlock / 2; it > 0; it >>= 1)
  {
    if (s_oneDOffset < it)
      s_minLogLum[s_oneDOffset] = min(s_minLogLum[s_oneDOffset], s_minLogLum[s_oneDOffset + it]);
    __syncthreads();
  }

  if(s_oneDOffset == 0)
    d_minLogLum[blockIdx.y * gridDim.x + blockIdx.x] = s_minLogLum[0];
  __syncthreads();
  
  for (size_t it = threadsPerBlock / 2; it > 0; it >>= 1)
  {
    if (s_oneDOffset < it)
      s_maxLogLum[s_oneDOffset] = max(s_maxLogLum[s_oneDOffset], s_maxLogLum[s_oneDOffset + it]);
    __syncthreads();
  }

  if(s_oneDOffset == 0)
    d_maxLogLum[blockIdx.y * gridDim.x + blockIdx.x] = s_maxLogLum[0];
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

  unsigned int threadsPerBlock = numRows * numCols;
  __shared__ float s_minLogLumArray[threadsPerBlock];
  __shared__ float s_maxLogLumArray[threadsPerBlock];

  s_minLogLumArray[i] = d_minLogLumArray[i];
  s_maxLogLumArray[i] = d_maxLogLumArray[i];
  __syncthreads();

  for (size_t it = threadsPerBlock / 2; it > 0; it >>= 1)
  {
    if (i < it)
      s_minLogLumArray[i] = max(s_minLogLumArray[i], s_minLogLumArray[i + it]);
    __syncthreads();
  }

  if(i == 0)
    *d_minLogLum = s_minLogLumArray[0];
 
  __syncthreads();

  for (size_t it = threadsPerBlock / 2; it > 0; it >>= 1)
  {
    if (i < it)
      s_maxLogLumArray[i] = max(s_maxLogLumArray[i], s_maxLogLumArray[i + it]);
    __syncthreads();
  }

  if(i == 0)
    *d_maxLogLum = s_maxLogLumArray[0];

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
  unsigned int threads = 32;

  // Number of blocks per grid
  unsigned int blocksX = (numCols + threads - 1) / threads;
  unsigned int blocksY = (numRows + threads - 1) / threads;

  // Allocate memory for min and max logLum
  checkCudaErrors(cudaMalloc(&d_minLogLumPtr, sizeof(float) * blockX * blockY);
  checkCudaErrors(cudaMalloc(&d_maxLogLumPtr, sizeof(float) * blockX * blockY);
  checkCudaErrors(cudaMemset(d_minLogLumPtr, 0, sizeof(float) * blockX * blockY);
  checkCudaErrors(cudaMemset(d_maxLogLumPtr, 0, sizeof(float) * blockX * blockY);

  dim3 threadsPerBlock(threads, threads, 1);
  dim3 blocksPerGrid((blocksX, blockY, 1);

  findMinMaxLogLumPerBlock<<<blocksPerGrid, threadsPerBlock>>>(d_logLuminance,
                        numRows, numCols, d_minLogLumPtr, d_maxLogLumPtr);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  float* d_minLogLum = nullptr;
  float* d_maxLogLum = nullptr;
  checkCudaErrors(cudaMalloc(&d_minLogLum, sizeof(float));
  checkCudaErrors(cudaMalloc(&d_maxLogLum, sizeof(float));
  checkCudaErrors(cudaMemset(d_minLogLum, 0, sizeof(float));
  checkCudaErrors(cudaMemset(d_maxLogLumP, 0, sizeof(float));
  reduceMinMaxLumPerBlock<<<1,blocksX*blocksY>>>(d_minLogLumPtr, d_maxLogLumPtr, blocksX, blockY, d_minLogLum, d_maxLogLum);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaMemcpy(&min_logLum, d_minLogLum, sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(&max_logLum, d_maxLogLum, sizeof(float), cudaMemcpyDeviceToHost));
}
