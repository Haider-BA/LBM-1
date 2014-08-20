#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cutil.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_X 16
#define BLOCK_Y 8
#define GPU 0

#define I2D(ni,i,j) (((ni)*(j)) + i)

//C functions
void stream(float *, float *, float *, float *, float *, float *, float *, float *, float *,
				float *, float *, float *, float *, float *, float *, float *, float *, float *, int, int);
void collide(int, int, float *, float *, float *, float *, float *, float *, 
				float *, float *, float *, float *, float *, float);
void solid_BC(float *, float *, float *, float *, float *, float *, float *, float *, float *, int, int);
void in_BC(float, float, int, int, float *, float *, float *);
void initF(float*, float*, float*, float*, float*, float*,
				float*, float*, float*, int*, int, float, float);
void drawBody(float, float, float, int);

//CUDA kernel C wrappers
void d_stream(void);
void d_collide(void);
void d_BCs(void);

//CUDA kernels
__global__ void stream_kernel (int pitch, float *f1_data, float *f2_data,
                               float *f3_data, float *f4_data, float *f5_data, float *f6_data,
                               float *f7_data, float *f8_data);

__global__ void collide_kernel (int pitch, float tau, float faceq1, float faceq2, float faceq3,
                                float *f0_data, float *f1_data, float *f2_data,
                                float *f3_data, float *f4_data, float *f5_data, float *f6_data,
                                float *f7_data, float *f8_data, float *plot_data);

__global__ void BCs_kernel (int ni, int nj, int pitch, float vxin, float roout,
                                  float faceq2, float faceq3,
                                  float *f0_data, float *f1_data, float *f2_data,
                                  float *f3_data, float *f4_data, float *f5_data, 
                                  float *f6_data, float *f7_data, float *f8_data, int* solid_data);