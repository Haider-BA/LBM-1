#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define I2D(ni,i,j) (((ni)*(j)) + i)

void stream( ?????? );
void collide( ????? );
void solid_BC( ???? );
void in_BC( ??????? );
void ex_BC_crude(?? );
void initF(float*, float*, float*, float*, float*, float*,
				float*, float*, float*, int*, int, float, float);
void drawBody(float, float, float, int);
void in_BC(float, float, int, int, float *, float *, float *);
void collide(int ni, int nj, float *u, float *v, float *f0, float *f1, float *f2, float *f3, 
				float *f4, float *f5, float *f6, float *f7, float *f8, float tau);
