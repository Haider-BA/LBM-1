#include "headers.h"

void initF(float* f0, float* f1, float* f2, float* f3, float* f4, 
	float* f5, float* f6, float* f7, float* f8, int* solid, int totpoints, float vxin, float roout){

	float faceq1, faceq2, faceq3;
	faceq1 = 4.f/9.f;
    faceq2 = 1.f/9.f;
    faceq3 = 1.f/36.f;

    int i;

    for (i=0; i<totpoints; i++){
		f0[i] = faceq1 * roout * (1.f                             - 1.5f*vxin*vxin);
		f1[i] = faceq2 * roout * (1.f + 3.f*vxin + 4.5f*vxin*vxin - 1.5f*vxin*vxin);
		f2[i] = faceq2 * roout * (1.f                             - 1.5f*vxin*vxin);
		f3[i] = faceq2 * roout * (1.f - 3.f*vxin + 4.5f*vxin*vxin - 1.5f*vxin*vxin);
		f4[i] = faceq2 * roout * (1.f                             - 1.5f*vxin*vxin);
		f5[i] = faceq3 * roout * (1.f + 3.f*vxin + 4.5f*vxin*vxin - 1.5f*vxin*vxin);
		f6[i] = faceq3 * roout * (1.f - 3.f*vxin + 4.5f*vxin*vxin - 1.5f*vxin*vxin);
		f7[i] = faceq3 * roout * (1.f - 3.f*vxin + 4.5f*vxin*vxin - 1.5f*vxin*vxin);
		f8[i] = faceq3 * roout * (1.f + 3.f*vxin + 4.5f*vxin*vxin - 1.5f*vxin*vxin);
		solid[i] = 1;
    }
}

void drawBody(float cx, float cy, float r, int num_segments)
{ 
	int i, j, ii, i0; 
	for(ii = 0; ii < num_segments; ii++) 
	{ 
		float theta = 2.0f * 3.1415926f * ii / num_segments;//get the current angle 

		float x = r * cosf(theta);//calculate the x component
		float y = r * sinf(theta);//calculate the y component

		i = x+cx;
		j = y+cy;

		i0 = I2D(ni, i, j);
		solid[i0] = 0;

	}
}

void stream(float *tmpf0, float* tmpf1, float *tmpf2, float *tmpf3, float *tmpf4, float *tmpf5, float *tmpf6,
				 float *tmpf7, float *tmpf8, float *f0, float *f1, float *f2, float *f3, float *f4, float *f5,
				 		float *f6, float *f7, float *f8, int ni, int nj){

// Move the f values one grid spacing in the directions that they are pointing
// i.e. f1 is copied one location to the right, etc.

    int i, j, im1, ip1, jm1, jp1, i0;

    // Initially the f's are moved to temporary arrays
    for(j=0; j<nj; j++){
		jm1=j-1;
		jp1=j+1;
		if(j==0) jm1=0;
		if(j==(nj-1)) jp1=nj-1;
		for(i=0; i<ni; i++){
		    i0  = I2D(ni,i,j);
		    im1 = i-1;
		    ip1 = i+1;
		    if(i==0) im1=0;
		    if(i==(ni-1)) ip1=ni-1;
		    tmpf1[i0] = f1[I2D(ni,im1,j)];
		    tmpf2[i0] = f2[I2D(ni,i,jm1)];
		    tmpf3[i0] = f3[I2D(ni,ip1,j)];
		    tmpf4[i0] = f4[I2D(ni,i,jp1)];
		    tmpf5[i0] = f5[I2D(ni,im1,jm1)];
		    tmpf6[i0] = f6[I2D(ni,ip1,jm1)];
		    tmpf7[i0] = f7[I2D(ni,ip1,jp1)];
		    tmpf8[i0] = f8[I2D(ni,im1,jp1)];
		}
    }

    // Now the temporary arrays are copied to the main f arrays
    for(j=0; j<nj; j++){
		for(i=0; i<ni; i++){
	    	i0 = I2D(ni,i,j);
	    	f1[i0] = tmpf1[i0];
	    	f2[i0] = tmpf2[i0];
	    	f3[i0] = tmpf3[i0];
	    	f4[i0] = tmpf4[i0];
	    	f5[i0] = tmpf5[i0];
	    	f6[i0] = tmpf6[i0];
	    	f7[i0] = tmpf7[i0];
	    	f8[i0] = tmpf8[i0];
		}
    }
}

void collide(int ni, int nj, float *u, float *v, float *f0, float *f1, float *f2, float *f3, 
				float *f4, float *f5, float *f6, float *f7, float *f8, float tau){

// Collisions between the particles are modeled here. We use the very simplest
// model which assumes the f's change toward the local equlibrium value (based
// on density and velocity at that point) over a fixed timescale, tau	 

    int i,j,i0;
    float rovx, rovy, vx, vy, v_sq_term;
    float f0eq, f1eq, f2eq, f3eq, f4eq, f5eq, f6eq, f7eq, f8eq;
    float rtau, rtau1;

    // Some useful constants
    rtau = 1.f/tau;
    rtau1 = 1.f - rtau;

    for (j=0; j<nj; j++) {
		for (i=0; i<ni; i++) {
		    i0 = I2D(ni,i,j);
	
			// Do the summations needed to evaluate the density and components of velocity
			ro = f0[i0] + f1[i0] + f2[i0] + f3[i0] + f4[i0] + f5[i0] + f6[i0] + f7[i0] + f8[i0];
			rovx = f1[i0] - f3[i0] + f5[i0] - f6[i0] - f7[i0] + f8[i0];
			rovy = f2[i0] - f4[i0] + f5[i0] + f6[i0] - f7[i0] - f8[i0];
			vx = rovx/ro;
			vy = rovy/ro;
			u[i0] = vx;
			w[i0] = vy;

			// Also load the velocity magnitude into plotvar - this is what we will
			// display using OpenGL later
			plotvar[i0] = sqrt(vx*vx + vy*vy);
			//printf("plot var = %f\n", plotvar[i0]);

			v_sq_term = 1.5f*(vx*vx + vy*vy);
	
			// Evaluate the local equilibrium f values in all directions
			f0eq = ro * faceq1 * (1.f - v_sq_term);
			f1eq = ro * faceq2 * (1.f + 3.f*vx + 4.5f*vx*vx - v_sq_term);
			f2eq = ro * faceq2 * (1.f + 3.f*vy + 4.5f*vy*vy - v_sq_term);
			f3eq = ro * faceq2 * (1.f - 3.f*vx + 4.5f*vx*vx - v_sq_term);
			f4eq = ro * faceq2 * (1.f - 3.f*vy + 4.5f*vy*vy - v_sq_term);
			f5eq = ro * faceq3 * (1.f + 3.f*(vx + vy) + 4.5f*(vx + vy)*(vx + vy) - v_sq_term);
			f6eq = ro * faceq3 * (1.f + 3.f*(-vx + vy) + 4.5f*(-vx + vy)*(-vx + vy) - v_sq_term);
			f7eq = ro * faceq3 * (1.f + 3.f*(-vx - vy) + 4.5f*(-vx - vy)*(-vx - vy) - v_sq_term);
			f8eq = ro * faceq3 * (1.f + 3.f*(vx - vy) + 4.5f*(vx - vy)*(vx - vy) - v_sq_term);

			// Simulate collisions by "relaxing" toward the local equilibrium
			f0[i0] = rtau1 * f0[i0] + rtau * f0eq;
			f1[i0] = rtau1 * f1[i0] + rtau * f1eq;
			f2[i0] = rtau1 * f2[i0] + rtau * f2eq;
			f3[i0] = rtau1 * f3[i0] + rtau * f3eq;
			f4[i0] = rtau1 * f4[i0] + rtau * f4eq;
			f5[i0] = rtau1 * f5[i0] + rtau * f5eq;
			f6[i0] = rtau1 * f6[i0] + rtau * f6eq;
			f7[i0] = rtau1 * f7[i0] + rtau * f7eq;
			f8[i0] = rtau1 * f8[i0] + rtau * f8eq;
		}
    }
}

void in_BC(float vxin, float roout, int ni, int nj, float *f1, float *f5, float *f8){

// This inlet BC is extremely crude but is very stable
// We set the incoming f values to the equilibirum values assuming:
// ro=roout; vx=vxin; vy=0

    int i0, j;
    float f1new, f5new, f8new, vx_term;
    float faceq1, faceq2, faceq3;
	faceq1 = 4.f/9.f;
    faceq2 = 1.f/9.f;
    faceq3 = 1.f/36.f;

    vx_term = 1.f + 3.f*vxin +3.f*vxin*vxin;
    f1new = roout * faceq2 * vx_term;
    f5new = roout * faceq3 * vx_term;
    f8new = f5new;

    for (j=0; j<nj; j++){
    	i0 = I2D(ni,0,j);
    	f1[i0] = f1new;
    	f5[i0] = f5new;
    	f8[i0] = f8new;
    }
}

void solid_BC(float* f0, float* f1, float* f2, float* f3, float* f4, float* f5, float* f6,
				 float* f7, float* f8, int ni, int nj){

// This is the boundary condition for a solid node. All the f's are reversed -
// this is known as "bounce-back"

    int i,j,i0;
    float f1old,f2old,f3old,f4old,f5old,f6old,f7old,f8old;
    
    for (j=0;j<nj;j++){
		for (i=0;i<ni;i++){
		    i0=I2D(ni,i,j);
		    if (solid[i0]==0) {
			f1old = f1[i0];
			f2old = f2[i0];
			f3old = f3[i0];
			f4old = f4[i0];
			f5old = f5[i0];
			f6old = f6[i0];
			f7old = f7[i0];
			f8old = f8[i0];

			f1[i0] = f3old;
			f2[i0] = f4old;
			f3[i0] = f1old;
			f4[i0] = f2old;
			f5[i0] = f7old;
			f6[i0] = f8old;
			f7[i0] = f5old;
			f8[i0] = f6old;
		    }
		}
    }
}