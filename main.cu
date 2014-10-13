#include "headers.h"
#include "utils.c"

void printProperties(float dia, float tau, float vxin){

	printf("The diameter of the cylindrical body: %f\n", dia);
	printf("Kinematic viscosity of fluid: %f\n", (tau-0.5)/3);
	printf("Inlet Velocity: %f\n", vxin);

}

//Declaring variables and arrays - HOST
float *f0,*f1,*f2,*f3,*f4,*f5,*f6,*f7,*f8;
float *tmpf0,*tmpf1,*tmpf2,*tmpf3,*tmpf4,*tmpf5,*tmpf6,*tmpf7,*tmpf8;
int *solid;
float *u, *w;

//Declaring - DEVICE
float *f0_data, *f1_data, *f2_data, *f3_data, *f4_data;
float *f5_data, *f6_data, *f7_data, *f8_data, *plot_data;
int *solid_data;

//Declaring variables
float tau, re, fx, fy, cd, cl, ro; 
float vxin, roout, dia, temp_rad;
float width, height;
int ni, nj, tstep;
int ncol;
int ipos_old,jpos_old, draw_solid_flag;
int array_size_2d, totpoints, i;

int main(){

	//Initializing basic properties of problem and fluid
	ni = 400;
    nj = 240;
    tstep = 1000;
    roout = 1.0;
    dia = 24.0;
    tau= 0.51;
    printf("Enter Reynolds number: ");
    scanf("%f", &re);
    vxin = (re/dia)*(tau - 0.5)/3;
    totpoints = ni*nj;    
    array_size_2d = ni*nj*sizeof(float);

    printProperties(dia, tau, vxin);

    //Allocating memory on Host
    f0 = malloc(array_size_2d);
    f1 = malloc(array_size_2d);
    f2 = malloc(array_size_2d);
    f3 = malloc(array_size_2d);
    f4 = malloc(array_size_2d);
    f5 = malloc(array_size_2d);
    f6 = malloc(array_size_2d);
    f7 = malloc(array_size_2d);
    f8 = malloc(array_size_2d);
    tmpf0 = malloc(array_size_2d);
    tmpf1 = malloc(array_size_2d);
    tmpf2 = malloc(array_size_2d);
    tmpf3 = malloc(array_size_2d);
    tmpf4 = malloc(array_size_2d);
    tmpf5 = malloc(array_size_2d);
    tmpf6 = malloc(array_size_2d);
    tmpf7 = malloc(array_size_2d);
    tmpf8 = malloc(array_size_2d);
    solid = malloc(ni*nj*sizeof(int));
    u = malloc(array_size_2d);
    w = malloc(array_size_2d);

    if(GPU){
        //Allocate memory on Device
        cudaMallocPitch((void **)&f0_data, &pitch, 
                                       sizeof(float)*ni, nj);
        cudaMallocPitch((void **)&f1_data, &pitch, 
                                       sizeof(float)*ni, nj);
        cudaMallocPitch((void **)&f2_data, &pitch, 
                                       sizeof(float)*ni, nj);
        cudaMallocPitch((void **)&f3_data, &pitch, 
                                       sizeof(float)*ni, nj);
        cudaMallocPitch((void **)&f4_data, &pitch, 
                                       sizeof(float)*ni, nj);
        cudaMallocPitch((void **)&f5_data, &pitch, 
                                       sizeof(float)*ni, nj);
        cudaMallocPitch((void **)&f6_data, &pitch, 
                                       sizeof(float)*ni, nj);
        cudaMallocPitch((void **)&f7_data, &pitch, 
                                       sizeof(float)*ni, nj);
        cudaMallocPitch((void **)&f8_data, &pitch, 
                                       sizeof(float)*ni, nj);
        cudaMallocPitch((void **)&solid_data, &pitch, 
                                       sizeof(int)*ni, nj);

        desc = cudaCreateChannelDesc<float>();
        cudaMallocArray(&f1_array, &desc, ni, nj);
        cudaMallocArray(&f2_array, &desc, ni, nj);
        cudaMallocArray(&f3_array, &desc, ni, nj);
        cudaMallocArray(&f4_array, &desc, ni, nj);
        cudaMallocArray(&f5_array, &desc, ni, nj);
        cudaMallocArray(&f6_array, &desc, ni, nj);
        cudaMallocArray(&f7_array, &desc, ni, nj);
        cudaMallocArray(&f8_array, &desc, ni, nj);
    }

    //Initialize 'f' values
    InitF(f0, f1, f2, f3, f4, f5, f6, f7, f8, solid, totpoints, vxin, roout);

    if(GPU){
        //Transfer initialized values to Device vars
        cudaMemcpy2D((void *)f0_data, pitch, (void *)f0,
                                    sizeof(float)*ni,sizeof(float)*ni, nj,
                                    cudaMemcpyHostToDevice);
        cudaMemcpy2D((void *)f1_data, pitch, (void *)f1,
                                    sizeof(float)*ni,sizeof(float)*ni, nj,
                                    cudaMemcpyHostToDevice);
        cudaMemcpy2D((void *)f2_data, pitch, (void *)f2,
                                    sizeof(float)*ni,sizeof(float)*ni, nj,
                                    cudaMemcpyHostToDevice);
        cudaMemcpy2D((void *)f3_data, pitch, (void *)f3,
                                    sizeof(float)*ni,sizeof(float)*ni, nj,
                                    cudaMemcpyHostToDevice);
        cudaMemcpy2D((void *)f4_data, pitch, (void *)f4,
                                    sizeof(float)*ni,sizeof(float)*ni, nj,
                                    cudaMemcpyHostToDevice);
        cudaMemcpy2D((void *)f5_data, pitch, (void *)f5,
                                    sizeof(float)*ni,sizeof(float)*ni, nj,
                                    cudaMemcpyHostToDevice);
        cudaMemcpy2D((void *)f6_data, pitch, (void *)f6,
                                    sizeof(float)*ni,sizeof(float)*ni, nj,
                                    cudaMemcpyHostToDevice);
        cudaMemcpy2D((void *)f7_data, pitch, (void *)f7,
                                    sizeof(float)*ni,sizeof(float)*ni, nj,
                                    cudaMemcpyHostToDevice);
        cudaMemcpy2D((void *)f8_data, pitch, (void *)f8,
                                    sizeof(float)*ni,sizeof(float)*ni, nj,
                                    cudaMemcpyHostToDevice);
        cudaMemcpy2D((void *)solid_data, pitch, (void *)solid,
                                    sizeof(int)*ni,sizeof(int)*ni, nj,
                                    cudaMemcpyHostToDevice);
    }

    //Draw the solid body
    for(temp_rad=1.0; temp_rad<=dia/2; temp_rad++){
        drawBody(40.0, 120.0, temp_rad, temp_rad*temp_rad*10);
    }

    //compute for all time steps
    if(GPU==0){
        while(tstep--){
            //streaming function
            stream(tmpf0, tmpf1, tmpf2, tmpf3, tmpf4, tmpf5, tmpf6, tmpf7, tmpf8,
                        f0, f1, f2, f3, f4, f5, f6, f7, f8, ni, nj);
            //solid boundary condition
            solid_BC(f0, f1, f2, f3, f4, f5, f6, f7, f8, ni, nj);
            //inlet boundary condition
            in_BC(vxin, roout, ni, nj, f1, f5, f8);
            //collision step
            collide(ni, nj, u, v, f0, f1, f2, f3, f4, f5, f6, f7, f8, tau);

            //write data to output files
        }
    }
    else{
        while(tstep--){
            //streaming function - Device
            d_stream();
            //boundary condition - Device
            d_BCs();
            //collision step - Device
            d_collide();

            //write data to output files
        }    
    }

	return 0;
}

__global__ void stream_kernel (int pitch, float *f1_data, float *f2_data,
                               float *f3_data, float *f4_data, float *f5_data,
                               float *f6_data, float *f7_data, float *f8_data){
// Stream CUDA kernel

    int i, j, i2d;

    i = blockIdx.x*TILE_I + threadIdx.x;
    j = blockIdx.y*TILE_J + threadIdx.y;

    i2d = i + j*pitch/sizeof(float);

    // look up the adjacent f's needed for streaming using textures
    // i.e. gather from textures, write to device memory: f1_data, etc
    f1_data[i2d] = tex2D(f1_tex, (float) (i-1)  , (float) j);
    f2_data[i2d] = tex2D(f2_tex, (float) i      , (float) (j-1));
    f3_data[i2d] = tex2D(f3_tex, (float) (i+1)  , (float) j);
    f4_data[i2d] = tex2D(f4_tex, (float) i      , (float) (j+1));
    f5_data[i2d] = tex2D(f5_tex, (float) (i-1)  , (float) (j-1));
    f6_data[i2d] = tex2D(f6_tex, (float) (i+1)  , (float) (j-1));
    f7_data[i2d] = tex2D(f7_tex, (float) (i+1)  , (float) (j+1));
    f8_data[i2d] = tex2D(f8_tex, (float) (i-1)  , (float) (j+1));
}

void d_stream(void){
// C wrapper
    // Device-to-device mem-copies to transfer data from linear memory (f1_data)
    // to CUDA format memory (f1_array) so we can use these in textures
    cudaMemcpy2DToArray(f1_array, 0, 0, (void *)f1_data, pitch,
                                       sizeof(float)*ni, nj,
                                       cudaMemcpyDeviceToDevice);
    cudaMemcpy2DToArray(f2_array, 0, 0, (void *)f2_data, pitch,
                                       sizeof(float)*ni, nj,
                                       cudaMemcpyDeviceToDevice);
    cudaMemcpy2DToArray(f3_array, 0, 0, (void *)f3_data, pitch,
                                       sizeof(float)*ni, nj,
                                       cudaMemcpyDeviceToDevice);
    cudaMemcpy2DToArray(f4_array, 0, 0, (void *)f4_data, pitch,
                                       sizeof(float)*ni, nj,
                                       cudaMemcpyDeviceToDevice);
    cudaMemcpy2DToArray(f5_array, 0, 0, (void *)f5_data, pitch,
                                       sizeof(float)*ni, nj,
                                       cudaMemcpyDeviceToDevice);
    cudaMemcpy2DToArray(f6_array, 0, 0, (void *)f6_data, pitch,
                                       sizeof(float)*ni, nj,
                                       cudaMemcpyDeviceToDevice);
    cudaMemcpy2DToArray(f7_array, 0, 0, (void *)f7_data, pitch,
                                       sizeof(float)*ni, nj,
                                       cudaMemcpyDeviceToDevice);
    cudaMemcpy2DToArray(f8_array, 0, 0, (void *)f8_data, pitch,
                                       sizeof(float)*ni, nj,
                                       cudaMemcpyDeviceToDevice);


    // Tell CUDA that we want to use f1_array etc as textures. Also
    // define what type of interpolation we want (nearest point)
    f1_tex.filterMode = cudaFilterModePoint;
    cudaBindTextureToArray(f1_tex, f1_array);

    f2_tex.filterMode = cudaFilterModePoint;
    cudaBindTextureToArray(f2_tex, f2_array);

    f3_tex.filterMode = cudaFilterModePoint;
    cudaBindTextureToArray(f3_tex, f3_array);

    f4_tex.filterMode = cudaFilterModePoint;
    cudaBindTextureToArray(f4_tex, f4_array);

    f5_tex.filterMode = cudaFilterModePoint;
    cudaBindTextureToArray(f5_tex, f5_array);

    f6_tex.filterMode = cudaFilterModePoint;
    cudaBindTextureToArray(f6_tex, f6_array);

    f7_tex.filterMode = cudaFilterModePoint;
    cudaBindTextureToArray(f7_tex, f7_array);

    f8_tex.filterMode = cudaFilterModePoint;
    cudaBindTextureToArray(f8_tex, f8_array);

    dim3 grid = dim3(ni/TILE_I, nj/TILE_J);
    dim3 block = dim3(TILE_I, TILE_J);

    stream_kernel<<<grid, block>>>(pitch, f1_data, f2_data, f3_data, f4_data,
                                   f5_data, f6_data, f7_data, f8_data);
    
    //CUT_CHECK_ERROR("stream failed.");

    cudaUnbindTexture(f1_tex);
    cudaUnbindTexture(f2_tex);
    cudaUnbindTexture(f3_tex);
    cudaUnbindTexture(f4_tex);
    cudaUnbindTexture(f5_tex);
    cudaUnbindTexture(f6_tex);
    cudaUnbindTexture(f7_tex);
    cudaUnbindTexture(f8_tex);
}

__global__ void BCs_kernel (int ni, int nj, int pitch, float vxin, float roout,
                                  float faceq2, float faceq3,
                                  float *f0_data, float *f1_data, float *f2_data,
                                  float *f3_data, float *f4_data, float *f5_data, 
                                  float *f6_data, float *f7_data, float *f8_data,
                                  int* solid_data){
// CUDA kernel all BC's apart from periodic boundaries:

    int i, j, i2d, i2d2;
    float v_sq_term;
    float f1old, f2old, f3old, f4old, f5old, f6old, f7old, f8old;
    
    i = blockIdx.x*TILE_I + threadIdx.x;
    j = blockIdx.y*TILE_J + threadIdx.y;

    i2d = i + j*pitch/sizeof(float);

    // Solid BC: "bounce-back"
    if (solid_data[i2d] == 0) {
      f1old = f1_data[i2d];
      f2old = f2_data[i2d];
      f3old = f3_data[i2d];
      f4old = f4_data[i2d];
      f5old = f5_data[i2d];
      f6old = f6_data[i2d];
      f7old = f7_data[i2d];
      f8old = f8_data[i2d];
      
      f1_data[i2d] = f3old;
      f2_data[i2d] = f4old;
      f3_data[i2d] = f1old;
      f4_data[i2d] = f2old;
      f5_data[i2d] = f7old;
      f6_data[i2d] = f8old;
      f7_data[i2d] = f5old;
      f8_data[i2d] = f6old;
    }


    // Inlet BC - very crude
    if (i == 0) {
      v_sq_term = 1.5f*(vxin * vxin);
      
      f1_data[i2d] = roout * faceq2 * (1.f + 3.f*vxin + 3.f*v_sq_term);
      f5_data[i2d] = roout * faceq3 * (1.f + 3.f*vxin + 3.f*v_sq_term);
      f8_data[i2d] = roout * faceq3 * (1.f + 3.f*vxin + 3.f*v_sq_term);

    }
        
    // Exit BC - very crude
    if (i == (ni-1)) {
      i2d2 = i2d - 1;
      f3_data[i2d] = f3_data[i2d2];
      f6_data[i2d] = f6_data[i2d2];
      f7_data[i2d] = f7_data[i2d2];

    }
}

void d_BCs(void){
// C wrapper
    dim3 grid = dim3(ni/TILE_I, nj/TILE_J);
    dim3 block = dim3(TILE_I, TILE_J);

    BCs_kernel<<<grid, block>>>(ni, nj, pitch, vxin, roout, faceq2,faceq3,
                                      f0_data, f1_data, f2_data,
                                      f3_data, f4_data, f5_data, 
                                      f6_data, f7_data, f8_data, solid_data);
    
    //CUT_CHECK_ERROR("apply_BCs failed.");
}

__global__ void collide_kernel (int pitch, float tau, float faceq1, float faceq2, float faceq3,
                                float *f0_data, float *f1_data, float *f2_data,
                                float *f3_data, float *f4_data, float *f5_data, float *f6_data,
                                float *f7_data, float *f8_data, float *plot_data){
// Collision CUDA kernel

    int i, j, i2d;
    float ro, vx, vy, v_sq_term, rtau, rtau1;
    float f0now, f1now, f2now, f3now, f4now, f5now, f6now, f7now, f8now;
    float f0eq, f1eq, f2eq, f3eq, f4eq, f5eq, f6eq, f7eq, f8eq;
    
    
    i = blockIdx.x*TILE_I + threadIdx.x;
    j = blockIdx.y*TILE_J + threadIdx.y;

    i2d = i + j*pitch/sizeof(float);

    rtau = 1.f/tau;
    rtau1 = 1.f - rtau;    

    // Read all f's and store in registers
    f0now = f0_data[i2d];
    f1now = f1_data[i2d];
    f2now = f2_data[i2d];
    f3now = f3_data[i2d];
    f4now = f4_data[i2d];
    f5now = f5_data[i2d];
    f6now = f6_data[i2d];
    f7now = f7_data[i2d];
    f8now = f8_data[i2d];

    // Macroscopic flow props:
    ro =  f0now + f1now + f2now + f3now + f4now + f5now + f6now + f7now + f8now;
    vx = (f1now - f3now + f5now - f6now - f7now + f8now)/ro;
    vy = (f2now - f4now + f5now + f6now - f7now - f8now)/ro;

    // Set plotting variable to velocity magnitude
    plot_data[i2d] = sqrtf(vx*vx + vy*vy);
    
    // Calculate equilibrium f's
    v_sq_term = 1.5f*(vx*vx + vy*vy);
    f0eq = ro * faceq1 * (1.f - v_sq_term);
    f1eq = ro * faceq2 * (1.f + 3.f*vx + 4.5f*vx*vx - v_sq_term);
    f2eq = ro * faceq2 * (1.f + 3.f*vy + 4.5f*vy*vy - v_sq_term);
    f3eq = ro * faceq2 * (1.f - 3.f*vx + 4.5f*vx*vx - v_sq_term);
    f4eq = ro * faceq2 * (1.f - 3.f*vy + 4.5f*vy*vy - v_sq_term);
    f5eq = ro * faceq3 * (1.f + 3.f*(vx + vy) + 4.5f*(vx + vy)*(vx + vy) - v_sq_term);
    f6eq = ro * faceq3 * (1.f + 3.f*(-vx + vy) + 4.5f*(-vx + vy)*(-vx + vy) - v_sq_term);
    f7eq = ro * faceq3 * (1.f + 3.f*(-vx - vy) + 4.5f*(-vx - vy)*(-vx - vy) - v_sq_term);
    f8eq = ro * faceq3 * (1.f + 3.f*(vx - vy) + 4.5f*(vx - vy)*(vx - vy) - v_sq_term);

    // Do collisions
    f0_data[i2d] = rtau1 * f0now + rtau * f0eq;
    f1_data[i2d] = rtau1 * f1now + rtau * f1eq;
    f2_data[i2d] = rtau1 * f2now + rtau * f2eq;
    f3_data[i2d] = rtau1 * f3now + rtau * f3eq;
    f4_data[i2d] = rtau1 * f4now + rtau * f4eq;
    f5_data[i2d] = rtau1 * f5now + rtau * f5eq;
    f6_data[i2d] = rtau1 * f6now + rtau * f6eq;
    f7_data[i2d] = rtau1 * f7now + rtau * f7eq;
    f8_data[i2d] = rtau1 * f8now + rtau * f8eq;
}

void d_collide(void){
// C wrapper


    dim3 grid = dim3(ni/TILE_I, nj/TILE_J);
    dim3 block = dim3(TILE_I, TILE_J);

    collide_kernel<<<grid, block>>>(pitch, tau, faceq1, faceq2, faceq3,
                                    f0_data, f1_data, f2_data, f3_data, f4_data,
                                    f5_data, f6_data, f7_data, f8_data, plot_data);
    
    //CUT_CHECK_ERROR("collide failed.");
}
