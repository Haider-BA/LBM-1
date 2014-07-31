#include "headers.h"
#include "utils.c"

void printProperties(float dia, float tau, float vxin){

	printf("The diameter of the cylindrical body: %f\n", dia);
	printf("Kinematic viscosity of fluid: %f\n", (tau-0.5)/3);
	printf("Inlet Velocity: %f\n", vxin);

}

int main(){
	
	//Declaring variables and arrays
	float *f0,*f1,*f2,*f3,*f4,*f5,*f6,*f7,*f8;
	float *tmpf0,*tmpf1,*tmpf2,*tmpf3,*tmpf4,*tmpf5,*tmpf6,*tmpf7,*tmpf8;

	//Declaring variables
	float tau,faceq1,faceq2,faceq3, re, fx, fy, cd, cl, ro; 
	float vxin, roout, dia;
	float width, height;
	int ni,nj;
	int ncol;
	int ipos_old,jpos_old, draw_solid_flag;
	int array_size_2d, totpoints, i;

	//Initializing basic properties of problem and fluid
	ni = 400;
    nj = 240;
    roout = 1.0;
    dia = 24.0;
    tau= 0.51;
    printf("Enter Reynolds number: ");
    scanf("%f", &re);
    vxin = (re/dia)*(tau - 0.5)/3;
    totpoints = ni*nj;    
    array_size_2d = ni*nj*sizeof(float);

    printProperties(dia, tau, vxin);

    //Allocating memory for arrays
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

    //Initialize 'f' values
    Init_f(f0, f1, f2, f3, f4, f5, f6, f7, f8, solid, totpoints, vxin, roout);

    

	return 0;
}