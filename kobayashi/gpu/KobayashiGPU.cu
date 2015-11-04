/*
 *   This is a simple example for a phase field code for the CPU
 *
 *   Created:    November 2015
 *   Author:     Raphael Schiedung;
 *   Email:      raphael.schiedung@rub.de
 */

#include <cstdlib>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string.h>
#include <sys/time.h>

using namespace std;

// Define number of Boundary Cells
#define BCELLS 1
// Define CUDA-Block dimensions
#define DIM_BLOCK_X 8
#define DIM_BLOCK_Y 8
#define DIM_BLOCK_Z 8
// Define CUDA-Grid dimensions
#define DIM_GRID_X  16
#define DIM_GRID_Y  16
#define DIM_GRID_Z  16

// Define computation domain size
const int Nx = DIM_GRID_X * DIM_BLOCK_X; // Domain size in x-direction
const int Ny = DIM_GRID_Y * DIM_BLOCK_Y; // Domain size in y-direction
const int Nz = DIM_GRID_Z * DIM_BLOCK_Z; // Domain size in z-direction

// Define memory size
const int Mx = Nx + 2 * BCELLS;  // Memory size in x-direction
const int My = Ny + 2 * BCELLS;  // Memory size in y-direction
const int Mz = Nz + 2 * BCELLS;  // Memory size in z-direction

__device__ __constant__ int devMx = DIM_GRID_X * DIM_BLOCK_X + 2 * BCELLS;
__device__ __constant__ int devMy = DIM_GRID_Y * DIM_BLOCK_Y + 2 * BCELLS;
__device__ __constant__ int devMz = DIM_GRID_Z * DIM_BLOCK_Z + 2 * BCELLS;

// Define number of time steps
const int Nt     = 2000; // Number of time steps
const int tOut   = 100;  // Output distance in time steps
bool WriteToDisk = false;

// Define grid spacing
__device__ __constant__ double dx = 0.03;    // Grid spacing in x-direction [m]
__device__ __constant__ double dy = 0.03;    // Grid spacing in y-direction [m]
__device__ __constant__ double dz = 0.03;    // Grid spacing in z-direction [m]
__device__ __constant__ double dt = 1.0e-4;  // Size of time step [s]

// Kobayashi's parameters (not exactly his..)
__device__ __constant__ double epsilon = 0.010;   // Gradient energy coefficient
__device__ __constant__ double tau     = 3.0e-4;  // Inverse of interface mobility [s]
__device__ __constant__ double alpha   = 0.8;     // Coefficient of driving force
__device__ __constant__ double Gamma   = 10.0;    // Coefficient of driving force
__device__ __constant__ double delta   = 0.10;    // Anisotropy in (0,1)
__device__ __constant__ double K       = 1.7;     // Referrers to latent heat (no-dimension)
__device__ __constant__ double T0      = 0.0;     // Initial temperature
__device__ __constant__ double Tm      = 1.0;     // Equilibrium temperature  (no-dimension)
__device__ __constant__ double ampl    = 0.01;    // Amplitude of noise
__device__ __constant__ int    seed    = 123;     // Random number seed

// Misc parameters
__device__ __constant__ double Radius        = 0.4;    // Initial radius of spherical grain
__device__ __constant__ double PhiPrecision  = 1.e-9;  // Phase-field cut off

void WriteToFile(const int tStep, double* field, string name);
__global__ void InitializeRandomNumbers( curandState *state);

__device__
inline int Index(int i, int j, int k)
{
    // Define index of the memory
    return (k * devMy + j) * devMx + i;
}

__global__
void InitializeSupercooledSphere(double* Phi, double* PhiDot, double* Temp,
        double* TempDot)
{
    // Initialization
    const double  x0 = Nx/2 * dx;
    const double  y0 = Ny/2 * dy;
    const double  z0 = Nz/2 * dz;

    // Define indices of the device memory
    int i = blockIdx.x * blockDim.x + threadIdx.x + BCELLS;
    int j = blockIdx.y * blockDim.y + threadIdx.y + BCELLS;
    int k = blockIdx.z * blockDim.z + threadIdx.z + BCELLS;

        int locIndex = Index(i,j,k);
        // Initialize the phase Field
        double r = sqrt(pow(i*dx-x0,2) + pow(j*dy-y0,2) + pow(k*dz-z0,2));
        if ( r < Radius)
        {
            Phi    [locIndex] = 1.0;
            Temp   [locIndex] = Tm;
        }
        else
        {
            Phi    [locIndex] = 0.0;
            Temp   [locIndex] = T0;
        }

        PhiDot [locIndex] = 0.0;
        TempDot[locIndex] = 0.0;
}

// Laplace operator
__device__
double Laplace(double* field, int i, int j, int k)
{
    double df2_dx2 = 0.0;
    double df2_dy2 = 0.0;
    double df2_dz2 = 0.0;

    df2_dx2 += field[Index(i+1,j,k)];
    df2_dx2 -= field[Index(i  ,j,k)] * 2.0;
    df2_dx2 += field[Index(i-1,j,k)];
    df2_dx2 /= dx*dx;

    df2_dy2 += field[Index(i,j+1,k)];
    df2_dy2 -= field[Index(i,j  ,k)] * 2.0;
    df2_dy2 += field[Index(i,j-1,k)];
    df2_dy2 /= dy*dy;

    df2_dz2 += field[Index(i,j,k+1)];
    df2_dz2 -= field[Index(i,j,k  )] * 2.0;
    df2_dz2 += field[Index(i,j,k-1)];
    df2_dz2 /= dz*dz;

    return df2_dx2 + df2_dy2 + df2_dz2;
}

__global__
void CalcTimeStep(double* Phi, double* PhiDot, double* Temp, double* TempDot, curandState * State)
{
    // Calculate PhiDot and TempDot
    int i = blockIdx.x * blockDim.x + threadIdx.x + BCELLS;
    int j = blockIdx.y * blockDim.y + threadIdx.y + BCELLS;
    int k = blockIdx.z * blockDim.z + threadIdx.z + BCELLS;

        int locIndex     = Index(i,j,k);
        double locPhiDot = 0.0;
        double locPhi    = Phi[locIndex];
        // Calculate driving force m
        if ((locPhi > 0.0) or  (locPhi < 1.0))
        {
            // Calculate gradient of Phi
            double gradX = (Phi[Index(i+1,j,k)] - Phi[Index(i-1,j,k)])/(2*dx);
            double gradY = (Phi[Index(i,j+1,k)] - Phi[Index(i,j-1,k)])/(2*dy);
            double gradZ = (Phi[Index(i,j,k+1)] - Phi[Index(i,j,k-1)])/(2*dz);
            double div   = pow(pow(gradX,2) + pow(gradY,2) + pow(gradZ,2),2);
            double theta;
            if ( div > 1.e-12)
                theta = (pow(gradX,4) + pow(gradY,4)+ pow(gradZ,4))/div;
            else
                theta = 0.0;
            double sigma = (1.-4.*delta*(1.-theta));

            // Calculate noise
            double noise = ampl * curand_uniform(&State[locIndex]);

            // Calculate driving force am
            double m = (alpha/M_PI) * atan(Gamma*(Tm - Temp[locIndex])*sigma);

            // Add driving force to PhiDot
            locPhiDot += locPhi*(1.0 - locPhi)*(locPhi - 0.5 + m + noise);
        }
        // Calculate Laplacian-term
        locPhiDot += pow(epsilon,2) * Laplace(Phi,i,j,k);
        locPhiDot /= tau;

        PhiDot [locIndex] += locPhiDot;
        TempDot[locIndex] += Laplace(Temp,i,j,k) + K * locPhiDot;

}

__global__
void ApplyTimeStep(double* Phi, double* PhiDot, double* Temp, double* TempDot)
{
    // Define global indices of the device memory
    int i = blockIdx.x * blockDim.x + threadIdx.x + BCELLS;
    int j = blockIdx.y * blockDim.y + threadIdx.y + BCELLS;
    int k = blockIdx.z * blockDim.z + threadIdx.z + BCELLS;

        int locIndex = Index(i,j,k);

        // Update phase field
        Phi    [locIndex] += dt * PhiDot [locIndex];
        PhiDot [locIndex]  = 0.0;
        Temp   [locIndex] += dt * TempDot[locIndex];
        TempDot[locIndex]  = 0.0;
}

__global__
void SetBoundariesX(double* field)
{
    // Define global indices of the device memory
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

        field[Index(0 ,j,k)] = field[Index(1   ,j,k)];
        field[Index(Nx,j,k)] = field[Index(Nx-1,j,k)];

}

__global__
void SetBoundariesY(double* field)
{
    // Define global indices of the device memory
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

        field[Index(i,0 ,k)] = field[Index(i,1   ,k)];
        field[Index(i,Ny,k)] = field[Index(i,Ny-1,k)];

}

__global__
void SetBoundariesZ(double* field)
{
    // Define global indices of the device memory
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

        field[Index(i,j,0 )] = field[Index(i,j,1   )];
        field[Index(i,j,Nz)] = field[Index(i,j,Nz-1)];

}

__host__
void SetBoundaries(double* field)
{

    // Define grid/block structure
    dim3 dimGridX(  1  , 1  , Mz );
    dim3 dimGridY(  Mx , 1  , 1  );
    dim3 dimGridZ(  1  , My , 1  );

    dim3 dimBlockX( 1  , My , 1 );
    dim3 dimBlockY( 1  , 1  , Mz);
    dim3 dimBlockZ( Mx , 1  , 1 );

    // Set Boundary conditions
    SetBoundariesX<<< dimGridX, dimBlockX >>>(field);
    SetBoundariesY<<< dimGridY, dimBlockY >>>(field);
    SetBoundariesZ<<< dimGridZ, dimBlockZ >>>(field);
}

int main()
{
    // Calculate memory size
    int numElements = Mx * My * Mz;
    size_t size = numElements * sizeof(double);

    // Define and allocate host memory
    double* Phi     = (double *)malloc(size);  memset(Phi,     0, size);
    double* Temp    = (double *)malloc(size);  memset(Temp,    0, size);

    // Allocate Device Memory
    double* devPhi;
    double* devPhiDot;
    double* devTemp;
    double* devTempDot;
    curandState* devState;  // Used for random number generation

    // Allocate device memory and initialize it
    cudaMalloc((void**)&devPhi,     size);  cudaMemset(devPhi,     0, size);
    cudaMalloc((void**)&devPhiDot,  size);  cudaMemset(devPhiDot,  0, size);
    cudaMalloc((void**)&devTemp,    size);  cudaMemset(devTemp,    0, size);
    cudaMalloc((void**)&devTempDot, size);  cudaMemset(devTempDot, 0, size);
    cudaMalloc((void**)&devState,   size);  cudaMemset(devState,   0, size);

    // Define grid/block structure
    dim3 dimGrid(   DIM_GRID_X  , DIM_GRID_Y  , DIM_GRID_Z );
    dim3 dimBlock(  DIM_BLOCK_X , DIM_BLOCK_Y , DIM_BLOCK_Z);

    // Initialize Fields
    cout << "Initialized Data: " << Nx << "x" << Ny << "x" << Nz << endl;
    InitializeSupercooledSphere<<< dimGrid, dimBlock >>>(devPhi, devPhiDot, devTemp, devPhiDot);

    // Initialize Random seed
    cout << "Initialize Random Seed.." << endl;
    InitializeRandomNumbers<<< dimGrid, dimBlock >>>(devState);
    cudaDeviceSynchronize();

    // Start run time measurement
    struct timeval start, end;
    gettimeofday(&start, NULL);

    // Start time loop
    for (int tStep = 0; tStep <= Nt; tStep++)
    {
        // Make Output if necessary
        if(tStep%tOut == 0)
        {
            cudaMemcpy(Phi,  devPhi,  size, cudaMemcpyDeviceToHost);
            cudaMemcpy(Temp, devTemp, size, cudaMemcpyDeviceToHost);
            cout << "Time step: " << tStep << "/" << Nt << endl;
            if (WriteToDisk)
            {
                WriteToFile(tStep, Phi,  "Phase-Field");
                WriteToFile(tStep, Temp, "Temperature");
            }
        }

        // Set boundary conditions
        SetBoundaries(devPhi);
        SetBoundaries(devTemp);

        // Calculate and apply time step
        CalcTimeStep <<< dimGrid, dimBlock >>>(devPhi, devPhiDot, devTemp, devTempDot, devState);
        ApplyTimeStep<<< dimGrid, dimBlock >>>(devPhi, devPhiDot, devTemp, devTempDot);
    }

    // Stop run time measurement
    gettimeofday(&end, NULL);
    double delta = ((end.tv_sec  - start.tv_sec) * 1000000u
            + end.tv_usec - start.tv_usec) / 1.e6;
    cout << "Calculation time for " << Nt << " time step: " << delta << " s" << endl;

    // Cleanup
    free(Phi);
    free(Temp);

    cudaFree(devPhi);
    cudaFree(devPhiDot);
    cudaFree(devTemp);
    cudaFree(devTempDot);

    cudaDeviceReset();

    return 0;
}

__global__
void InitializeRandomNumbers( curandState *state)
{
    // Define indices of the device memory
    int i = blockIdx.x * blockDim.x + threadIdx.x + BCELLS;
    int j = blockIdx.y * blockDim.y + threadIdx.y + BCELLS;
    int k = blockIdx.z * blockDim.z + threadIdx.z + BCELLS;

    int locIndex = Index(i,j,k);

    curand_init ( seed, locIndex, 0, &state[locIndex] );
}

void WriteToFile(const int tStep, double* field, string name)
{
    {
    stringstream filename;
    filename << "Out_" << name << "_"<< tStep << ".vtk";
    string FileName = filename.str();

    ofstream vtk_file(FileName.c_str());
    vtk_file << "# vtk DataFile Version 3.0\n";
    vtk_file << name << " \n";
    vtk_file << "ASCII\n";
    vtk_file << "DATASET RECTILINEAR_GRID\n";
    vtk_file << "DIMENSIONS " << Nx << " " << Ny << " " << Nz << endl;
    vtk_file << "X_COORDINATES " << Nx << " double\n";
    for (int i = 0; i < Nx; i++) vtk_file << i << " ";
    vtk_file << endl;
    vtk_file << "Y_COORDINATES " << Ny << " double\n";
    for (int j = 0; j < Ny; j++) vtk_file << j << " ";
    vtk_file << endl;
    vtk_file << "Z_COORDINATES " << Nz << " double\n";
    for (int k = 0; k < Nz; k++) vtk_file << k << " ";
    vtk_file << endl;
    vtk_file << "POINT_DATA " << Nx*Ny*Nz << endl;

    vtk_file << "SCALARS " << name << " double 1\n";
    vtk_file << "LOOKUP_TABLE default\n";
    for (int k = BCELLS; k < Nz + BCELLS; ++k)
    for (int j = BCELLS; j < Ny + BCELLS; ++j)
    for (int i = BCELLS; i < Nx + BCELLS; ++i)
    {
        int locIndex = (k * My + j) * Mx + i;
        vtk_file << field[locIndex] << endl;
    }
    vtk_file.close();
    }
}
