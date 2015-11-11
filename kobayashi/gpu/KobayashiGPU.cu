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
const int Nx = DIM_GRID_X * DIM_BLOCK_X;  // Domain size in x-direction
const int Ny = DIM_GRID_Y * DIM_BLOCK_Y;  // Domain size in y-direction
const int Nz = DIM_GRID_Z * DIM_BLOCK_Z;  // Domain size in z-direction

// Define memory size
const int Mx = Nx + 2 * BCELLS;  // Memory size in x-direction
const int My = Ny + 2 * BCELLS;  // Memory size in y-direction
const int Mz = Nz + 2 * BCELLS;  // Memory size in z-direction

// Define number of time steps
const int Nt     = 16000; // Number of time steps
const int tOut   = 8000;   // Output distance in time steps
bool WriteToDisk = true;

// Define grid spacing
__constant__ double dt = 1.0e-4;  // Size of time step [s]
__constant__ double dx = 3.0e-2;  // Grid spacing in x-direction [m]
__constant__ double dy = 3.0e-2;  // Grid spacing in y-direction [m]
__constant__ double dz = 3.0e-2;  // Grid spacing in z-direction [m]

// Kobayashi's parameters (not exactly his..)
__constant__ double Gamma     = 10.0;    // Coefficient of driving force
__constant__ double Precision = 1.e-9;   // Calculation precision
__constant__ double Radius    = 0.1;     // Initial radius of spherical grain
__constant__ double T0        = 0.0;     // Initial temperature
__constant__ double Tm        = 1.0;     // Equilibrium temperature  (no-dimension)
__constant__ double alpha     = 0.8;     // Coefficient of driving force
__constant__ double ampl      = 0.01;    // Amplitude of noise
__constant__ double delta     = 0.20;    // Anisotropy in (0,1)
__constant__ double epsilon   = 0.010;   // Gradient energy coefficient
__constant__ double kappa     = 1.7;     // Referrers to latent heat (no-dimension)
__constant__ double tau       = 3.0e-4;  // Inverse of interface mobility [s]

// Misc parameters
const int seed = 123; // Random number seed
__constant__ double LaplacianStencil27[3][3][3] = {{{1.0/30.0,   1.0/10.0, 1.0/30.0},
                                             {1.0/10.0,   7.0/15.0, 1.0/10.0},
                                             {1.0/30.0,   1.0/10.0, 1.0/30.0}},

                                            {{1.0/10.0,   7.0/15.0, 1.0/10.0},
                                             {7.0/15.0, -64.0/15.0, 7.0/15.0},
                                             {1.0/10.0,   7.0/15.0, 1.0/10.0}},

                                            {{1.0/30.0,   1.0/10.0, 1.0/30.0},
                                             {1.0/10.0,   7.0/15.0, 1.0/10.0},
                                             {1.0/30.0,   1.0/10.0, 1.0/30.0}}};///< 27 point Laplacian stencil by Spotz and Carey (1995)

void WriteToFile(const int tStep, double* field, string name);
__global__ void InitializeRandomNumbers( curandState *state);

__host__ __device__
inline int Index(int i, int j, int k)
{
    // Define index of the memory
    return (k * My + j) * Mx + i;
}

__global__
void InitializeSupercooledSphere(double* Phi, double* PhiDot, double* Temp,
        double* TempDot)
{
    // Initialization
    const double x0 = BCELLS * dx;
    const double y0 = BCELLS * dy;
    const double z0 = BCELLS * dz;

    // Define indices of the device memory
    int i = blockIdx.x * blockDim.x + threadIdx.x + BCELLS;
    int j = blockIdx.y * blockDim.y + threadIdx.y + BCELLS;
    int k = blockIdx.z * blockDim.z + threadIdx.z + BCELLS;

        int locIndex = Index(i,j,k);
        // Initialize the phase Field
        double r = sqrt(pow(i*dx-x0,2) + pow(j*dy-y0,2) + pow(k*dz-z0,2));
        if (r < Radius)
        {
            Phi [locIndex] = 1.0;
            Temp[locIndex] = Tm;
        }
        else
        {
            Phi [locIndex] = 0.0;
            Temp[locIndex] = T0;
        }
        PhiDot [locIndex] = 0.0;
        TempDot[locIndex] = 0.0;
}

// Laplace operator
__device__
double Laplace(double* field, int i, int j, int k)
{

    double Laplacian = 0.0;
    for (int ii = -1; ii <= +1; ++ii)
    for (int jj = -1; jj <= +1; ++jj)
    for (int kk = -1; kk <= +1; ++kk)
        Laplacian += LaplacianStencil27[ii+1][jj+1][kk+1]*field[Index(i+ii,j+jj,k+kk)];
    Laplacian /= dx * dx;
    return Laplacian;
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
        if ((locPhi > Precision) or  (locPhi < 1.0 - Precision))
        {

            double Q1 = 1.2*M_PI/4.0;
            double Q2 = 0.0;
            double Q3 = M_PI/4.0;

            double c1 = cos(Q1);
            double c2 = cos(Q2);
            double c3 = cos(Q3);

            double s1 = sin(Q1);
            double s2 = sin(Q2);
            double s3 = sin(Q3);

            //This matrix follows XYZ notations (http://en.wikipedia.org/wiki/Euler_angles)
            double Rot[3][3] = {{           c2*c3,           - c2*s3,      s2},
                                {c1*s3 + c3*s1*s2,  c1*c3 - s1*s2*s3,  -c2*s1},
                                {s1*s3 - c1*c3*s2,  c3*s1 + c1*s2*s3,   c1*c2}};

            // Calculate gradient of Phi
            double gradX = (Phi[Index(i+1,j,k)] - Phi[Index(i-1,j,k)])/(2*dx);
            double gradY = (Phi[Index(i,j+1,k)] - Phi[Index(i,j-1,k)])/(2*dy);
            double gradZ = (Phi[Index(i,j,k+1)] - Phi[Index(i,j,k-1)])/(2*dz);

            double grad[3] = {gradX, gradY, gradZ};
            double gradR[3] = {0.0, 0.0, 0.0};

            for(int ii = 0; ii < 3; ++ii)
            for(int jj = 0; jj < 3; ++jj)
            {
                gradR[ii] += Rot[ii][jj] * grad[jj];
            }

            gradX = gradR[0];
            gradY = gradR[1];
            gradZ = gradR[2];

            double div   = pow(pow(gradX,2) + pow(gradY,2) + pow(gradZ,2),2);
            double theta = 0.0;
            if ( div > 1.e-12)
                theta = (pow(gradX,4) + pow(gradY,4)+ pow(gradZ,4))/div;
            else
                theta = 0.0;
            double sigma = (1.-4.*delta*(1.-theta));

            // Calculate noise
            double noise = 0.0;//ampl * curand_uniform(&State[locIndex]);

            // Calculate driving force am
            double m = (alpha/M_PI) * atan(Gamma*(Tm - Temp[locIndex])*sigma);

            // Add driving force to PhiDot
            locPhiDot += locPhi*(1.0 - locPhi)*(locPhi - 0.5 + m + noise);
        }
        // Calculate Laplacian-term
        locPhiDot += pow(epsilon,2) * Laplace(Phi,i,j,k);
        locPhiDot /= tau;

        PhiDot [locIndex] += locPhiDot;
        TempDot[locIndex] += Laplace(Temp,i,j,k) + kappa * locPhiDot;

}

__global__
void ApplyTimeStep(double* field, double* fieldDot)
{
    // Define global indices of the device memory
    int i = blockIdx.x * blockDim.x + threadIdx.x + BCELLS;
    int j = blockIdx.y * blockDim.y + threadIdx.y + BCELLS;
    int k = blockIdx.z * blockDim.z + threadIdx.z + BCELLS;

        int locIndex = Index(i,j,k);
        // Update fields
        field    [locIndex] += dt * fieldDot [locIndex];
        fieldDot [locIndex]  = 0.0;

}

__global__
void SetBoundariesYZ(double* field)
{
    // Define global indices of the device memory
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    //int b = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

        // Apply mirror boundary conditions
        //field[Index(b     ,j,k)] = field[Index( 2*BCELLS-1-b,j,k)];
        //field[Index(Mx-1-b,j,k)] = field[Index(Mx-2*BCELLS+b,j,k)];
        field[Index(0   ,j,k)] = field[Index(1   ,j,k)];
        field[Index(Mx-1,j,k)] = field[Index(Mx-2,j,k)];

}

__global__
void SetBoundariesXZ(double* field)
{
    // Define global indices of the device memory
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //int b = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

        // Apply mirror boundary conditions
        //field[Index(i,b     ,k)] = field[Index(i, 2*BCELLS-1-b,k)];
        //field[Index(i,My-1-b,k)] = field[Index(i,My-2*BCELLS+b,k)];
        field[Index(i,0   ,k)] = field[Index(i,   1,k)];
        field[Index(i,My-1,k)] = field[Index(i,My-2,k)];

}

__global__
void SetBoundariesXY(double* field)
{
    // Define global indices of the device memory
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //int b = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.z * blockDim.z + threadIdx.z;

        // Apply mirror boundary conditions
        //field[Index(i,j,b     )] = field[Index(i,j, 2*BCELLS-1-b)];
        //field[Index(i,j,Mz-1-b)] = field[Index(i,j,Mz-2*BCELLS+b)];
        field[Index(i,j,0   )] = field[Index(i,j,   1)];
        field[Index(i,j,Mz-1)] = field[Index(i,j,Mz-2)];

}

__host__
void SetBoundaries(double* field)
{

    // Define grid/block structure
    dim3 dimGridYZ( 1 , 1 , Mz );
    dim3 dimGridXZ( 1 , 1 , Mx );
    dim3 dimGridXY( 1 , 1 , My );

    dim3 dimBlockYZ( My , BCELLS , 1 );
    dim3 dimBlockXZ( Mz , BCELLS , 1 );
    dim3 dimBlockXY( Mx , BCELLS , 1 );

    // Set Boundary conditions
    SetBoundariesYZ<<< dimGridYZ, dimBlockYZ >>>(field);
    SetBoundariesXZ<<< dimGridYZ, dimBlockYZ >>>(field);
    SetBoundariesXY<<< dimGridXY, dimBlockXY >>>(field);
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
    //cudaMalloc((void**)&devState,   size);  cudaMemset(devState,   0, size);

    // Verify that allocations succeeded
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess)
    {
        cout << "Failed to allocate device memory! "
             << "(" << cudaGetErrorString(err) << ")" << endl;
        exit(EXIT_FAILURE);
    }

    // Define grid/block structure
    dim3 dimGrid(   DIM_GRID_X  , DIM_GRID_Y  , DIM_GRID_Z );
    dim3 dimBlock(  DIM_BLOCK_X , DIM_BLOCK_Y , DIM_BLOCK_Z);

    // Initialize Fields
    cout << "Initialized Data: " << Nx << "x" << Ny << "x" << Nz << endl;
    InitializeSupercooledSphere<<< dimGrid, dimBlock >>>(devPhi, devPhiDot, devTemp, devPhiDot);

    // Initialize Random seed
    cout << "Initialize Random Seed.." << endl;
    //InitializeRandomNumbers<<< dimGrid, dimBlock >>>(devState);
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
                WriteToFile(tStep, Phi,  "PhaseField");
                WriteToFile(tStep, Temp, "Temperature");
            }
        }

        // Set boundary conditions
        SetBoundaries(devPhi);
        SetBoundaries(devTemp);

        // Calculate and apply time step
        CalcTimeStep <<< dimGrid, dimBlock >>>(devPhi, devPhiDot, devTemp, devTempDot, devState);

        ApplyTimeStep<<< dimGrid , dimBlock >>>(devPhi  , devPhiDot);
        ApplyTimeStep<<< dimGrid , dimBlock >>>(devTemp , devTempDot);
    }

    // Stop run time measurement
    gettimeofday(&end, NULL);
    double simTime = ((end.tv_sec  - start.tv_sec) * 1000000u
            + end.tv_usec - start.tv_usec) / 1.e6;
    cout << "Calculation time for " << Nt << " time step: " << simTime << " s" << endl;

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
        int locIndex = Index(i,j,k);
        vtk_file << field[locIndex] << endl;
    }
    vtk_file.close();
    }
}
