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
const int tOut   = 100; // Output distance in time steps
bool WriteToDisk = true;

// Define grid spacing
__device__ __constant__ double dx = 1.0e-3;  // Grid spacing in x-direction [m]
__device__ __constant__ double dy = 1.0e-3;  // Grid spacing in y-direction [m]
__device__ __constant__ double dz = 1.0e-3;  // Grid spacing in z-direction [m]
__device__ __constant__ double dt = 1.0e-3;  // Size of time step [s]

// Some material parameters (aluminium)
#define rho 2.702000e+3  // Density Al [Kg/m3] 
#define dH  1.069768e+5  // Latent heat of fusion [J/mol] 
#define cp  3.169866e+1  // Heat capacity[J/(mol K)] 
#define a   2.698150e-2  // Molar mass [Kg/mol]

// Physical parameters
__device__ __constant__ double mu     = 1.0e-5;     // Interface mobility [m^2/J*s]
__device__ __constant__ double sigma0 = 0.55;       // Interface energy [J/m]
__device__ __constant__ double dS     = 11.4744;    // Entropy constant [J/(K mol?)]
__device__ __constant__ double eta    = 1.0e-2;     // Interface width [m]
__device__ __constant__ double alpha0 = 3.52e-5;    // Thermal diffusivity [m2/s]
__device__ __constant__ double alpha1 = 6.80e-5;    // Thermal diffusivity [m2/s]
__device__ __constant__ double kappa  = a*dH/(cp);  // Latent heat parameter [K] L/(rho*cp)
__device__ __constant__ double Tm     = 933.47;     // Melting temperature
__device__ __constant__ double Ts     = 933.00;     // Initial solid temperature
//#define mu 4*alpha0/(3*dS*kappa*eta)

// Misc parameters
__device__ __constant__ double ampl         = 0.01;   // Amplitude of noise
__device__ __constant__ int    seed         = 123;    // Random number seed
__device__ __constant__ int    iRadius      = 10;     // Initial radius of spherical grain
__device__ __constant__ double PhiPrecision = 1.e-9;  // Phase-field cut off

void WriteToFile(const int tStep, double* field, string name);
__global__ void InitializeRandomNumbers( curandState *state);


// Calculates the 1d memory index at certain grind point (i,j,k)
__device__
inline int Index(int i, int j, int k)
{
    // Define index of the memory
    return (k * devMy + j) * devMx + i;
}

__device__
double temperature_steady_state(const double x, const double v)
{
    double A  = pow(M_PI*alpha0,2) / (pow(M_PI*alpha0,2) + pow(eta*v,2));
    double Tx = 0.0;

    // Calculate the temperature Tx at the point x
    if ((x >= 0.0) and (x <= eta))
    {
        Tx -= cos(M_PI * x / eta);
        Tx += v * eta / (M_PI * alpha0) * sin(M_PI * x / eta);
        Tx += exp(-v * x / alpha0);
        Tx *= A;
        Tx -= 1.0 - cos(M_PI * x / eta);
        Tx *= 0.5 * kappa;
        Tx += Ts;
    }
    else
    {
        Tx += A * exp(-v * x / alpha0) * (exp(v * eta / alpha0) + 1.0) - 2.0 ;
        Tx *= 0.5 * kappa ;
        Tx += Ts;
    }

    return Tx;
}

__global__
void InitializeSupercooledSphere(double* Phi, double* PhiDot, double* Temp,
        double* TempDot)
{
    // Initialization
    const double  i0 = Nx/2;
    const double  j0 = Ny/2;
    const double  k0 = Nz/2;

    // Define indices of the device memory
    int i = blockIdx.x * blockDim.x + threadIdx.x + BCELLS;
    int j = blockIdx.y * blockDim.y + threadIdx.y + BCELLS;
    int k = blockIdx.z * blockDim.z + threadIdx.z + BCELLS;

        double velocity = 1.0e-3;
        int locIndex = Index(i,j,k);
        // Initialize the phase Field
        double r = sqrt(pow((i-i0)*dx,2) + pow((j-j0)*dy,2) + pow((k-k0)*dz,2));
        double x = r - iRadius * dx;
        if (x < 0.0)
        {
            Phi [locIndex] = 1.0;
            Temp[locIndex] = Ts;
        }
        else if (x < eta)
        {
            Phi [locIndex] = 0.5 + 0.5 * cos(M_PI / eta * x);
            Temp[locIndex] = temperature_steady_state(x,velocity);
        }
        else
        {
            Phi [locIndex] = 0.0;
            Temp[locIndex] = temperature_steady_state(x,velocity);
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

// Calculates the thermal diffusivity inside the interface
__device__
inline double alpha(double Phi)
{
    return Phi * alpha1 + (1.0 - Phi) * alpha0; 
}

// Calculates the interface energy
__device__
inline double sigma(int i, int j, int k)
{
    return sigma0;
}

__global__
void CalcTimeStep(double* Phi, double* PhiDot, double* Temp, double* TempDot, curandState * State)
{
    // Calculate PhiDot and TempDot
    int i = blockIdx.x * blockDim.x + threadIdx.x + BCELLS;
    int j = blockIdx.y * blockDim.y + threadIdx.y + BCELLS;
    int k = blockIdx.z * blockDim.z + threadIdx.z + BCELLS;

        int    locIndex   = Index(i,j,k);
        double locPhi     = Phi[locIndex];
        double locPhiDot  = 0.0;
        double locTempDot = 0.0;

        double sig = 0.0;
        if ((locPhi < 1.0) or (locPhi > 0.0)) sig =  1.0;
        if ((locPhi > 1.0) or (locPhi < 0.0)) sig = -1.0;

        locPhiDot  += sig * sigma(i,j,k) * pow(M_PI/eta,2) * (locPhi - 0.5);

        double qPhi = 0.0;
        if (sig == 1.0) qPhi = locPhi * (1.0 - locPhi);
        locPhiDot  -= M_PI/(eta) * sqrt(qPhi) * dS * (Temp[locIndex] - Tm);

        // Calculate Laplacian of the phase field
        locPhiDot  += sigma(i,j,k)  * Laplace(Phi,i,j,k);

        // Apply interface mobility
        locPhiDot  *= mu;
        
        // Calculate time derivative of the temperature
        locTempDot += alpha(locPhi) * Laplace(Temp,i,j,k) + kappa * locPhiDot;

        // Write time derivatives into the device memory
        PhiDot [locIndex] += locPhiDot;
        TempDot[locIndex] += locTempDot;
}

__global__
void ApplyTimeStep(double* Phi, double* PhiDot, double* Temp, double* TempDot)
{
    // Define global indices of the device memory
    int i = blockIdx.x * blockDim.x + threadIdx.x + BCELLS;
    int j = blockIdx.y * blockDim.y + threadIdx.y + BCELLS;
    int k = blockIdx.z * blockDim.z + threadIdx.z + BCELLS;

        int locIndex = Index(i,j,k);

        // Update fields
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
    dim3 dimGrid(  DIM_GRID_X  , DIM_GRID_Y  , DIM_GRID_Z );
    dim3 dimBlock( DIM_BLOCK_X , DIM_BLOCK_Y , DIM_BLOCK_Z);

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
