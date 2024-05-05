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
//#include <numbers>
//#include <sys/time.h>

#include "wrapper.h"
#include "vtk.h"

constexpr double pi = 3.14159265359;

using namespace std;


// Define number of time steps
const int Nt      = 5000; // Number of time steps
const int tOut    = 1000;  // Output distance in time steps
const int tScreen = 10;    // Output distance to screen
bool WriteToDisk = false;

// Define grid spacing
__constant__ float dt = 1.0e-4;  // Size of time step [s]
__constant__ float dx = 3.0e-2;  // Grid spacing in x-direction [m]
__constant__ float dy = 3.0e-2;  // Grid spacing in y-direction [m]
__constant__ float dz = 3.0e-2;  // Grid spacing in z-direction [m]

// Kobayashi's parameters (not exactly his..)
__constant__ float Gamma     = 10.0;    // Coefficient of driving force
__constant__ float Precision = 1.e-9;   // Calculation precision
__constant__ float Radius    = 0.1;     // Initial radius of spherical grain
__constant__ float T0        = 0.0;     // Initial temperature
__constant__ float Tm        = 1.0;     // Equilibrium temperature  (no-dimension)
__constant__ float alpha     = 0.8;     // Coefficient of driving force
__constant__ float ampl      = 0.01;    // Amplitude of noise
__constant__ float delta     = 0.20;    // Anisotropy in (0,1)
__constant__ float epsilon   = 0.010;   // Gradient energy coefficient
__constant__ float kappa     = 1.7;     // Referrers to latent heat (no-dimension)
__constant__ float tau       = 3.0e-4;  // Inverse of interface mobility [s]

__constant__ float LaplacianStencil27[3][3][3] = {{{1.0/30.0,   1.0/10.0, 1.0/30.0},
                                             {1.0/10.0,   7.0/15.0, 1.0/10.0},
                                             {1.0/30.0,   1.0/10.0, 1.0/30.0}},

                                            {{1.0/10.0,   7.0/15.0, 1.0/10.0},
                                             {7.0/15.0, -64.0/15.0, 7.0/15.0},
                                             {1.0/10.0,   7.0/15.0, 1.0/10.0}},

                                            {{1.0/30.0,   1.0/10.0, 1.0/30.0},
                                             {1.0/10.0,   7.0/15.0, 1.0/10.0},
                                             {1.0/30.0,   1.0/10.0, 1.0/30.0}}};///< 27 point Laplacian stencil by Spotz and Carey (1995)
// Misc parameters
const int seed = 123; // Random number seed

__global__
void InitializeSupercooledSphere(float* Phi, float* PhiDot, float* Temp,
        float* TempDot)
{
    // Initialization
    const float x0 = BCELLS * dx;
    const float y0 = BCELLS * dy;
    const float z0 = BCELLS * dz;
    const int locIndex = index();

    // Initialize the phase Field
    float r = sqrt(pow(index_i()*dx-x0,2) + pow(index_j()*dy-y0,2) + pow(index_k()*dz-z0,2));
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

// Laplace operatorhttps://hub.virtamate.com/resources/yuffie-ffvii-monthly-commission.15728/
__device__
float Laplace(float* field, int i, int j, int k)
{
    float Laplacian = 0.0;
    for (int ii = -1; ii <= +1; ++ii)
    for (int jj = -1; jj <= +1; ++jj)
    for (int kk = -1; kk <= +1; ++kk)
        Laplacian += LaplacianStencil27[ii+1][jj+1][kk+1]*field[Index(i+ii,j+jj,k+kk)];
    Laplacian /= dx * dx;
    return Laplacian;
}

__global__
void CalcTimeStep(float* Phi, float* PhiDot, float* Temp, float* TempDot)//, curandState * State)
{
    // Calculate PhiDot and TempDot
    const int i = index_i();
    const int j = index_j();
    const int k = index_k();
    const int locIndex = index();

    float locPhiDot = 0.0;
    float locPhi    = Phi[locIndex];
    // Calculate driving force m
    if ((locPhi > Precision) ||  (locPhi < 1.0 - Precision))
    {

        float Q1 = 1.2*pi/4.0;
        float Q2 = 0.0;
        float Q3 = pi/4.0;

        float c1 = cos(Q1);
        float c2 = cos(Q2);
        float c3 = cos(Q3);

        float s1 = sin(Q1);
        float s2 = sin(Q2);
        float s3 = sin(Q3);

        //This matrix follows XYZ notations (http://en.wikipedia.org/wiki/Euler_angles)
        float Rot[3][3] = {{           c2*c3,           - c2*s3,      s2},
                            {c1*s3 + c3*s1*s2,  c1*c3 - s1*s2*s3,  -c2*s1},
                            {s1*s3 - c1*c3*s2,  c3*s1 + c1*s2*s3,   c1*c2}};

        // Calculate gradient of Phi
        float gradX = (Phi[Index(i+1,j,k)] - Phi[Index(i-1,j,k)])/(2*dx);
        float gradY = (Phi[Index(i,j+1,k)] - Phi[Index(i,j-1,k)])/(2*dy);
        float gradZ = (Phi[Index(i,j,k+1)] - Phi[Index(i,j,k-1)])/(2*dz);

        float grad[3] = {gradX, gradY, gradZ};
        float gradR[3] = {0.0, 0.0, 0.0};

        for(int ii = 0; ii < 3; ++ii)
        for(int jj = 0; jj < 3; ++jj)
        {
            gradR[ii] += Rot[ii][jj] * grad[jj];
        }

        gradX = gradR[0];
        gradY = gradR[1];
        gradZ = gradR[2];

        float div   = pow(pow(gradX,2) + pow(gradY,2) + pow(gradZ,2),2);
        float theta = 0.0;
        if ( div > 1.e-12)
            theta = (pow(gradX,4) + pow(gradY,4)+ pow(gradZ,4))/div;
        else
            theta = 0.0;
        float sigma = (1.-4.*delta*(1.-theta));

        // Calculate noise
        float noise = 0.0;//ampl * curand_uniform(&State[locIndex]);

        // Calculate driving force am
        float m = (alpha/pi) * atan(Gamma*(Tm - Temp[locIndex])*sigma);

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
void ApplyTimeStep(float* field, float* fieldDot)
{
    const int locIndex = index();

    field    [locIndex] += dt * fieldDot [locIndex];
    fieldDot [locIndex]  = 0.0;
}


int main()
{
    // Calculate memory size
    int numElements = Mx * My * Mz;
    size_t size = numElements * sizeof(float);

    float* Phi;     cudaMallocManaged((void**)&Phi,     size); cudaMemset(Phi,     0.f, size);
    float* Temp;    cudaMallocManaged((void**)&Temp,    size); cudaMemset(Temp,    0.f, size);
    float* PhiDot;  cudaMalloc       ((void**)&PhiDot,  size); cudaMemset(PhiDot,  0.f, size);
    float* TempDot; cudaMalloc       ((void**)&TempDot, size); cudaMemset(TempDot, 0.f, size);

    // Verify that allocations succeeded
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess)
    {
        cout << "Failed to allocate device memory! "
             << "(" << cudaGetErrorString(err) << ")\n" << endl;
        exit(EXIT_FAILURE);
    }

    // Initialize Fields
    cout << "Initialized Data: " << Nx << "x" << Ny << "x" << Nz << endl;
    InvokeKernel(InitializeSupercooledSphere, Phi, PhiDot, Temp, PhiDot);

    // Initialize Random seed
    //cout << "Initialize Random Seed.." << endl;
    cudaDeviceSynchronize();

    // Start run time measurement
    //struct timeval start, end;
    //gettimeofday(&start, NULL);

    // Start time loop
    for (int tStep = 0; tStep <= Nt; tStep++)
    {
        // Make Output if necessary
        if(tStep%tScreen == 0)
        {
            cout << "Time step: " << tStep << "/" << Nt << "\n";
            if (WriteToDisk &&  tStep%tOut == 0)
            {
                cout << "Write to file \n";
                WriteToFile(tStep, Phi,  "PhaseField");
                WriteToFile(tStep, Temp, "Temperature");
            }
        }

        SetBoundaries(Phi);
        SetBoundaries(Temp);

        InvokeKernel(CalcTimeStep,  Phi,  PhiDot, Temp, TempDot);
        InvokeKernel(ApplyTimeStep, Phi,  PhiDot);
        InvokeKernel(ApplyTimeStep, Temp, TempDot);
    }

    // Stop run time measurement
    //gettimeofday(&end, NULL);
    //float simTime = ((end.tv_sec  - start.tv_sec) * 1000000u
    //        + end.tv_usec - start.tv_usec) / 1.e6;
    //cout << "Calculation time for " << Nt << " time step: " << simTime << " s" << endl;

    // Cleanup
    cudaFree(Phi);
    cudaFree(PhiDot);
    cudaFree(Temp);
    cudaFree(TempDot);

    cudaDeviceReset();

    return 0;
}


