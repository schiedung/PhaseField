/*
 *   This is a simple example for a phase field code for the CPU
 *
 *   Created:    November 2015
 *   Author:     Raphael Schiedung;
 *   Email:      raphael.schiedung@rub.de
 */

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <random>
#include <sstream>
#include <string.h>
#include <sys/time.h>

using namespace std;

// Define number of Boundary Cells
#define BCELLS 1

// Define computation domain size
const int Nx = 128;  // Domain size in x-direction
const int Ny = 128;  // Domain size in y-direction
const int Nz = 128;  // Domain size in z-direction

// Define memory size
const int Mx = Nx + 2 * BCELLS;  // Memory size in x-direction
const int My = Ny + 2 * BCELLS;  // Memory size in y-direction
const int Mz = Nz + 2 * BCELLS;  // Memory size in z-direction

// Define number of time steps
const int Nt     = 5000; // Number of time steps
const int tOut   = 1000;  // Output distance in time steps
bool WriteToDisk = false;

// Define grid spacing
const float dt = 1.0e-4;  // Size of time step [s]
const float dx = 3.0e-2;  // Grid spacing in x-direction [m]
const float dy = 3.0e-2;  // Grid spacing in y-direction [m]
const float dz = 3.0e-2;  // Grid spacing in z-direction [m]

// Kobayashi's parameters (not exactly his..)
const float Gamma     = 10.0;    // Coefficient of driving force
const float Precision = 1.e-9;   // Calculation precision
const float Radius    = 0.1;     // Initial radius of spherical grain
const float T0        = 0.0;     // Initial temperature
const float Tm        = 1.0;     // Equilibrium temperature  (no-dimension)
const float alpha     = 0.8;     // Coefficient of driving force
const float ampl      = 0.01;    // Amplitude of noise
const float delta     = 0.20;    // Anisotropy in (0,1)
const float epsilon   = 0.010;   // Gradient energy coefficient
const float kappa     = 1.7;     // Referrers to latent heat (no-dimension)
const float tau       = 3.0e-4;  // Inverse of interface mobility [s]

const float LaplacianStencil27[3][3][3] = {{{1.0/30.0,   1.0/10.0, 1.0/30.0},
                                             {1.0/10.0,   7.0/15.0, 1.0/10.0},
                                             {1.0/30.0,   1.0/10.0, 1.0/30.0}},

                                            {{1.0/10.0,   7.0/15.0, 1.0/10.0},
                                             {7.0/15.0, -64.0/15.0, 7.0/15.0},
                                             {1.0/10.0,   7.0/15.0, 1.0/10.0}},

                                            {{1.0/30.0,   1.0/10.0, 1.0/30.0},
                                             {1.0/10.0,   7.0/15.0, 1.0/10.0},
                                             {1.0/30.0,   1.0/10.0, 1.0/30.0}}};///< 27 point Laplacian stencil by Spotz and Carey (1995)

void WriteToFile(const int tStep, float* field, string name);
std::default_random_engine generator;
std::normal_distribution<float> distribution(0.0,0.5);

// Calculates the 1d memory index at certain grind point (i,j,k)
inline int Index(int i, int j, int k)
{
    // Define index of the memory
    return (k * My + j) * Mx + i;
}

void InitializeSupercooledSphere(float* Phi, float* PhiDot, float* Temp,
        float* TempDot)
{
    // Initialization
    const float x0 = BCELLS * dx;
    const float y0 = BCELLS * dy;
    const float z0 = BCELLS * dz;

    #pragma omp parallel for schedule(auto) collapse(2)
    for (int i = BCELLS; i < Nx + BCELLS; i++)
    for (int j = BCELLS; j < Ny + BCELLS; j++)
    for (int k = BCELLS; k < Nz + BCELLS; k++)
    {
        int locIndex = Index(i,j,k);
        // Initialize the phase Field
        float r = sqrt(pow(i*dx-x0,2) + pow(j*dy-y0,2) + pow(k*dz-z0,2));
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
}

// Laplace operator
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


void CalcTimeStep(float* Phi, float* PhiDot, float* Temp, float* TempDot)
{
    #pragma omp parallel for schedule(auto) collapse(2)
    for (int i = BCELLS; i < Nx + BCELLS; i++)
    for (int j = BCELLS; j < Ny + BCELLS; j++)
    for (int k = BCELLS; k < Nz + BCELLS; k++)
    {
        int locIndex     = Index(i,j,k);
        float locPhiDot = 0.0;
        float locPhi    = Phi[locIndex];
        // Calculate driving force m
        if ((locPhi > Precision) or  (locPhi < 1.0 - Precision))
        {

            float Q1 = 1.2*M_PI/4.0;
            float Q2 = 0.0;
            float Q3 = M_PI/4.0;

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
            float noise = ampl;// *  distribution(generator);

            // Calculate driving force am
            float m = (alpha/M_PI) * atan(Gamma*(Tm - Temp[locIndex])*sigma);

            // Add driving force to PhiDot
            locPhiDot += locPhi*(1.0 - locPhi)*(locPhi - 0.5 + m + noise);
        }
        // Calculate Laplacian-term
        locPhiDot += pow(epsilon,2) * Laplace(Phi,i,j,k);
        locPhiDot /= tau;

        PhiDot [locIndex] += locPhiDot;
        TempDot[locIndex] += Laplace(Temp,i,j,k) + kappa * locPhiDot;
    }
}

void ApplyTimeStep(float* field, float* fieldDot)
{
    #pragma omp parallel for schedule(auto) collapse(2)
    for (int i = BCELLS; i < Nx + BCELLS; i++)
    for (int j = BCELLS; j < Ny + BCELLS; j++)
    for (int k = BCELLS; k < Nz + BCELLS; k++)
    {
        int locIndex = Index(i,j,k);
        // Update fields
        field    [locIndex] += dt * fieldDot [locIndex];
        fieldDot [locIndex]  = 0.0;
    }
}

void SetBoundariesYZ(float* field)
{
    #pragma omp parallel for schedule(auto) collapse(1)
    for (int j = 0; j < My;     j++)
    for (int k = 0; k < Mz;     k++)
    for (int b = 0; b < BCELLS; b++)
    {
        // Apply mirror boundary conditions
        field[Index(b     ,j,k)] = field[Index( 2*BCELLS-1-b,j,k)];
        field[Index(Mx-1-b,j,k)] = field[Index(Mx-2*BCELLS+b,j,k)];
    }
}

void SetBoundariesXZ(float* field)
{
    #pragma omp parallel for schedule(auto) collapse(1)
    for (int i = 0; i < Mx;     i++)
    for (int k = 0; k < Mz;     k++)
    for (int b = 0; b < BCELLS; b++)
    {
        // Apply mirror boundary conditions
        field[Index(i,b     ,k)] = field[Index(i, 2*BCELLS-1-b,k)];
        field[Index(i,My-1-b,k)] = field[Index(i,My-2*BCELLS+b,k)];
    }
}

void SetBoundariesXY(float* field)
{
    #pragma omp parallel for schedule(auto) collapse(1)
    for (int i = 0; i < Mx;     i++)
    for (int j = 0; j < My;     j++)
    for (int b = 0; b < BCELLS; b++)
    {
        // Apply mirror boundary conditions
        field[Index(i,j,b     )] = field[Index(i,j, 2*BCELLS-1-b)];
        field[Index(i,j,Mz-1-b)] = field[Index(i,j,Mz-2*BCELLS+b)];
    }
}

void SetBoundaries(float* field)
{
    // Set Boundary conditions
    SetBoundariesYZ(field);
    SetBoundariesXZ(field);
    SetBoundariesXY(field);
}
int main()
{
    // Calculate memory size
    int numElements = Mx * My * Mz;
    size_t size = numElements * sizeof(float);

    // Define and allocate host memory
    float* Phi     = (float *)malloc(size);  memset(Phi,     0, size);
    float* PhiDot  = (float *)malloc(size);  memset(PhiDot,  0, size);
    float* Temp    = (float *)malloc(size);  memset(Temp,    0, size);
    float* TempDot = (float *)malloc(size);  memset(TempDot, 0, size);

    // Initialize Fields
    cout << "Initialized Data: " << Nx << "x" << Ny << "x" << Nz << endl;
    InitializeSupercooledSphere(Phi, PhiDot, Temp, PhiDot);

    // Start run time measurement
    struct timeval start, end;
    gettimeofday(&start, NULL);

    // Start time loop
    for (int tStep = 0; tStep <= Nt; tStep++)
    {
        // Make Output if necessary
        if(tStep%tOut == 0)
        {
            cout << "Time step: " << tStep << "/" << Nt << endl;
            if (WriteToDisk)
            {
                WriteToFile(tStep, Phi,  "PhaseField");
                WriteToFile(tStep, Temp, "Temperature");
            }
        }

        // Set boundary conditions
        SetBoundaries(Phi);
        SetBoundaries(Temp);

        // Calculate time step
        CalcTimeStep(Phi,  PhiDot, Temp, TempDot);

        // Apply time step
        ApplyTimeStep(Phi  , PhiDot);
        ApplyTimeStep(Temp , TempDot);
    }

    // Stop run time measurement
    gettimeofday(&end, NULL);
    float simTime = ((end.tv_sec  - start.tv_sec) * 1000000u
            + end.tv_usec - start.tv_usec) / 1.e6;
    cout << "Calculation time for " << Nt << " time step: " << simTime << " s" << endl;

    // Cleanup
    free(Phi);
    free(PhiDot);
    free(Temp);
    free(TempDot);
    return 0;
}


void WriteToFile(const int tStep, float* field, string name)
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
    vtk_file << "X_COORDINATES " << Nx << " float\n";
    for (int i = 0; i < Nx; i++) vtk_file << i << " ";
    vtk_file << endl;
    vtk_file << "Y_COORDINATES " << Ny << " float\n";
    for (int j = 0; j < Ny; j++) vtk_file << j << " ";
    vtk_file << endl;
    vtk_file << "Z_COORDINATES " << Nz << " float\n";
    for (int k = 0; k < Nz; k++) vtk_file << k << " ";
    vtk_file << endl;
    vtk_file << "POINT_DATA " << Nx*Ny*Nz << endl;

    vtk_file << "SCALARS " << name << " float 1\n";
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
