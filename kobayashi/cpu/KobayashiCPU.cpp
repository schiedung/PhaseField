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
const int Nt     = 2000; // Number of time steps
const int tOut   = 100;  // Output distance in time steps
bool WriteToDisk = false;

// Define grid spacing
const double dx = 0.03;   // Grid spacing in x-direction [m]
const double dy = 0.03;   // Grid spacing in y-direction [m]
const double dz = 0.03;   // Grid spacing in z-direction [m]
const double dt = 1.0e-4; // Size of time step [s]

// Kobayashi's parameters (not exactly his..)
const double epsilon = 0.010;   // Gradient energy coefficient
const double tau     = 3.0e-4;  // Inverse of interface mobility [s]
const double alpha   = 0.8;     // Coefficient of driving force
const double Gamma   = 10.0;    // Coefficient of driving force
const double delta   = 0.10;    // Anisotropy in (0,1)
const double K       = 1.7;     // Referrers to latent heat (no-dimension)
const double T0      = 0.0;     // Initial temperature
const double Tm      = 1.0;     // Equilibrium temperature  (no-dimension)
const double ampl    = 0.01;    // Amplitude of noise
const int    seed    = 123;     // Random number seed

// Misc parameters
const double Radius        = 0.4;    // Initial radius of spherical grain
const double PhiPrecision  = 1.e-9;  // Phase-field cut off

void WriteToFile(const int tStep, double* field, string name);

inline int Index(int i, int j, int k)
{
    // Define index of the memory
    return (k * My + j) * Mx + i;
}

void InitializeSupercooledSphere(double* Phi, double* PhiDot, double* Temp,
        double* TempDot)
{
    // Initialization
    const double  x0 = Nx/2 * dx;
    const double  y0 = Ny/2 * dy;
    const double  z0 = Nz/2 * dz;

    #pragma omp parallel for schedule(auto) collapse(2)
    for (int i = BCELLS; i < Nx + BCELLS; i++)
    for (int j = BCELLS; j < Ny + BCELLS; j++)
    for (int k = BCELLS; k < Nz + BCELLS; k++)
    {
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
}

// Laplace operator
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


void CalcTimeStep(double* Phi, double* PhiDot, double* Temp, double* TempDot)
{
    // Calculate PhiDot and TempDot
    #pragma omp parallel for schedule(auto) collapse(2)
    for (int i = BCELLS; i < Nx + BCELLS; i++)
    for (int j = BCELLS; j < Ny + BCELLS; j++)
    for (int k = BCELLS; k < Nz + BCELLS; k++)
    {
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
            double noise = ampl * rand()/RAND_MAX;

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
}

void ApplyTimeStep(double* Phi, double* PhiDot, double* Temp, double* TempDot)
{
    #pragma omp parallel for schedule(auto) collapse(2)
    for (int i = BCELLS; i < Nx + BCELLS; i++)
    for (int j = BCELLS; j < Ny + BCELLS; j++)
    for (int k = BCELLS; k < Nz + BCELLS; k++)
    {
        int locIndex = Index(i,j,k);

        // Update phase field
        Phi    [locIndex] += dt * PhiDot [locIndex];
        PhiDot [locIndex]  = 0.0;
        Temp   [locIndex] += dt * TempDot[locIndex];
        TempDot[locIndex]  = 0.0;
    }
}

void SetBoundariesX(double* field)
{
    #pragma omp parallel for schedule(auto) collapse(1)
    for (int j = 0; j < My; j++)
    for (int k = 0; k < Mz; k++)
    {
        field[Index(0 ,j,k)] = field[Index(1   ,j,k)];
        field[Index(Nx,j,k)] = field[Index(Nx-1,j,k)];
    }
}

void SetBoundariesY(double* field)
{
    #pragma omp parallel for schedule(auto) collapse(1)
    for (int i = 0; i < Mx; i++)
    for (int k = 0; k < Mz; k++)
    {
        field[Index(i,0 ,k)] = field[Index(i,1   ,k)];
        field[Index(i,Ny,k)] = field[Index(i,Ny-1,k)];
    }
}

void SetBoundariesZ(double* field)
{
    #pragma omp parallel for schedule(auto) collapse(1)
    for (int i = 0; i < Mx; i++)
    for (int j = 0; j < My; j++)
    {
        field[Index(i,j,0 )] = field[Index(i,j,1   )];
        field[Index(i,j,Nz)] = field[Index(i,j,Nz-1)];
    }
}

void SetBoundaries(double* field)
{
    // Set Boundary conditions
    SetBoundariesX(field);
    SetBoundariesY(field);
    SetBoundariesZ(field);
}
int main()
{
    // Calculate memory size
    int numElements = Mx * My * Mz;
    size_t size = numElements * sizeof(double);

    // Define and allocate host memory
    double* Phi     = (double *)malloc(size);  memset(Phi,     0, size);
    double* PhiDot  = (double *)malloc(size);  memset(PhiDot,  0, size);
    double* Temp    = (double *)malloc(size);  memset(Temp,    0, size);
    double* TempDot = (double *)malloc(size);  memset(TempDot, 0, size);

    // Initialize Fields
    cout << "Initialized Data: " << Nx << "x" << Ny << "x" << Nz << endl;
    InitializeSupercooledSphere(Phi, PhiDot, Temp, PhiDot);

    // Initialize Random seed
    cout << "Initialize Random Seed.." << endl;
    srand(seed);

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
                WriteToFile(tStep, Phi,  "Phase-Field");
                WriteToFile(tStep, Temp, "Temperature");
            }
        }

        // Set boundary conditions
        SetBoundaries(Phi);
        SetBoundaries(Temp);

        // Calculate and apply time step
        CalcTimeStep(Phi,  PhiDot, Temp, TempDot);
        ApplyTimeStep(Phi, PhiDot, Temp, TempDot);
    }

    // Stop run time measurement
    gettimeofday(&end, NULL);
    double delta = ((end.tv_sec  - start.tv_sec) * 1000000u
            + end.tv_usec - start.tv_usec) / 1.e6;
    cout << "Calculation time for " << Nt << " time step: " << delta << " s" << endl;

    // Cleanup
    free(Phi);
    free(PhiDot);
    free(Temp);
    free(TempDot);
    return 0;
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
