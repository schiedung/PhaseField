#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iostream>
//#include <omp.h>
#include <sstream>
#include <cstring>
#include <chrono>

// Define number of Boundary Cells
#define BCELLS 1

// Define computation domain size
const int Nx = 64;  // Domain size in x-direction
const int Ny = 64;  // Domain size in y-direction
const int Nz = 256;  // Domain size in z-direction

// Define memory size
const int Mx = Nx + 2 * BCELLS;  // Memory size in x-direction
const int My = Ny + 2 * BCELLS;  // Memory size in y-direction
const int Mz = Nz + 2 * BCELLS;  // Memory size in z-direction

// Define number of time steps
const int Nt     = 4000; // Number of time steps
const int tOut   = 100;  // Output distance in time steps
bool WriteToDisk = true;

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
const double delta   = 0.00;    // Anisotropy in (0,1)
const double K       = 1.7;     // Referrers to latent heat (no-dimension)
const double T0      = 0.0;     // Initial temperature
const double Tm      = 1.0;     // Equilibrium temperature  (no-dimension)

// Misc parameters
const double Radius        = 0.4;    // Initial radius of spherical grain
const double PhiPrecision  = 1.e-9;  // Phase-field cut off

void WriteToFile(const int tStep, double* field, std::string name)
{
    {
    std::stringstream filename;
    filename << "Out_" << name << "_"<< tStep << ".vtk";
    std::string FileName = filename.str();

    std::ofstream vtk_file(FileName.c_str());
    vtk_file << "# vtk DataFile Version 3.0\n";
    vtk_file << name << " \n";
    vtk_file << "ASCII\n";
    vtk_file << "DATASET RECTILINEAR_GRID\n";
    vtk_file << "DIMENSIONS " << Nx << " " << Ny << " " << Nz << "\n";
    vtk_file << "X_COORDINATES " << Nx << " double\n";
    for (int i = 0; i < Nx; i++) vtk_file << i << " ";
    vtk_file << "\n";
    vtk_file << "Y_COORDINATES " << Ny << " double\n";
    for (int j = 0; j < Ny; j++) vtk_file << j << " ";
    vtk_file << "\n";
    vtk_file << "Z_COORDINATES " << Nz << " double\n";
    for (int k = 0; k < Nz; k++) vtk_file << k << " ";
    vtk_file << "\n";
    vtk_file << "POINT_DATA " << Nx*Ny*Nz << "\n";

    vtk_file << "SCALARS " << name << " double 1\n";
    vtk_file << "LOOKUP_TABLE default\n";
    for (int k = BCELLS; k < Nz + BCELLS; ++k)
    for (int j = BCELLS; j < Ny + BCELLS; ++j)
    for (int i = BCELLS; i < Nx + BCELLS; ++i)
    {
        int locIndex = (k * My + j) * Mx + i;
        vtk_file << field[locIndex] << "\n";
    }
    vtk_file.close();
    }
}

int Index(int i, int j, int k)
{
    // Define index of the memory
    return (k * My + j) * Mx + i;
}

void InitializeSupercooledSphere(double* Phi, double* PhiDot, double* Temp,
        double* TempDot)
{
    // Initialization
    const double x0 = 0.0;//Nx/2 * dx;
    const double y0 = 0.0;//Ny/2 * dy;
    const double z0 = 0.0;//Nz/2 * dz;

    #pragma omp parallel for schedule(auto) collapse(2)
    for (int i = BCELLS; i < Nx + BCELLS; i++)
    for (int j = BCELLS; j < Ny + BCELLS; j++)
    for (int k = BCELLS; k < Nz + BCELLS; k++)
    {
        int locIndex = Index(i,j,k);
        // Initialize the phase Field
        double r = std::sqrt(std::pow(i*dx-x0,2) + std::pow(j*dy-y0,2) + std::pow(k*dz-z0,2));
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
        int    locIndex  = Index(i,j,k);
        double locPhi    = Phi[locIndex];
        double locPhiDot = 0.0;
        // Calculate driving force m
        if ((locPhi > 0.0) or  (locPhi < 1.0))
        {
            // Calculate gradient of Phi
            double gradX = (Phi[Index(i+1,j,k)] - Phi[Index(i-1,j,k)])/(2*dx);
            double gradY = (Phi[Index(i,j+1,k)] - Phi[Index(i,j-1,k)])/(2*dy);
            double gradZ = (Phi[Index(i,j,k+1)] - Phi[Index(i,j,k-1)])/(2*dz);
            double div   = std::pow(std::pow(gradX,2) + std::pow(gradY,2) + std::pow(gradZ,2),2);
            double theta;
            if ( div > 1.e-12)
                theta = (std::pow(gradX,4) + std::pow(gradY,4)+ std::pow(gradZ,4))/div;
            else
                theta = 0.0;
            double sigma = (1.-4.*delta*(1.-theta));

            // Calculate driving force am
            double m = (alpha/M_PI) * std::atan(Gamma*(Tm - Temp[locIndex])*sigma);

            // Add driving force to PhiDot
            locPhiDot += locPhi*(1.0 - locPhi)*(locPhi - 0.5 + m);
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

void SetBoundaryConditions(double* field)
{
    // Set Boundary conditions
    SetBoundariesX(field);
    SetBoundariesY(field);
    SetBoundariesZ(field);
}

void StartSimulation()
{
    // Calculate memory size
    size_t size = Mx * My * Mz * sizeof(double);

    // Define and allocate host memory
    double* Phi     = (double *)malloc(size);  std::memset(Phi,     0, size);
    double* PhiDot  = (double *)malloc(size);  std::memset(PhiDot,  0, size);
    double* Temp    = (double *)malloc(size);  std::memset(Temp,    0, size);
    double* TempDot = (double *)malloc(size);  std::memset(TempDot, 0, size);

    // Initialize Fields
    std::cout << "Initialized Data: " << Nx << "x" << Ny << "x" << Nz << "\n";
    InitializeSupercooledSphere(Phi, PhiDot, Temp, PhiDot);

    // Start run time measurement
    auto start = std::chrono::high_resolution_clock::now();

    // Start time loop
    for (int tStep = 0; tStep <= Nt; tStep++)
    {
        // Make Output if necessary
        if(tStep%tOut == 0)
        {
            std::cout << "Time step: " << tStep << "/" << Nt << "\n";
            if (WriteToDisk)
            {
                WriteToFile(tStep, Phi,  "Phase-Field");
                WriteToFile(tStep, Temp, "Temperature");
            }
        }

        SetBoundaryConditions(Phi);
        SetBoundaryConditions(Temp);

        // Calculate and apply time step
        CalcTimeStep(Phi,  PhiDot, Temp, TempDot);
        ApplyTimeStep(Phi, PhiDot, Temp, TempDot);
    }

    // Stop run time measurement
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Calculation time for " << Nt << " time step: " << duration.count() << " s\n" ;

    // Cleanup
    free(Phi);
    free(PhiDot);
    free(Temp);
    free(TempDot);
}

int main()
{
    try
    {
        StartSimulation();
    }
    catch (std::exception& ex)
    {
        std::cerr << "Exception: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
