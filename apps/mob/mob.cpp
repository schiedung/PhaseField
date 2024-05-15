#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numbers>
#include <filesystem>
#include <sstream>
#include <cfenv>
//#include <omp.h>

// Define number of Boundary Cells
#define BCELLS 1

// Define computation domain size
const int Nx = 64;  // Domain size in x-direction
const int Ny = 64;  // Domain size in y-direction
const int Nz = 1;   // Domain size in z-direction
const int dim = (Nx > 1) + (Ny > 1) + (Nz > 1);

// Define memory size
const int Mx = Nx + 2 * BCELLS;  // Memory size in x-direction
const int My = Ny + 2 * BCELLS;  // Memory size in y-direction
const int Mz = Nz + 2 * BCELLS;  // Memory size in z-direction

// Define number of time steps
const int Nt     = 4000; // Number of time steps
const int tOut   = 10;   // Output distance in time steps
bool WriteToDisk = true;

// Define grid spacing
const double dx = 1.0; // Grid spacing in x-direction [m]
const double dy = 1.0; // Grid spacing in y-direction [m]
const double dz = 1.0; // Grid spacing in z-direction [m]

const double sigma    = 1.0;     // Interface energy coefficient
const double ieta     = 5.0;     // Interface thickness [1]
const double eta      = ieta*dx; // Interface thickness [m]

const double alpha    =  1.0;     // Thermal diffusivity [m^2/s]
const double rho      =  1.0;     // Density [kg/m^3]
const double L        =  1.0;     // Latent heat [J/kg]
const double T0       = -1.0;     // Initial undercooling [K]
const double Tm       =  0.0;     // Equilibrium temperature [K]

const double pi = std::numbers::pi;
const double M0 = 4.0*pi*pi/(pi*pi-4.0)*alpha/L/rho/eta;
const double dt_phase   = (dim > 1) ? ((dim > 2) ? (dx*dx/M0*sigma/6.0) : (dx*dx/M0*sigma/4.0)) : (dx*dx/M0*sigma/2.0);
const double dt_thermal = (dim > 1) ? ((dim > 2) ? (dx*dx/alpha/6.0)    : (dx*dx/alpha/4.0))    : (dx*dx/alpha/2.0);
const double dt = std::min(dt_phase, dt_thermal); // Size of time step [s]

// Misc parameters
const double Radius  = 15*dx;    // Initial radius of spherical grain

namespace fs = std::filesystem;

int Index(int i, int j, int k)
{
    // Define index of the memory
    return (k * My + j) * Mx + i;
}

void WriteToFile(const int tStep, double* field, std::string name)
{
    // Create output directory
    fs::path output_dir = "output";
    if (!fs::exists(output_dir)) fs::create_directory(output_dir);

    // Create filename
    std::string filename = name + "_" + std::to_string(tStep) + ".vtk";
    fs::path file_path = output_dir / filename;

    // Write to VTK file
    std::ofstream vtk_file(file_path);
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
        vtk_file << field[Index(i,j,k)] << "\n";
    }
    vtk_file.close();
}



void InitializeSupercooledSphere(double* Phi, double* PhiDot, double* Temp,
        double* TempDot)
{
    // Initialization
    const double x0 = Nx/2 * dx;
    const double y0 = Ny/2 * dy;
    const double z0 = Nz/2 * dz;

    #pragma omp parallel for schedule(auto) collapse(2)
    for (int i = BCELLS; i < Nx + BCELLS; i++)
    for (int j = BCELLS; j < Ny + BCELLS; j++)
    for (int k = BCELLS; k < Nz + BCELLS; k++)
    {
        int locIndex = Index(i,j,k);
        // Initialize the phase Field
        //double r = std::sqrt(std::pow(i*dx-x0,2) + std::pow(j*dy-y0,2) + std::pow(k*dz-z0,2));
        //if ( r < Radius)
        if ( i*dx < x0)
        {
            Phi    [locIndex] = 1.0;
            Temp   [locIndex] = 0.0;
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
        const int    locIndex  = Index(i,j,k);
        const double pi        = std::numbers::pi;
        const double dg        = L*rho*Temp[locIndex];
        const double locPhi    = Phi[locIndex];
        double locPhiDot = 0.0;
        locPhiDot += sigma*Laplace(Phi,i,j,k);
        locPhiDot -= sigma*pi*pi/eta/eta/2.0*(0.5-locPhi);
        locPhiDot -= pi/eta*std::sqrt(locPhi*(1.0-locPhi))*dg;
        locPhiDot *= M0;

        PhiDot [locIndex] += locPhiDot;
        TempDot[locIndex] += alpha*Laplace(Temp,i,j,k) + locPhiDot; //TODO: account for limited phase field
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

        // Limit phase-field values to [0,1]
        if      (Phi[locIndex] < 0.0) Phi[locIndex] = 0.0;
        else if (Phi[locIndex] > 1.0) Phi[locIndex] = 1.0;
    }
}

void SetBoundariesX(double* field)
{
    #pragma omp parallel for schedule(auto) collapse(1)
    for (int j = 0; j < My; j++)
    for (int k = 0; k < Mz; k++)
    {
        field[Index(0   ,j,k)] = field[Index(1   ,j,k)];
        field[Index(Mx-1,j,k)] = field[Index(Mx-2,j,k)];
    }
}

void SetBoundariesY(double* field)
{
    #pragma omp parallel for schedule(auto) collapse(1)
    for (int i = 0; i < Mx; i++)
    for (int k = 0; k < Mz; k++)
    {
        field[Index(i,0   ,k)] = field[Index(i,1   ,k)];
        field[Index(i,My-1,k)] = field[Index(i,My-2,k)];
    }
}

void SetBoundariesZ(double* field)
{
    #pragma omp parallel for schedule(auto) collapse(1)
    for (int i = 0; i < Mx; i++)
    for (int j = 0; j < My; j++)
    {
        field[Index(i,j,0   )] = field[Index(i,j,1   )];
        field[Index(i,j,Mz-1)] = field[Index(i,j,Mz-2)];
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
    //TODO no standard exception handling
    feenableexcept(FE_DIVBYZERO | FE_INVALID);

    // Calculate memory size
    size_t size = Mx * My * Mz * sizeof(double);

    // Define and allocate host memory
    double* Phi     = new double [size];
    double* PhiDot  = new double [size];
    double* Temp    = new double [size];
    double* TempDot = new double [size];

    // Initialize Fields
    std::cout << "Initialized Data: " << Nx << "x" << Ny << "x" << Nz << "\n";
    InitializeSupercooledSphere(Phi, PhiDot, Temp, PhiDot);
    SetBoundaryConditions(Phi);
    SetBoundaryConditions(Temp);

    // Start run time measurement
    auto start = std::chrono::high_resolution_clock::now();

    std::cout << "Start Simulation\n";
    std::cout << "max dt phase:   " << dt_phase   << "\n";
    std::cout << "max dt thermal: " << dt_thermal << "\n";

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
    delete[] Phi;
    delete[] PhiDot;
    delete[] Temp;
    delete[] TempDot;
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
