#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numbers>
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
const int tOut   = 100;  // Output distance in time steps
bool WriteToDisk = true;

// Define grid spacing
const double dx = 1.0e-6; // Grid spacing in x-direction [m]
const double dy = 1.0e-6; // Grid spacing in y-direction [m]
const double dz = 1.0e-6; // Grid spacing in z-direction [m]

const double sigma    = 1.0;     // Interface energy coefficient
const double ieta     = 5.0;     // Interface thickness [1]
const double eta      = ieta*dx; // Interface thickness [m]

const double alpha    = 1.0;     // Thermal diffusivity [m^2/s]
const double rho      = 10.0;     // Density [kg/m^3]
const double L        = 1.0;     // Latent heat [J/kg]
const double cp       = 1.0;     // Specific heat [J/kgK]
const double cv       = rho*cp;  // volumetric heat capacity [J/m^3K]
const double T0       = -1.0;     // Initial temperature [K]
const double Tm       =  0.0;     // Equilibrium temperature [K]

const double pi = std::numbers::pi;
const double M0 = 4.0*pi*pi/(pi*pi-4.0)*alpha/cp/eta/L*cp;
const double dt_phase   = (dim > 1) ? ((dim > 2) ? (dx*dx/M0/6.0) : (dx*dx/M0/4.0)   ) : (dx*dx/M0/2.0);
const double dt_thermal = (dim > 1) ? ((dim > 2) ? (dx*dx/M0/6.0) : (dx*dx/alpha/4.0)) : (dx*dx/M0/2.0);
const double dt = std::min(dt_phase, dt_thermal); // Size of time step [s]

// Misc parameters
const double Radius  = 15*dx;    // Initial radius of spherical grain

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

    //df2_dz2 += field[Index(i,j,k+1)];
    //df2_dz2 -= field[Index(i,j,k  )] * 2.0;
    //df2_dz2 += field[Index(i,j,k-1)];
    //df2_dz2 /= dz*dz;

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
        const double dg        = cv*(Temp[locIndex] - Tm);
        const double locPhi    = Phi[locIndex];
        double locPhiDot = 0.0;
        locPhiDot += sigma*Laplace(Phi,i,j,k);
        locPhiDot -= sigma*pi*pi/eta/eta/2.0*(0.5-locPhi);
        locPhiDot -= pi/eta*std::sqrt(locPhi*(1.0-locPhi))*dg;
        locPhiDot *= M0;

        PhiDot [locIndex] += locPhiDot;
        TempDot[locIndex] += alpha*Laplace(Temp,i,j,k) + L/cp * locPhiDot;
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
    if (Nx > 0) SetBoundariesX(field);
    if (Ny > 0) SetBoundariesY(field);
    if (Nz > 0) SetBoundariesZ(field);
}

void StartSimulation()
{
    //TODO no standard exception handling
    feenableexcept(FE_DIVBYZERO | FE_INVALID);

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
