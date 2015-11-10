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
//#include <omp.h>
#include <random>
#include <sstream>
#include <string.h>
#include <sys/time.h> 
using namespace std;

// Define number of Boundary Cells
#define BCELLS 1

// Define computation domain size
const int Nx = 256;  // Domain size in x-direction
const int Ny = 1;  // Domain size in y-direction
const int Nz = 1;    // Domain size in z-direction

// Define memory size
const int Mx = Nx + 2 * BCELLS;  // Memory size in x-direction
const int My = Ny + 2 * BCELLS;  // Memory size in y-direction
const int Mz = Nz + 2 * BCELLS;  // Memory size in z-direction

// Define number of time steps
const int Nt                 = 100000000; // Number of time steps
      int InterfacePosOut    = 20;
      int InterfacePosOutInc = 5;
bool WriteToDisk = true;

// Define grid spacing
double dt = 1.0e-13;  // Size of time step [s]
double dx = 1.0e-7;  // Grid spacing in x-direction [m]
double dy = 1.0e-7;  // Grid spacing in y-direction [m]
double dz = 1.0e-7;  // Grid spacing in z-direction [m]

// Some material parameters (aluminium)
#define M   2.698150e-2  // Molar mass [Kg/mol]
#define cp  3.169866e+1  // Heat capacity[J/(mol K)]
#define dH  1.069768e+4  // Latent heat of fusion [J/mol]
#define rho 2.702000e+3  // Density Al [Kg/m3]
#define Tm  933.47       // Melting temperature

// Physical parameters
const double alpha0 = 3.52e-5;        // Thermal diffusivity [m2/s]
const double alpha1 = 6.80e-5;        // Thermal diffusivity [m2/s]
const double dS     = rho*dH/(Tm*M);  // Entropy of fusion [J/m3 K]
const double delta  = 0.20;           // Anisotropy in (0,1)
      double eta    = 10 * dx;        // Interface width [m]
const double kappa  = M*dH/(cp);      // Latent heat parameter [K]
const double mu     = 1.0e-5;         // Interface mobility [m2/J s]
const double sigma0 = 0.55;           // Interface energy [J/m2]

// Misc parameters
const double Precision = 1.e-6;     // Phase-field cut off
const double ampl      = 0.01;      // Amplitude of noise
      double Radius    = 0.1*Nx*dx;// Initial radius of spherical grain

void WriteToFile(const int tStep, double* field, string name);
std::default_random_engine generator;
std::normal_distribution<double> distribution(0.0,0.5);

// Calculates the 1d memory index at certain grind point (i,j,k)
inline int Index(int i, int j, int k)
{
    // Define index of the memory
    return (k * My + j) * Mx + i;
}

double temperature_steady_state(double x, double v, double Ts)
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
        Tx -= 10*alpha0/v * (exp(-v*x/alpha0) -1.0);
    }
    else
    {
        Tx += A * exp(-v * x / alpha0) * (exp(v * eta / alpha0) + 1.0) - 2.0 ;
        Tx *= 0.5 * kappa ;
        Tx += Ts;
        Tx -= 10*alpha0/v * (exp(-v*x/alpha0) -1.0);
    }
    return Tx;
}

void InitializePlanarFront(double* Phi, double* PhiDot, double* Temp,
        double* TempDot)
{
    #pragma omp parallel for schedule(auto) collapse(2)
    for (int i = BCELLS; i < Nx + BCELLS; i++)
    for (int j = BCELLS; j < Ny + BCELLS; j++)
    for (int k = BCELLS; k < Nz + BCELLS; k++)
    {
//        double velocity = 1000;
        int locIndex = Index(i,j,k);
        // Initialize the phase Field
        double x = i*dx - Radius;
        if (x < 0.0)
        {
            Phi [locIndex] = 1.0;
            Temp[locIndex] = Tm;
        }
        else if (x < eta)
        {
            Phi [locIndex] = 0.5 + 0.5 * cos(M_PI / eta * x);
            Temp[locIndex] = Tm-kappa;//temperature_steady_state(x,velocity);
        }
        else
        {
            Phi [locIndex] = 0.0;
            Temp[locIndex] = Tm-kappa;//temperature_steady_state(x,velocity);
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

// derivative operator
double df_dx(double* field, int i, int j, int k)
{
    return (field[Index(i+1,j,k)] - field[Index(i-1,j,k)])/(2*dx);
}

double df_dy(double* field, int i, int j, int k)
{
    return (field[Index(i,j+1,k)] - field[Index(i,j-1,k)])/(2*dy);
}

double df_dz(double* field, int i, int j, int k)
{
    return (field[Index(i,j,k+1)] - field[Index(i,j,k-1)])/(2*dz);
}

// Calculates the thermal diffusivity inside the interface
inline double alpha(double Phi)
{
    return Phi * alpha1 + (1.0 - Phi) * alpha0;
}

// Calculates the interface energy
double sigma(double *Phi, int i, int j, int k)
{
    if (delta != 0.0)
    {
        double gradX = df_dx(Phi,i,j,k);
        double gradY = df_dy(Phi,i,j,k);
        double gradZ = df_dz(Phi,i,j,k);

        double NormGrad = sqrt(pow(gradX,2) + pow(gradY,2) + pow(gradZ,2));

        if (NormGrad >= Precision)
        {
            double NormX = gradX/NormGrad;
            double NormY = gradY/NormGrad;
            double NormZ = gradZ/NormGrad;
            double gamma = pow(NormX, 4) + pow(NormY, 4) + pow(NormZ, 4);
            double sigma = sigma0*(1.0 + delta * (1.5 - 2.5*(gamma)));

            return sigma;
        }
        else return sigma0;
    }
    else return sigma0;
}

double CalcVariationDerivativePhi(double* Phi, double Temp, int i, int j, int k)
{
    int    locIndex   = Index(i,j,k);
    double locPhi     = Phi[locIndex];
    double locPhiDot  = 0.0;
    // Calculate variation derivative of the DO-potential
    if ((locPhi < 0.0) or (locPhi > 1.0))
    {
        locPhiDot  -= sigma0 * pow(M_PI/eta,2) * (locPhi - 0.5);
    }
    else if ((locPhi > 0) and (locPhi < 1))
    {
        // Calculate and apply the driving force
        double qPhi = locPhi * (1.0 - locPhi);
        locPhiDot  -= M_PI/(eta) * sqrt(qPhi) * dS * (Temp - Tm);
        locPhiDot  += sigma(Phi,i,j,k) * pow(M_PI/eta,2) * (locPhi - 0.5);
    }
    
    // Calculate Laplacian of the phase field
    locPhiDot  += sigma(Phi,i,j,k)  * Laplace(Phi,i,j,k);
    
    return locPhiDot;
}

double mu_effecitve(double Phi, double PhiDot, double GradPhiNormal, double Ts)
{
    //double v = PhiDot * GradPhiNormal;
    //double x = eta/M_PI * acos(2.0 * Phi - 1.0);
    //double A = (Ts - temperature_steady_state(x,v,Ts))/v;
    //return mu/(1-A); 
    return mu;
}

void CalcPhiDot(double* Phi, double* PhiDot, double* Temp)
{
    #pragma omp parallel for schedule(auto) collapse(2)
    for (int i = BCELLS; i < Nx + BCELLS; i++)
    for (int j = BCELLS; j < Ny + BCELLS; j++)
    for (int k = BCELLS; k < Nz + BCELLS; k++)
    {
        int    locIndex  = Index(i,j,k);
        double dF_dPhi   = CalcVariationDerivativePhi(Phi,Temp[locIndex],i,j,k);
        double locPhi    = Phi[locIndex];
        double locPhiDot = mu * dF_dPhi;  // The normal local Phi_Dot

        if ((locPhi > 0) and (locPhi < 1))
        {
            double locGradPhiNormal = df_dx(Phi,i,j,k);
            double locPhi           = Phi [locIndex];

            // TODO calculate local solid Temperature
            double locTs = Tm;

            double locPhiDotA = -1.0/dt;
            double locPhiDotB =  1.0/dt;

            double fa = mu_effecitve(locPhi, locPhiDotA, locGradPhiNormal, locTs) * dF_dPhi - locPhiDotA;
            double fb = mu_effecitve(locPhi, locPhiDotB, locGradPhiNormal, locTs) * dF_dPhi - locPhiDotB;
            double fc = mu_effecitve(locPhi, locPhiDot , locGradPhiNormal, locTs) * dF_dPhi - locPhiDot;

            // Start iterative algorithm
            const double epsilonF     = 1.0e-10;
            const double epsilonP     = 1.0e-10;
            const int    MaxIteration = 100000;
            int  iteration = 0;
            bool iterate   = (fabs(fc) > epsilonF);
            while (iterate)
            {
                  cout << fabs(fc) << endl;
            //    if      (((fa > 0) and (fc > 0)) or((fa < 0) and (fc < 0))) locPhiDotA = locPhiDot;
            //    else if (((fb > 0) and (fc > 0)) or((fb < 0) and (fc < 0))) locPhiDotB = locPhiDot;
            //    else break;

            //    //locPhiDot = (locPhiDotA * fb - locPhiDotB * fa)/(fb - fa);
            //    locPhiDot = 0.5 * (locPhiDotB + locPhiDotA);
            //    fc = mu_effecitve(locPhi, locPhiDot , locGradPhiNormal, locTs) * dF_dPhi - locPhiDot;

                iteration ++;
                if (iteration > MaxIteration) break;
                iterate = (fabs(fc) > epsilonF) xor (fabs(locPhiDotA-locPhiDotB) > epsilonP);
            }
        }

        // Limit phase field
        //double newPhi = Phi[locIndex] + dt * locPhiDot;
        //if (newPhi > 1.0) locPhiDot -= (newPhi-1.0)/dt;
        //if (newPhi < 0.0) locPhiDot -=  newPhi/dt;

        // Calculate and write time derivatives into the device memory
        PhiDot [locIndex] += locPhiDot;
    }
}

void CalcTempDot(double* Phi, double* PhiDot, double* Temp, double* TempDot)
{
    #pragma omp parallel for schedule(auto) collapse(2)
    for (int i = BCELLS; i < Nx + BCELLS; i++)
    for (int j = BCELLS; j < Ny + BCELLS; j++)
    for (int k = BCELLS; k < Nz + BCELLS; k++)
    {
        // Calculate time derivative of the temperature
        int    locIndex   = Index(i,j,k);
        double locTempDot = 0.0;
        locTempDot += alpha(Phi[locIndex]) * Laplace(Temp,i,j,k);
        locTempDot += kappa * PhiDot[locIndex];

        // Write time derivatives into the device memory
        TempDot[locIndex] += locTempDot;
    }
}


void ApplyTimeStep(double* field, double* fieldDot)
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

void SetBoundariesX(double* field)
{
    #pragma omp parallel for schedule(auto) collapse(1)
    for (int j = 0; j < My;     j++)
    for (int k = 0; k < Mz;     k++)
    {
        double df0 = field[Index(1,j,k)];
        double df1 = 0.0;
        double df2 = 0.0;
        double df3 = 0.0;
        //df1 -=     field[Index(1,j,k)];
        //df1 +=     field[Index(2,j,k)];

        df1 -= 137./60. * field[Index(1,j,k)];
        df1 +=       5. * field[Index(2,j,k)];
        df1 -=       5. * field[Index(3,j,k)];
        df1 +=   10./3. * field[Index(4,j,k)];
        df1 -=    5./4. * field[Index(5,j,k)];
        df1 +=    1./5. * field[Index(6,j,k)];

        df2 +=     field[Index(1,j,k)];
        df2 -= 2 * field[Index(2,j,k)];
        df2 +=     field[Index(3,j,k)];

        df3 -=     field[Index(1,j,k)];
        df3 += 3 * field[Index(2,j,k)];
        df3 -= 3 * field[Index(3,j,k)];
        df3 +=     field[Index(4,j,k)];

        // Make Taylor expansion
        field[Index(0,j,k)] = df0 - df1 + 0.5*df2 - 1./6.*df3;
    }
    #pragma omp parallel for schedule(auto) collapse(1)
    for (int j = 0; j < My;     j++)
    for (int k = 0; k < Mz;     k++)
    {
        double df0 = field[Index(Mx-2,j,k)];
        double df1 = 0.0;
        double df2 = 0.0;
        double df3 = 0.0;
        //df1 -=     field[Index(Mx-3,j,k)];
        //df1 +=     field[Index(Mx-2,j,k)];

        df1 -= 137./60. * field[Index(Mx-7,j,k)];
        df1 +=       5. * field[Index(Mx-6,j,k)];
        df1 -=       5. * field[Index(Mx-5,j,k)];
        df1 +=   10./3. * field[Index(Mx-4,j,k)];
        df1 -=    5./4. * field[Index(Mx-3,j,k)];
        df1 +=    1./5. * field[Index(Mx-2,j,k)];

        df2 +=     field[Index(Mx-4,j,k)];
        df2 -= 2 * field[Index(Mx-3,j,k)];
        df2 +=     field[Index(Mx-2,j,k)];

        df3 -=     field[Index(Mx-5,j,k)];
        df3 += 3 * field[Index(Mx-4,j,k)];
        df3 -= 3 * field[Index(Mx-3,j,k)];
        df3 +=     field[Index(Mx-2,j,k)];

        // Make Taylor expansion
        field[Index(Mx-1,j,k)] = df0 + df1 + 0.5*df2 + 1./6.*df3;
    }

}

void SetBoundariesY(double* field)
{
    #pragma omp parallel for schedule(auto) collapse(1)
    for (int i = 0; i < Mx;     i++)
    for (int k = 0; k < Mz;     k++)
    for (int b = 0; b < BCELLS; b++)
    {
        field[Index(i,b     ,k)] = field[Index(i,BCELLS+Ny-1-b,k)];
        field[Index(i,My-1-b,k)] = field[Index(i,BCELLS   -1+b,k)];
    }
}

void SetBoundariesZ(double* field)
{
    #pragma omp parallel for schedule(auto) collapse(1)
    for (int i = 0; i < Mx;     i++)
    for (int j = 0; j < My;     j++)
    for (int b = 0; b < BCELLS; b++)
    {
        field[Index(i,j,b     )] = field[Index(i,j,BCELLS+Nz-1-b)];
        field[Index(i,j,Mz-1-b)] = field[Index(i,j,BCELLS   -1+b)];
    }
}

void SetBoundaries(double* field)
{
    // Set Boundary conditions
    SetBoundariesX(field);
    SetBoundariesY(field);
    SetBoundariesZ(field);
}

int MeasureInterfaceWidth(double* Phi)
{
    // Measure interface width
    int Phi1 = 0; int Phi0 = 0;
    for (int i = 0; i < Mx; i++)
    if (Phi[Index(i,My/2,Mz/2)] <  1.0 - Precision)
    {
        Phi1 = i;
        break;
    }
    for (int i = Phi1; i < Mx; i++)
    if (Phi[Index(i,My/2,Mz/2)] < Precision)
    {
        Phi0 = i;
        break;
    }
    return Phi0 - Phi1;
}

int MeasureInterfacePosition(double* Phi)
{
    int pos = 0;
    for (int i = BCELLS; i < Mx; i++)
    if (Phi[Index(i,My/2,Mz/2)] < 0.5 )
    {
        pos = i;
        break;
    }
    return pos;
}

int main(int argc, char* argv[])
{
    // Read input parameters
    if (argc > 3) dt = stod(argv[3]);
    if (argc > 2)
    {
        dx = stod(argv[2]);
        dy = stod(argv[2]);
        dy = stod(argv[2]);
    }
    if (argc > 1) eta = stoi(argv[1]) * dx;
    
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
    cout << "Initialized eta:  " << int(eta/dx) << endl;
    cout << "Initialized dt:   " << dt << endl;
    cout << "Initialized dx:   " << dx << endl << endl;
    InitializePlanarFront(Phi, PhiDot, Temp, PhiDot);
    WriteToFile(0, Phi,  "PhaseField");
    WriteToFile(0, Temp, "Temperature");

    // Start run time measurement
    struct timeval start, end;
    gettimeofday(&start, NULL);

    // Initialize speed measurement
    double xStart = MeasureInterfacePosition(Phi);
    int    tStart = 0;

    // Start time loop
    cout << "Start time loop.." << endl;
    for (int tStep = 0; tStep <= Nt; tStep++)
    {
        const int InterfacePos = MeasureInterfacePosition(Phi)*100/Mx;
        if (InterfacePos == InterfacePosOut)
        {
            // Set new output position
            InterfacePosOut += InterfacePosOutInc;

            // Measure interface-width
            double InterfaceWidth = MeasureInterfaceWidth(Phi);

            // Measure solidification speed
            double Speed = 0.0;
            double DeltaT = (tStep - tStart) * dt;
            double DeltaX = abs(InterfacePos - xStart) * dx;
            Speed  = DeltaX/DeltaT;
            xStart = InterfacePos;
            tStart = tStep;

            // Make output to screen
            cout << "[ " << InterfacePos << "% ]"
                 << "  Eta: " << InterfaceWidth 
                 << "  Speed: " << Speed << " m/s" << endl;

            if (WriteToDisk)
            {
                WriteToFile(InterfacePos, Phi,  "PhaseField");
                WriteToFile(InterfacePos, Temp, "Temperature");
            }

            // Stop simulation if the end is reached
            if (InterfacePosOut >= 85) break; 
        }

        // Set boundary conditions
        SetBoundaries(Phi);
        SetBoundaries(Temp);

        // Calculate time-step
        CalcPhiDot( Phi, PhiDot, Temp);
        CalcTempDot(Phi, PhiDot, Temp, TempDot);

        // Apply time-step
        ApplyTimeStep(Phi, PhiDot);
        ApplyTimeStep(Temp, TempDot);
    }

    // Stop run time measurement
    gettimeofday(&end, NULL);
    double simTime = ((end.tv_sec  - start.tv_sec) * 1000000u
            + end.tv_usec - start.tv_usec) / 1.e6;
    cout << "Calculation time: " << simTime << " s" << endl;

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
        int locIndex = Index(i,j,k);
        vtk_file << field[locIndex] << endl;
    }
    vtk_file.close();
    }
}
