#ifndef VTK
#define VTK

#include <iostream>

void WriteToFile(const int tStep, float* field, std::string name)
{
    std::stringstream filename;
    filename << "Out_" << name << "_"<< tStep << ".vtk";
    std::string FileName = filename.str();

    std::ofstream vtk_file(FileName.c_str());
    vtk_file << "# vtk DataFile Version 3.0\n";
    vtk_file << name << "\n";
    vtk_file << "ASCII\n";
    vtk_file << "DATASET RECTILINEAR_GRID\n";
    vtk_file << "DIMENSIONS " << Nx << " " << Ny << " " << Nz << "\n";
    vtk_file << "X_COORDINATES " << Nx << " float\n";
    for (int i = 0; i < Nx; i++) vtk_file << i << " \n";
    vtk_file << "Y_COORDINATES " << Ny << " float\n";
    for (int j = 0; j < Ny; j++) vtk_file << j << " \n";
    vtk_file << "Z_COORDINATES " << Nz << " float\n";
    for (int k = 0; k < Nz; k++) vtk_file << k << " \n";
    vtk_file << "POINT_DATA " << Nx*Ny*Nz << "\n";

    vtk_file << "SCALARS " << name << " float 1\n";
    vtk_file << "LOOKUP_TABLE default\n";
    for (int k = BCELLS; k < Nz + BCELLS; ++k)
    for (int j = BCELLS; j < Ny + BCELLS; ++j)
    for (int i = BCELLS; i < Nx + BCELLS; ++i)
    {
        int locIndex = Index(i,j,k);
        vtk_file << field[locIndex] << "\n";
    }
    vtk_file.close();
}

#endif
