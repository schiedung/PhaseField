#ifndef WRAPPER
#define WRAPPER

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

__device__ int index_i()  { return blockIdx.x * blockDim.x + threadIdx.x + BCELLS; }
__device__ int index_j()  { return blockIdx.y * blockDim.y + threadIdx.y + BCELLS; }
__device__ int index_k()  { return blockIdx.z * blockDim.z + threadIdx.z + BCELLS; }
__device__ int indexB_i() { return blockIdx.x * blockDim.x + threadIdx.x; }
__device__ int indexB_j() { return blockIdx.y * blockDim.y + threadIdx.y; }
__device__ int indexB_k() { return blockIdx.z * blockDim.z + threadIdx.z; }
__device__ int index() { return (index_k() * My + index_j()) * Mx + index_i(); }

__host__ __device__
inline int Index(int i, int j, int k) { return (k * My + j) * Mx + i; }

template<class function, typename... Args>
void InvokeKernel(function func, Args... args)
{
    const dim3 dimGrid( DIM_GRID_X  , DIM_GRID_Y  , DIM_GRID_Z );
    const dim3 dimBlock(DIM_BLOCK_X , DIM_BLOCK_Y , DIM_BLOCK_Z);

    func<<< dimGrid, dimBlock >>>(args...);
}

template<class function, typename... Args>
void InvokeKernelXY(function func, Args... args)
{
    const dim3 dimGridXY( 1 , 1 , My );
    const dim3 dimBlockXY( Mx , BCELLS , 1 );

    func<<< dimGridXY, dimBlockXY >>>(args...);
}

template<class function, typename... Args>
void InvokeKernelXZ(function func, Args... args)
{
    const dim3 dimGridXZ( 1 , 1 , Mx );
    const dim3 dimBlockXZ( Mz , BCELLS , 1 );

    func<<< dimGridXZ, dimBlockXZ >>>(args...);
}

template<class function, typename... Args>
 void InvokeKernelYZ(function func, Args... args)
{
    const dim3 dimGridYZ( 1 , 1 , Mz );
    const dim3 dimBlockYZ( My , BCELLS , 1 );
    func<<< dimGridYZ, dimBlockYZ >>>(args...);
}

__global__
void SetBoundariesYZ(float* field)
{
    const int j = indexB_i();
    const int b = indexB_j();
    const int k = indexB_k();

    field[Index(b     ,j,k)] = field[Index( 2*BCELLS-1-b,j,k)];
    field[Index(Mx-1-b,j,k)] = field[Index(Mx-2*BCELLS+b,j,k)];
}

__global__
void SetBoundariesXZ(float* field)
{
    const int i = indexB_i();
    const int b = indexB_j();
    const int k = indexB_k();

    field[Index(i,b     ,k)] = field[Index(i, 2*BCELLS-1-b,k)];
    field[Index(i,My-1-b,k)] = field[Index(i,My-2*BCELLS+b,k)];
}

__global__
void SetBoundariesXY(float* field)
{
    int i = indexB_i();
    int b = indexB_j();
    int j = indexB_k();

    // Apply mirror boundary conditions
    field[Index(i,j,b     )] = field[Index(i,j, 2*BCELLS-1-b)];
    field[Index(i,j,Mz-1-b)] = field[Index(i,j,Mz-2*BCELLS+b)];
}

void SetBoundaries(float* field)
{
    InvokeKernelXY(SetBoundariesXY,field);
    InvokeKernelXZ(SetBoundariesXZ,field);
    InvokeKernelYZ(SetBoundariesYZ,field);
}

//struct device_data_t
//{
//    float* data;
//    device_data_t(size_t size)
//    {
//        cudaMalloc((void**)&data,size);
//        cudaMemset(data,0.0,size);
//    }
//    ~device_data_t()
//    {
//        cudaFree(data);
//    }
//    float& [](size_t idx)
//    {
//        return[idx];
//    }
//};

#endif
