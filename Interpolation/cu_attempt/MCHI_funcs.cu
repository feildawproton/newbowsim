#include <stdlib.h>

__global__ void _interpolate_Ts(const unsigned N_Ts, const float *Ts, const float T_scale,
                                const unsigned N, const unsigned numThreads,
                                float *T_interpolated, unsigned *Ts_k_indices)
{
    //grid stride if needed
    unsigned gindx = threadIdx.x + blockDim.x * blockIdx.x;	
    for(unsigned n = gindx; n < N; n += numThreads)
	{
		float T_intrpltd_val = ((n / (N - 1)) * T_scale) + Ts[0];

        unsigned k = 0;
        for(unsigned i = 0; i < N_Ts; i++)
        {
            if(T_intrpltd_val > Ts[i])
            {
                k = i;
            }
        }

        Ts_k_indices[n]     = k;
        T_interpolated[n]   = T_intrpltd_val;
	}
}

//rellying on on caller to provide a lot of 
//N is the size of the desired interpolation
extern "C" {
void interpolate_Ts(const unsigned N_Ts, const float *Ts, 
                    const unsigned N, float *T_interpolated, unsigned *Ts_k_indices)
{

    float *Ts_dev, *T_interpolated_dev;
    unsigned *Ts_k_indices_dev;

    cudaError_t status;
    status = cudaMalloc((void**)&Ts_dev, N_Ts * sizeof(float));
    status = cudaMalloc((void**)&T_interpolated_dev, N * sizeof(float));
    status = cudaMalloc((void**)&Ts_k_indices_dev, N * sizeof(unsigned));

    cudaMemcpy(Ts_dev, Ts, N_Ts*sizeof(float), cudaMemcpyHostToDevice);

    // get device properties
    // maybe not useful for this trivial workload
	int deviceID;							                        //device id and properties
	cudaGetDevice(&deviceID);
	
	cudaDeviceProp props;						                    //get properties to make best use of device
	cudaGetDeviceProperties(&props, deviceID);

	unsigned ThreadsPerBlock    = props.warpSize * 4;		        //threads per block should be some multiple warpsize or just set it to maxThreadsPerBlock
	unsigned BlocksPerGrid      = props.multiProcessorCount * 2;	//blocks per grid should be some multiple of the number of streaming multiprocessors
	unsigned numThreads         = BlocksPerGrid * ThreadsPerBlock;

    float T_scale = Ts[N_Ts - 1] - Ts[0];
    _interpolate_Ts<<<BlocksPerGrid, ThreadsPerBlock>>>(N_Ts, Ts_dev, T_scale,
                                                        N, numThreads,
                                                        T_interpolated_dev, Ts_k_indices_dev);
    
    cudaDeviceSynchronize();

    cudaMemcpy(T_interpolated, T_interpolated_dev, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Ts_k_indices, Ts_k_indices_dev, N * sizeof(unsigned), cudaMemcpyDeviceToHost);

    cudaFree(Ts_k_indices_dev);
    cudaFree(T_interpolated_dev);
    cudaFree(Ts_dev);

    status = cudaGetLastError();
}
}