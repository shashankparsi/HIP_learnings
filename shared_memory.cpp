#include<stdio.h>
#include<stdlib.h>
#include<hip/hip_runtime.h>

__global__ void vectoradd(const float *a,const float *b,float *c,int N)
{
        //define shared memory
        __shared__ float SA[256];
        __shared__ float SB[256];
        int tid=blockIdx.x * blockDim.x + threadIdx.x;
        //load data in to memory
        if(tid<N) {
        SA[threadIdx.x] = a[tid];
        SB[threadIdx.x] = b[tid];
        }
        __syncthreads(); //synchronize threads
        if(tid<N)
        {
                c[tid] = SA[threadIdx.x] + SB[threadIdx.x];
        }
}

int main()
{
        int N=1000000000;
        //Allocate Host memory
        float *h_a=(float*)malloc(sizeof(float)*N);
        float *h_b=(float*)malloc(sizeof(float)*N);
        float *h_c=(float*)malloc(sizeof(float)*N);
        //initialize host arrays
        for(int i=1;i<N;i++) {
                h_a[i] = (float)i;
                h_b[i] = (float)(i*2);
        }
        //Allocate Device memory
        float *d_a,*d_b,*d_c;
        hipMalloc((void**)&d_a,sizeof(float)*N);
        hipMalloc((void**)&d_b,sizeof(float)*N);
        hipMalloc((void**)&d_c,sizeof(float)*N);
        //copy data from Host to Device
        hipMemcpy(d_a,h_a,sizeof(float)*N,hipMemcpyHostToDevice);
        hipMemcpy(d_b,h_b,sizeof(float)*N,hipMemcpyHostToDevice);
        //launching kernel
        int threadsperblock=256;
        int blockspergrid=(N+threadsperblock-1)/threadsperblock;
        vectoradd<<<blockspergrid,threadsperblock>>>(d_a,d_b,d_c,N);
        hipDeviceSynchronize();
        //copy results back from device to host
        hipMemcpy(h_c,d_c,sizeof(float)*N,hipMemcpyDeviceToHost);
        printf("The first index of resultant array is : %0.2f\n",h_c[1]);
        free(h_a);
        free(h_b);
        free(h_c);
        hipFree(d_a);
        hipFree(d_b);
        hipFree(d_c);
        return 0;
}

        

