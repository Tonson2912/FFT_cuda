
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<cufft.h>

#define CHECK_CUDA(errorInfo,cudaSuccInfo){ \
	if((errorInfo)!=cudaSuccInfo){  \
		fprintf(stderr,"CUDA error in line %d of file %s\:%s\n",__LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()));\
		system("pause");\
		exit(-1);\
	}\
} \

#if _DEBUG
#define RUNTIME_CUDA_ERROR(errorInfo) CHECK_CUDA(errorInfo,cudaSuccess)		   //runtime API函数
#define CUFFT_CUDA_ERROR(errorInfo) CHECK_CUDA(errorInfo,CUFFT_SUCCESS)        //cuBlas API函数
#else
#define RUNTIME_CUDA_ERROR(errorInfo) errorInfo
#define CUFFT_CUDA_ERROR(errorInfo) errorInfo
#endif

//CUDA释放宏
#define CUDA_FREE(d_ptr){ \
	if (d_ptr != nullptr)RUNTIME_CUDA_ERROR(cudaFree(d_ptr)); d_ptr = nullptr;\
}\

//读GPU数据
#define READ_CUDA_DATA(d_ptr,type,size){\
	type* check=new type[size];\
	cudaMemcpy(check,d_ptr,sizeof(type)*size,cudaMemcpyDeviceToHost);\
	for(int i=0;i<size;++i){\
		cout << i + 1 << "  :" << check[i] << endl;\
	}\
}\

#define READ_CUDA_DATA_COMP(d_ptr,size){\
	cufftDoubleComplex* check=new cufftDoubleComplex[size];\
	cudaMemcpy(check,d_ptr,sizeof(cufftDoubleComplex)*size,cudaMemcpyDeviceToHost);\
	for(int i=0;i<size;++i){\
		cout << i + 1 << "  x:" << check[i].x<<" y:"<<check[i].y << endl;\
	}\
}\

//核函数
template<typename T>
__global__ void cufft_R2C_kernel(T *d_input, cufftDoubleComplex *d_complex,unsigned int length)
{
	unsigned int ix = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y*blockDim.y;
	unsigned int i = ix + iy * blockDim.x*gridDim.x;
	if (i < length)
	{
		d_complex[i].x = (double)d_input[i];
		d_complex[i].y = 0;
	}
}

template<typename T>
__device__ void comp_abs(cufftDoubleComplex &comp, T &real)
{
	real = (T)sqrt((comp.x*comp.x + comp.y*comp.y));
}

template<typename T>
__global__ void cufft_comp_asb_kernel(cufftDoubleComplex *d_complex, T *d_output, unsigned int length)
{
	unsigned int ix = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y*blockDim.y;
	unsigned int i = ix + iy * blockDim.x*gridDim.x;
	if (i < length)
	{
		comp_abs(d_complex[i], d_output[i]);
	}

}
