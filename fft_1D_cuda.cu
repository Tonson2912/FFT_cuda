#include"func_header.h"
#include"fft_1D_cuda.cuh"


template<typename T>
class fft_1D_cuda :public func_header<T>
{
public:
	fft_1D_cuda(T *dataline, unsigned int length,unsigned int batch) :_length(length), _dataline(dataline),_batch(batch) {
		cuda_init();
	}
	void get_dft_result(T* dft_result) 
	{
		compute(dft_result);
	}
	void reset()
	{
		_dataline = nullptr;
		_length = 0;
	}
	void del_object() {
		delete this;
	}

private:
	~fft_1D_cuda()
	{
		cuda_free();
		cufftDestroy(_fftPlanFwd);
	}
	inline int iDivide(int a, int b) {
		return a % b != 0 ? a / b + 1 : a / b;
	}
	void cuda_init();
	void compute(T *dft_result);
	void cuda_free();
	unsigned int _length;
	unsigned int _batch;
	T* _dataline;//外部进来的数据用基础指针，不用智能指针。
	//gpu设备内存
	T* _d_dataline;
	cufftDoubleComplex* _d_dataline_comp;
	T* _d_dft_result;
	cufftDoubleComplex* _d_dft_result_comp;
	cufftHandle _fftPlanFwd;
};

template<typename T>
void fft_1D_cuda<T>::cuda_init()
{
	RUNTIME_CUDA_ERROR(cudaMalloc(&_d_dataline, sizeof(T)*_length*_batch));
	RUNTIME_CUDA_ERROR(cudaMalloc(&_d_dataline_comp, sizeof(cufftDoubleComplex)*_length*_batch));
	RUNTIME_CUDA_ERROR(cudaMalloc(&_d_dft_result, sizeof(T)*_length*_batch));
	RUNTIME_CUDA_ERROR(cudaMalloc(&_d_dft_result_comp, sizeof(cufftDoubleComplex)*_length*_batch));
	int n[1] = { _length };
	int inembed[1] = { 0 };
	int onembed[1] = { 0 };
	CUFFT_CUDA_ERROR(cufftPlanMany(&_fftPlanFwd, 1, n, inembed, 1, _length, onembed, 1, _length, CUFFT_Z2Z, _batch));
}

template<typename T>
void fft_1D_cuda<T>::compute(T *dft_result)
{
	RUNTIME_CUDA_ERROR(cudaMemcpy(_d_dataline, _dataline, sizeof(T)*_length*_batch, cudaMemcpyHostToDevice));

	//进行傅立叶变换
	cufft_R2C_kernel << <iDivide(_length*_batch, 256), 256 >> > (_d_dataline, _d_dataline_comp, _length*_batch);

	CUFFT_CUDA_ERROR(cufftExecZ2Z(_fftPlanFwd, _d_dataline_comp, _d_dft_result_comp, CUFFT_FORWARD));
	//READ_CUDA_DATA_COMP(_d_dft_result_comp, _length*_batch);

	cufft_comp_asb_kernel << <iDivide(_length*_batch, 256), 256 >> > (_d_dft_result_comp, _d_dft_result, _length*_batch);
	RUNTIME_CUDA_ERROR(cudaGetLastError());


	RUNTIME_CUDA_ERROR(cudaMemcpy(dft_result, _d_dft_result, sizeof(T)*_length*_batch, cudaMemcpyDeviceToHost));
}

template<typename T>
void fft_1D_cuda<T>::cuda_free()
{
	CUDA_FREE(_d_dataline);
	CUDA_FREE(_d_dft_result);
	CUDA_FREE(_d_dataline_comp);
	CUDA_FREE(_d_dft_result_comp);
	CUFFT_CUDA_ERROR(cufftDestroy(_fftPlanFwd));
}

func_header<float> * get_fft_1D_GPU(float *dataline, unsigned int length,unsigned int batch)
{
	return new fft_1D_cuda<float>(dataline, length, batch);
}