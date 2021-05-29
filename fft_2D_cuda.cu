#include"func_header.h"
#include"fft_1D_cuda.cuh"

template<typename T>
class fft_2D_cuda :public func_header<T>
{
public:
	fft_2D_cuda(T *dataImage, unsigned int height,unsigned int width, unsigned int batch) : _dataImage(dataImage),_height(height), _width(width), _batch(batch) {
		cuda_init();
	}
	void get_dft_result(T* dft_result)
	{
		compute(dft_result);
	}
	void reset()
	{
		_dataImage = nullptr;
	}
	void del_object() {
		delete this;
	}

private:
	~fft_2D_cuda()
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
	unsigned int _height;
	unsigned int _width;
	unsigned int _batch;
	T* _dataImage;//外部进来的数据用基础指针，不用智能指针。
	//gpu设备内存
	T* _d_dataImage;
	cufftDoubleComplex* _d_dataImage_comp;
	T* _d_dft_result;
	cufftDoubleComplex* _d_dft_result_comp;
	cufftHandle _fftPlanFwd;
};

template<typename T>
void fft_2D_cuda<T>::cuda_init()
{
	RUNTIME_CUDA_ERROR(cudaMalloc(&_d_dataImage, sizeof(T)*_height*_width*_batch));
	RUNTIME_CUDA_ERROR(cudaMalloc(&_d_dataImage_comp, sizeof(cufftDoubleComplex)*_height*_width*_batch));
	RUNTIME_CUDA_ERROR(cudaMalloc(&_d_dft_result, sizeof(T)*_height*_width*_batch));
	RUNTIME_CUDA_ERROR(cudaMalloc(&_d_dft_result_comp, sizeof(cufftDoubleComplex)*_height*_width*_batch));
	int n[2] = { _height,_width };
	int inembed[] = { _height,_width };
	int onembed[] = { _height,_width };
	CUFFT_CUDA_ERROR(cufftPlanMany(&_fftPlanFwd, 2, n, inembed, 1, _height*_width, onembed, 1, _height*_width, CUFFT_Z2Z, _batch));
}

template<typename T>
void fft_2D_cuda<T>::compute(T *dft_result)
{
	RUNTIME_CUDA_ERROR(cudaMemcpy(_d_dataImage, _dataImage, sizeof(T)*_height*_width*_batch, cudaMemcpyHostToDevice));

	//进行傅立叶变换
	cufft_R2C_kernel << <iDivide(_height*_width*_batch, 512), 512 >> > (_d_dataImage, _d_dataImage_comp, _height*_width*_batch);

	CUFFT_CUDA_ERROR(cufftExecZ2Z(_fftPlanFwd, _d_dataImage_comp, _d_dft_result_comp, CUFFT_FORWARD));
	//READ_CUDA_DATA_COMP(_d_dft_result_comp, _length*_batch);

	cufft_comp_asb_kernel << <iDivide(_height*_width*_batch, 512), 512 >> > (_d_dft_result_comp, _d_dft_result, _height*_width*_batch);
	RUNTIME_CUDA_ERROR(cudaGetLastError());


	RUNTIME_CUDA_ERROR(cudaMemcpy(dft_result, _d_dft_result, sizeof(T)*_height*_width*_batch, cudaMemcpyDeviceToHost));
}

template<typename T>
void fft_2D_cuda<T>::cuda_free()
{
	CUDA_FREE(_d_dataImage);
	CUDA_FREE(_d_dataImage_comp);
	CUDA_FREE(_d_dft_result_comp);
	CUDA_FREE(_d_dft_result);
	CUFFT_CUDA_ERROR(cufftDestroy(_fftPlanFwd));
}

func_header<float> * get_fft_2D_GPU(float *dataImage, unsigned int height,unsigned int width, unsigned int batch)
{
	return new fft_2D_cuda<float>(dataImage, height, width, batch);
}