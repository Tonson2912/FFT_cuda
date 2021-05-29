#pragma once
/*************************************************************************
	> 模块: 
	> 此版本适用情景:
	> 链接外部库: 
	> 时间: 
	> 文件：
 ************************************************************************/
#include"func_header.h"
#include<opencv2\opencv.hpp>

template<typename T>
class fft_2D :public func_header<T>
{
public:
	fft_2D(T *dataImage, unsigned int height,unsigned int width) :_dataImage(dataImage),_height(height), _width(width) {}
	void get_dft_result(T* dft_result) {
		compute(dft_result);
	}

	void del_object() {
		delete this;
	}
private:
	~fft_2D()
	{

	}
	void compute(T *dft_result);
	unsigned int _height;
	unsigned int _width;
	T* _dataImage;//外部进来的数据用基础指针，不用智能指针。
//	T* _dft;
};

template<typename T>
void fft_2D<T>::compute(T *dft_result)
{
	//指针检查
	assert(_dataImage);

	//傅里叶输入数据
	cv::Mat df1_input = cv::Mat(_height, _width, CV_32FC1, _dataImage);//_dataImage和df1_input.data指向同一块内存
	cv::Mat temp = df1_input.clone();//这里需要使用clone进行深拷贝，否则源数据会被修改
	cv::Mat comp_mat_vector[] = { cv::Mat_<float>(temp), cv::Mat::zeros(df1_input.size(),CV_32FC1) };
	cv::Mat complex_mat;
	cv::merge(comp_mat_vector, 2, complex_mat);

	//进行正向傅里叶变化
	cv::dft(complex_mat, complex_mat);
	cv::split(complex_mat, comp_mat_vector);


	//对结果求绝对值
	cv::Mat df1_output = cv::Mat(_height, _width, CV_32FC1);
	df1_output = comp_mat_vector[0].mul(comp_mat_vector[0]) + comp_mat_vector[1].mul(comp_mat_vector[1]);
	cv::sqrt(df1_output, df1_output);

	//拷贝结果
	//_dft = (float *)df1_output.data;
	memcpy_s(dft_result, sizeof(T)*_height*_width, (T *)df1_output.data, sizeof(T)*_height*_width);


}
func_header<float> * get_fft_2D_CPU(float *dataImage, unsigned int height,unsigned int width)
{
	return new fft_2D<float>(dataImage, height, width);
}