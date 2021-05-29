#pragma once
/*************************************************************************
	> 模块: 
	> 此版本适用情景:
	> 链接外部库: 
	> 时间: 
	> 文件：
 ************************************************************************/
#define _CRT_SECURE_NO_WARNINGS
#include<iostream>

using namespace std;

template<class Y>
class func_header
{
public:
	virtual void get_dft_result(Y* dft_result) = 0;
	virtual void del_object() = 0;
};


func_header<float> * get_fft_1D_CPU(float *dataline, unsigned int length);//若CPU想要求解多个一维信号，需要采用for循环
func_header<float> * get_fft_1D_GPU(float *dataline, unsigned int length, unsigned int batch);//batch表示并行的信号数量

func_header<float> * get_fft_2D_CPU(float *dataImage, unsigned int height, unsigned int width);
func_header<float> * get_fft_2D_GPU(float *dataImage, unsigned int height, unsigned int width, unsigned int batch);//batch表示并行的二维信号数量