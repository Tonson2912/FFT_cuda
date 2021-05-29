#define _CRT_SECURE_NO_WARNINGS
#include"func_header.h"
#include<fstream>
#include<opencv2\opencv.hpp>

//�������ļ���ȡ����
template<class T>
void Input_data(T *P, unsigned int row, unsigned int col, unsigned int depth, int shft, const char *s1)
{
	FILE *fid0 = fopen(s1, "rb");
	if (fid0 == NULL)
	{
		std::cout << "Open error !!!" << std::endl;
		std::fclose(fid0);
	}
	//int shft = 0;
	fseek(fid0, shft, SEEK_CUR);

	memset(P, 0, sizeof(T));
	fread(P, row*col*depth * sizeof(T), 1, fid0);
	fclose(fid0);
}

//�ı��ļ���д����
template<class T>
void Input_data1(T *P, unsigned int length,const char *s1)
{
	fstream f;
	f.open(s1, ios::in);
	if (!f)
	{
		std::cout << "Open error !!!" << std::endl;
		f.close();
	}
	//int shft = 0;
	int index_temp = 0;
	T x_pos_temp = 0, y_pos_temp = 0;
	int i = 0;

	while (!f.eof())
	{
		f >> P[i++];
		//f >> index_temp >> x_pos_temp >> y_pos_temp;
		//index.push_back(index_temp);
		//x_pos.push_back(x_pos_temp);
		//y_pos.push_back(y_pos_temp);
		//cout << index_temp << "\t" << x_pos_temp << "\t" << y_pos_temp << endl;
	}
	f.close();
}

using func_ptr_cpu = func_header<float>* (*)(float *dataline, unsigned int length);
using func_ptr_gpu = func_header<float>* (*)(float *dataline, unsigned int length,unsigned int batch);

int main()
{
	//��ȡ����
	const char *s1 = "E:\\Something about OCT\\Kֵ��ɫɢ����\\С��\\20201110150304-3100-3150.txt";
	unsigned int length = 2048;
	float *dataline = new float[length]();//ԭʼ�ź�����
	float *df1 = new float[length]();//����Ҷ�ź�����
	Input_data1<float>(dataline, length, s1);//�ı��ļ���ȡ����Ҫ�ö������ļ���ȡ����ѡ��Input_data
	
	func_ptr_gpu gpu_compute = get_fft_1D_GPU;//����ѡ��GPU�����һά���ά����Ҷ
	func_ptr_cpu cpu_compute = get_fft_1D_CPU;//����ѡ��CPU�����һά���ά����Ҷ

	//���и���Ҷ����
	auto fft = gpu_compute(dataline, length, 1);
	fft->get_dft_result(df1);
	fft->del_object();

	//��ӡ������������޸�
	for (int i = 0; i < length; ++i)
	{
		cout << i + 1 << "  :" << df1[i] << endl;
	}

	system("pause");

	delete[] dataline;
	delete[] df1;
}

//int main()
//{
//	unsigned int height = 1040; unsigned int width = 1392; unsigned int num = 1;//����5��
//	const char *s1 = "E:\\Something about OCT\\Kֵ��ɫɢ����\\fft_Func\\sample2008+Single[1392x1040x30].txt";//��ʵ��Ʒ
//	float *realSample = new float[height*width*num](); Input_data<float>(realSample, height, width, num, 0, s1);
//
//	cv::Mat im = cv::Mat(height, width, CV_32FC1, realSample);
//	//im = im * (1.0f / 65535) * 255;
//	//im.convertTo(im, CV_8UC1);
//
//
//	auto fft = get_fft_2D_CPU(realSample, height, width);
//	float *df1 = new float[height*width]();
//	fft->get_dft_result(df1);
//	fft->del_object();
//	
//	cv::Mat out = cv::Mat(height, width, CV_32FC1, df1);
//	system("pause");
//
//}