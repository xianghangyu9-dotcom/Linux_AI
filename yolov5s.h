#ifndef __YOLOVSS_H
#define __YOLOVSS_H

#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<dlfcn.h>
#include<sys/time.h>

#include"rga.h"
#include"im2d.h"
#include"process.h"

#include<opencv2/imgproc.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>

#include"rknn_api.h"

using namespace std;

class Yolov5s
{
    public:
        Yolov5s(const char *model_path);
        ~Yolov5s();
        int inference_imge(const cv::Mat &orig_img,detect_result_group_t &group);
        int draw_result(cv::Mat &orig_img,detect_result_group_t &group);


        rknn_context ctx;
        rknn_sdk_version version;
        
        rknn_input_output_num io_num;   
        vector<rknn_tensor_attr> input_attrs;    
        vector<rknn_tensor_attr> output_attrs;

        int model_height;
        int model_width;
        int model_channel;

        int img_height;
        int img_width;
        int img_channel;


    private:
        unsigned char* model_data;
        unsigned char* resize_buf;
        int model_data_size;
        unsigned char * load_model(const char *file_name,int *model_size);
        unsigned char * load_data(FILE *fp,size_t ofst,size_t sz);
};


#endif