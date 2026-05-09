#ifndef __PROCESS_H
#define __PROCESS_H

#include<stdint.h>
#include<vector>
#include<fstream>
#include<string.h>
#include<string>
#include<iostream>
#include<math.h>
#include<algorithm>
#include<set>

#define OBJ_CLASS_NUM       80
#define LABEL_PATH          "/home/topeet/project01/model/coco_80_labels_list.txt" 

#define OBJ_NAME_MAX_SIZE   64
#define LABEL_NAME_MAX_SIZE 32
#define BOX_THRESHOLD       0.5
#define NMS_THRESHOLD       0.5
#define BOX_NUM             (OBJ_CLASS_NUM+5)

using namespace std; 

typedef struct _box_t
{
    int left_top_x;
    int left_top_y;
    int right_bottom_x;
    int right_bottom_y;   
}box_t;

typedef struct _detect_result_t
{
    char label_name[LABEL_NAME_MAX_SIZE];
    float box_conf;
    box_t box;
}detect_result_t;

typedef struct _detect_result_group_t
{
    int box_count;
    detect_result_t result[OBJ_NAME_MAX_SIZE];
}detect_result_group_t;

int process(int8_t *input,int *anchor,int grid_h,int grid_w,int model_height,int model_width,int stride,
    vector<float>& boxes,vector<float>& objprobs,vector<int> &classid,float box_threshold,int32_t zp,float scale);
int post_process(int8_t *output0,int8_t *output1,int8_t *output2,int model_height,int model_width,float box_threshold,
float nms_threshold,float scale_w,float scale_h,vector<int32_t> &qnt_zps,vector<float> &qnt_scales,detect_result_group_t &group);

#endif