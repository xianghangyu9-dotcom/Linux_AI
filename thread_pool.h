#ifndef __THREAD_POOL_H
#define __THREAD_POOL_H

#include<iostream>
#include<queue>
#include<mutex>
#include<map>
#include<thread>
#include<condition_variable>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include"yolov5s.h"

using namespace std;
using namespace cv;

class   ThreadPool
{
    public:
        ThreadPool(const char *model_path,int num_threads);
        ~ThreadPool();
        int submit_task(const Mat& img,int index);
        int get_result(Mat& img,int index);
        queue<pair<int,Mat>> tasks;
        bool is_result_ready(int index); 

    private:
        vector<shared_ptr<Yolov5s>> yolo_group;

        int init(const char *model_path,int num_threads);
        bool run;
        mutex task_mtx;
        condition_variable task_cond;
        map<int,Mat> img_results;  
        mutex res_mtx;

        vector<thread> threads;
        void worker(int index);
};

#endif