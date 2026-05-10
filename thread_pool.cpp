#include"thread_pool.h"

ThreadPool::ThreadPool(const char *model_path,int num_threads)
{
    printf("初始化线程池\r\n");
    this->run = true;
    init(model_path,num_threads);

}

ThreadPool::~ThreadPool()
{
    printf("析构线程池\r\n");
    this->run = false;
    task_cond.notify_all();
    
    for(thread& t:threads)
    {
        if(t.joinable())
        {
            t.join();
        }
    }

}

int ThreadPool::init(const char *model_path,int num_threads)
{
    if(!num_threads)
    {
        return -1;
    }
    
    for(size_t i = 0; i < num_threads; i++) {
        rknn_core_mask mask;
        if (i % 3 == 0) mask = RKNN_NPU_CORE_0;
        else if (i % 3 == 1) mask = RKNN_NPU_CORE_1;
        else mask = RKNN_NPU_CORE_2;
        //rknn_core_mask mask = RKNN_NPU_CORE_AUTO;

        // 创建Yolov5s实例并绑定核心
        shared_ptr<Yolov5s> yolov5s = make_shared<Yolov5s>(model_path, mask);
        yolo_group.emplace_back(yolov5s);
        threads.emplace_back(&ThreadPool::worker, this, i);
    }
    printf("ThreadPool is ok\r\n");
    return 0;
}

void ThreadPool::worker(int index)
{
    auto yolo = yolo_group[index];
    while(run)
    {
        pair<int,Mat> task;
        {
            unique_lock<mutex> lock(task_mtx);
            task_cond.wait(lock,[&]{return (!tasks.empty() || !run);}); 

            if(!run)
            {
                printf("worker%d is relaxing!\r\n",index);
                break;
            }

            task=tasks.front();     
            tasks.pop();
        }
        //图像初始化和画框部分
        detect_result_group_t group;
        yolo->inference_imge(task.second,group);

        yolo->draw_result(task.second,group);

        {
            lock_guard<mutex> lock(res_mtx);
            img_results.insert({task.first,task.second});  
        }
    }
}

int ThreadPool::submit_task(const Mat& img,int index)
{
    while(tasks.size() > 5)
    {
        this_thread::sleep_for(chrono::milliseconds(1));
    }

    //保存任务
    {
        lock_guard<mutex> lock(task_mtx);
        tasks.push({index,img});
    }
    task_cond.notify_one();
    return 0;

}

int ThreadPool::get_result(Mat& img,int index)
{
    while (true)
    {
        {
            lock_guard<mutex> lock(res_mtx); 
            if (img_results.find(index) != img_results.end()) {
                img = img_results[index];
                img_results.erase(index);
                return 0; 
            }
        }
        this_thread::sleep_for(chrono::milliseconds(2)); 
    }
}

bool ThreadPool::is_result_ready(int index) 
        {
            lock_guard<mutex> lock(res_mtx);
            return img_results.find(index) != img_results.end();
        }