#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<thread>
#include<unistd.h>
#include<queue>
#include<mutex>
#include<map>
#include<atomic>

#include"safequeue.h"
#include"thread_pool.h"
#include"yolov5s.h"
#include"process.h"

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<fcntl.h>
#include<errno.h>
#include<sys/ioctl.h>
#include<vector>

#include<linux/videodev2.h>
#include"sys/mman.h"
#include<sys/poll.h>

using namespace std;
using namespace cv;

#define MODEL_PATH          "/home/topeet/project01/model/yolov5_quantized.rknn"

struct FrameData
{
    Mat frame;
    int index;  
};

struct buf_app
{
    void *start;
    size_t length;
};

ThreadPool thread_pool(MODEL_PATH,2);

SafeQueue<FrameData> ReadFrameQueue(4); 
//SafeQueue<FrameData> WriteFrameQueue(30);
SafeQueue<Mat> StreamQueue(5);

void read_function(int fd,vector<buf_app>& buf_a,SafeQueue<FrameData>& r_queue,int& img_index)
{
    int ret = 0;

    while(true){

        FrameData frame_temp;

        struct pollfd poll_fd[1];
        poll_fd[0].fd = fd;
        poll_fd[0].events = POLLIN;

        poll(poll_fd,1,500);

        v4l2_buffer buf;
        memset(&buf,0,sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

        ret = ioctl(fd,VIDIOC_DQBUF,&buf);
        if(ret < 0)
        {
            perror("dequeue buffer");
            break;
        }

        // char filename[20];
        // snprintf(filename,sizeof(filename),"picture-%d.jpg",img_index);
        
        vector<uchar>encoded((uchar*)buf_a[buf.index].start,(uchar*)buf_a[buf.index].start + buf.bytesused);
        frame_temp.frame = imdecode(encoded,IMREAD_COLOR); 
        
        if(!frame_temp.frame.empty())
        {
            //imwrite(filename,frame_temp.frame); //测试
            frame_temp.index = img_index;
            r_queue.enqueue(frame_temp); 
            img_index++;
        }
        else
        {
            printf("frame is empty\n");
            break;
        }

        ret = ioctl(fd,VIDIOC_QBUF,&buf);
        if(ret < 0)
        {
            perror("queue buffer");
            break;
        }
    }
    printf("read end!\r\n");
}

void process_function(SafeQueue<FrameData>& r_queue, SafeQueue<Mat>& s_queue, bool& read_finished, bool& process_finished)
{
    int send_index = 0;
    int recv_index = 0;

    while(true) 
    {
        bool has_work = false;

        if(!r_queue.empty()) {
            FrameData f;
            r_queue.dequeue(f); 
            thread_pool.submit_task(f.frame, f.index);
            send_index++;
            f.frame.release(); 
            has_work = true;
        }

        if(thread_pool.is_result_ready(recv_index))
        {
            Mat res;
            thread_pool.get_result(res, recv_index);
            //w_queue.enqueue({res, recv_index});
            s_queue.enqueue(res);
            if (recv_index % 100 == 0) printf("process index %d finished!\n", recv_index);
            recv_index++;
            has_work = true;
        }

        if (read_finished && r_queue.empty() && recv_index >= send_index && send_index > 0)
        {
            process_finished = true;
            break;
        }

        if (!has_work) {
            this_thread::sleep_for(chrono::milliseconds(1));
        }
    }
    printf("process end!\n");
}

void stream_function(SafeQueue<Mat>& s_queue,bool& process_finished,const string& rtmp_url)
{
    string cmd = "ffmpeg -f rawvideo -pixel_format bgr24 -video_size 1280x720 -framerate 30 -i - "
                 "-c:v h264_rkmpp -b:v 2M -f flv " + rtmp_url;
    FILE* pipe = popen(cmd.c_str(), "w");
    if (!pipe) {
        printf("Failed to start FFmpeg\n");
        return;
    }

    while (true) {
        Mat frame;
        if (!s_queue.empty()) {
            s_queue.dequeue(frame);
            fwrite(frame.data, 1, frame.total() * frame.elemSize(), pipe);
            //fflush(pipe);
        } else if (process_finished) {
            break;
        }
    }

    pclose(pipe);
}

// void write_fuction(VideoWriter& writer,SafeQueue<FrameData>& img_queue,bool& process_finished)
// {
//     FrameData frame_temp;
//     auto start = chrono::high_resolution_clock::now();
//     while (true)
//     {
//         if(!img_queue.empty())
//          {
//                 img_queue.dequeue(frame_temp); 
//                 if(!frame_temp.frame.empty())
//                 {   
//                     writer.write(frame_temp.frame);
//                     frame_temp.frame.release(); 
//                 }
//                 if(frame_temp.index % 20 == 0)
//                 {
//                     printf("write index %d finished!\r\n",frame_temp.index);
//                 }
//         }
//         else if(process_finished)
//         {
            
//             break;
//         }
//         else
//         {
//             this_thread::sleep_for(chrono::milliseconds(1)); 
//         }
//     }
//     printf("write end!\r\n");
// }

int main(void){

    int fd = open("/dev/video10",O_RDWR | O_NONBLOCK);
    int ret = 0;
    if(fd == -1)
    {
        perror("video open failed!\n");
        return -1;
    }

    struct v4l2_capability cap;

    ret = ioctl(fd,VIDIOC_QUERYCAP,&cap);

    if(ret < 0)
    {
        perror("capability!\n");
        return -1;
    }

    printf("driver is %s,version is %d,device_Caps is %u\n",cap.driver,cap.version,cap.device_caps);

    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = 1280;
    fmt.fmt.pix.height = 720;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    ret = ioctl(fd,VIDIOC_S_FMT,&fmt);
    if(ret < 0)
    {
        perror("format!\n");
        return -1;
    }

    struct v4l2_streamparm streamparm;
    memset(&streamparm, 0, sizeof(streamparm));
    streamparm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    streamparm.parm.capture.timeperframe.numerator = 1;
    streamparm.parm.capture.timeperframe.denominator = 30;
    //int fps =  streamparm.parm.capture.timeperframe.denominator;
    ret = ioctl(fd, VIDIOC_S_PARM,&streamparm); 
    if(ret < 0)
    {
        perror("stream!\n");
        return -1;
    }
    
    v4l2_requestbuffers req
    {
        .count = 4,
        .type = V4L2_BUF_TYPE_VIDEO_CAPTURE,
        .memory = V4L2_MEMORY_MMAP
    };
    ioctl(fd,VIDIOC_REQBUFS,&req);
    if(ret < 0)
    {
        perror("reqest buffers");
        return -1;
    }

    std::vector<buf_app> buf_a;
    buf_a.resize(4); 

    v4l2_buffer buf_v;
    for(int i = 0;i<4;i++)
    {
        memset(&buf_v,0,sizeof(buf_v));
        buf_v.index = i;
        buf_v.memory = V4L2_MEMORY_MMAP;
        buf_v.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        ret = ioctl(fd,VIDIOC_QUERYBUF,&buf_v);
        if(ret < 0)
        {
            perror("buffer query");
            return -1;
        }

        buf_a[i].length =  buf_v.length;
        buf_a[i].start = mmap(NULL,buf_v.length,PROT_READ | PROT_WRITE,MAP_SHARED,fd,buf_v.m.offset);
    }

    for(int i=0;i<4;i++)
    {   
        v4l2_buffer buf;
        memset(&buf,0,sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.index = i;
        buf.memory = V4L2_MEMORY_MMAP;
        ret = ioctl(fd,VIDIOC_QBUF,&buf);
        if(ret < 0)
        {
            perror("query buffer");
            return -1;
        } 

    }

    v4l2_buf_type type_1;
    type_1 = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ret = ioctl(fd,VIDIOC_STREAMON,&type_1);
    if(ret < 0)
    {
        perror("stream on");
        return -1;
    }
    
    int index=-1;
    bool read_finished=false;
    bool process_finished=false;

    thread video_readers(read_function,ref(fd),ref(buf_a),ref(ReadFrameQueue),ref(index));
    
    //处理视频
    thread video_process(process_function,ref(ReadFrameQueue),ref(StreamQueue),ref(read_finished),ref(process_finished));
    
    //创建写入视频的对象
    //Size framesize(fmt.fmt.pix.width,fmt.fmt.pix.height);
    //VideoWriter writer("/home/topeet/project01/stream.avi",VideoWriter::fourcc('M','J','P','G'), fps, framesize);
    //创建线程将队列中的视频帧写入对象中
    //thread video_writer(write_fuction,ref(writer),ref(WriteFrameQueue), ref(process_finished));
    string rtmp_url = "rtmp://127.0.0.1/live/app";
    thread rtmp_thread(stream_function, std::ref(StreamQueue), std::ref(process_finished), rtmp_url);

    video_readers.join();
    read_finished=true;

    v4l2_buf_type type_2;
    type_2 = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ret = ioctl(fd,VIDIOC_STREAMOFF,&type_2);
    if(ret < 0)
    {
        perror("stream off");
        return -1;
    }

    //解除映射
    for(int i = 0;i<4;i++)
    {
        munmap(buf_a[i].start,buf_a[i].length);
    }

    close(fd);

    video_process.join();
    //video_writer.join();
    rtmp_thread.join();

    return 0;

}