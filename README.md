# 基于RK3568的实时YOLOv5目标检测与RTMP推流

本项目在瑞芯微RK3568开发板上实现实时视频采集、YOLOv5s目标检测、检测框绘制及RTMP推流。

## 主要特性
- 使用V4L2采集MJPEG摄像头视频流
- 通过RKNN API加载量化YOLOv5s模型，在NPU上推理
- 多线程并发处理（采集、推理、推流线程）与有界队列缓冲
- 结果帧编码为JPEG，通过FFmpeg硬件编码（h264_rkmpp）推流至RTMP服务器
- 针对嵌入式平台优化内存，采取JPEG压缩、队列容量控制等策略

## 运行平台
- 硬件：RK3568 (例如Firefly ITX-3568Q或同系列)
- 系统：Ubuntu 22.04 (aarch64)
- 依赖：OpenCV, RKNN Toolkit Lite, RGA (可选), FFmpeg with MPP

## 使用说明
- 编译：mkdir build && cd build && cmake .. && make
- 运行：./app
- 推流地址默认：rtmp://127.0.0.1/live/app，可修改源码中的rtmp_url
- 观看：在PC端使用ffplay打开同样的rtmp地址