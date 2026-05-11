# 基于RK3588的三核并行YOLOv5目标检测与RTMP推流
- 本项目在瑞芯微RK3588开发板上实现实时视频采集、YOLOv5s多核并行推理、检测框绘制及低延迟RTMP推流。

## 主要特性
- 使用V4L2采集MJPEG摄像头视频流
- 通过RKNN API加载量化YOLOv5s模型，利用三核NPU并行推理（Core 0/1/2绑定）
- 多线程并发处理（采集、推理、推流线程）与有界队列缓冲
- RGA硬件加速图像预处理，原始像素直送FFmpeg，消除额外编解码开销
- 结果帧通过RKMPP硬件H.265编码（无B帧）推流至RTMP服务器，延迟低于150ms
- 针对嵌入式平台优化内存与调度，CPU占用极低

## 运行平台
- 硬件：RK3588（如Firefly、Orange Pi 5 Plus等）
- 系统：Ubuntu 20.04 或更高版本 (aarch64)
- 依赖：OpenCV, RKNN Toolkit2, RGA, MPP, FFmpeg with rkmpp

## 使用说明
- 编译：mkdir build && cd build && cmake .. && make -j8
- 运行：./app
- 默认RTMP地址可在命令参数中指定,推流地址默认：rtmp://127.0.0.1/live/app，可修改源码中的rtmp_url
- 观看：在PC端使用ffplay rtmp://ip/live/app 拉流