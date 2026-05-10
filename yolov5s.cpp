#include"yolov5s.h"
#include"process.h"

static void print_tensor_attr(rknn_tensor_attr &attr)
{   
    string shape_str = attr.n_dims<1 ? "" : to_string(attr.dims[0]);  
    for(int i= 1;i<attr.n_dims;i++)
    {
        string current_str = to_string(attr.dims[i]);
        shape_str += "," + current_str;                 
    }
    printf("index = %d, name = %s, n_dims = %d, dims = [%s],\nsize = %d, fmt = %s,\nscale = %f, zp = %d\n",
        attr.index,attr.name,attr.n_dims,shape_str.c_str(),attr.size,get_format_string(attr.fmt),attr.scale,attr.zp);
}   

//Yolov5s::Yolov5s(const char *model_path)
Yolov5s::Yolov5s(const char *model_path, rknn_core_mask core_mask)
{
    int ret;

    this->model_data_size = 0;
    this->model_data      = load_model(model_path,&this->model_data_size);

    ret                   = rknn_init(&this->ctx, model_data, model_data_size, 0, NULL);
    ret                   = rknn_set_core_mask(this->ctx, core_mask);
    if(ret<0)
    {
        printf("model_init failed,error code = %d\n",ret);
    }
    else
    {
        printf("model init is ok\n");
    }

    //rknn_query查询版本和驱动
    rknn_query(ctx,RKNN_QUERY_SDK_VERSION,&this->version,sizeof(this->version));
    //printf("sdk version : %s ,driver version : %s\n",version.api_version,version.drv_version); 

    ret = rknn_query(ctx,RKNN_QUERY_IN_OUT_NUM,&this->io_num,sizeof(this->io_num));
    if(ret<0)
    {
        printf("get io num failed\n");
    }
    //printf("input num: %d,output num: %d\n",io_num.n_input,io_num.n_output);


    input_attrs.resize(io_num.n_input);
    output_attrs.resize(io_num.n_output);

    for(int i=0;i<io_num.n_input;i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx,RKNN_QUERY_INPUT_ATTR,&(input_attrs[i]),sizeof(rknn_tensor_attr));
        if(ret<0)
        {
        printf("get index %d input failed\n",i);
        }
        //print_tensor_attr(input_attrs[i]); 
    }

    for(int i=0;i<io_num.n_output;i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx,RKNN_QUERY_OUTPUT_ATTR,&(output_attrs[i]),sizeof(rknn_tensor_attr));
        if(ret<0)
        {
        printf("get index %d output failed\n",i);
        }
        //print_tensor_attr(output_attrs[i]);

    }

    //获取format
    if(input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        model_channel = input_attrs[0].dims[1];
        model_height = input_attrs[0].dims[2];
        model_width = input_attrs[0].dims[3];
    }
    else if(input_attrs[0].fmt == RKNN_TENSOR_NHWC)
    {
        model_height = input_attrs[0].dims[1];
        model_width = input_attrs[0].dims[2];
        model_channel = input_attrs[0].dims[3];
    }
    this->resize_buf = (unsigned char*)malloc(model_width * model_height * 3);
}


Yolov5s::~Yolov5s()
{
    rknn_destroy(ctx);
    if(model_data)
    {
        free(model_data);
    }
    if(this->resize_buf) free(this->resize_buf);
}

unsigned char * Yolov5s::load_model(const char *file_name,int *model_size)
{
    FILE *fp;
    unsigned char* data;

    fp = fopen(file_name,"rb"); 
    if(fp == NULL)
    {
        printf("model open file %s is failed.\n",file_name);
        return NULL;
    }

    fseek(fp,0,SEEK_END);
    int size = ftell(fp);
    data = load_data(fp,0,size);
    fclose(fp);
    *model_size=size;
    return data;
}

unsigned char * Yolov5s::load_data(FILE *fp,size_t ofst,size_t sz)
{
    unsigned char* data;
    int ret;

    if(fp == NULL)
    {
        return NULL;
    }
    ret = fseek(fp,ofst,SEEK_SET);
    if(ret != 0)
    {
        printf("load_data seek failed\n");
    }

    data = (unsigned char *)malloc(sz);
    if(data == NULL)
    {
        printf("load failed\n");
        return NULL;
    }
    ret = fread(data,1,sz,fp);
    if(ret < 0)
    {
        printf("load_model read failed\n");
        return NULL;
    }

    return data;
}

int Yolov5s::inference_imge(const cv::Mat &orig_img,detect_result_group_t &group)
{
    int ret=0;

    this->img_height    = orig_img.rows;        
    this->img_width     = orig_img.cols;
    this->img_channel   = orig_img.channels();
    
    //预处理
    rga_buffer_t src = wrapbuffer_virtualaddr((void*)orig_img.data, orig_img.cols, orig_img.rows, RK_FORMAT_BGR_888);
    rga_buffer_t dst = wrapbuffer_virtualaddr((void*)this->resize_buf, model_width, model_height, RK_FORMAT_RGB_888);

    ret = improcess(src, dst, {}, {}, {}, {}, IM_SYNC); 
    if (ret < 0) {
        printf("RGA Process error!\n");
        // 如果 RGA 失败，降级使用 OpenCV
        cv::Mat img_rgb, img_resize;
        cv::cvtColor(orig_img, img_rgb, cv::COLOR_BGR2RGB);
        cv::resize(img_rgb, img_resize, cv::Size(model_width, model_height));
        memcpy(this->resize_buf, img_resize.data, model_width * model_height * 3);
    }

    //推理
    //input初始化
    int inputs_num = io_num.n_input;
    rknn_input inputs[inputs_num];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = model_height * model_width * 3;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = this->resize_buf; // 使用硬件加速处理后的数据

    rknn_inputs_set(ctx, 1, inputs);
    ret = rknn_run(ctx, NULL);

    //output初始化
    //printf("set output...\n");
    int outputs_num = io_num.n_output;
    rknn_output outputs[outputs_num];
    memset(outputs,0,sizeof(outputs));
    for(int i=0;i<outputs_num;i++)
    {
        outputs[i].want_float = 0;
    }

    ret = rknn_outputs_get(ctx,outputs_num,outputs,NULL);

    //后处理
    float scale_w = (float)model_width / img_width;
    float scale_h = (float)model_height / img_height;

    vector<int32_t> qnt_zps;
    vector<float>   qnt_scales;

    for(int i = 0;i<outputs_num;i++)
    {
        qnt_zps.emplace_back(output_attrs[i].zp);
        qnt_scales.emplace_back(output_attrs[i].scale);
    }

    post_process((int8_t *)outputs[0].buf,(int8_t *)outputs[1].buf,(int8_t *)outputs[2].buf,model_height,model_width,BOX_THRESHOLD,
    NMS_THRESHOLD,scale_w,scale_h,qnt_zps,qnt_scales,group);

    rknn_outputs_release(ctx, io_num.n_output, outputs);
    
    return 0;
}

int Yolov5s::draw_result(cv::Mat &orig_img,detect_result_group_t &group)
{
    char label_name[256];
    for(int i = 0;i < group.box_count;i++)
    {   
        detect_result_t *result = &(group.result[i]);
        snprintf(label_name,sizeof(label_name),"%s %.1f%%",result->label_name,result->box_conf*100.0);
        int xmin    = result->box.left_top_x;
        int ymin    = result->box.left_top_y;
        int xmax    = result->box.right_bottom_x;
        int ymax    = result->box.right_bottom_y;
        //用cv函数画框,并设定框的参数
        cv::rectangle(orig_img,cv::Point(xmin,ymin),cv::Point (xmax,ymax),cv::Scalar(255,0,0),3);
        //设置字体格式
        cv::putText(orig_img,label_name,cv::Point(xmin+10,ymin+10),cv::FONT_HERSHEY_SCRIPT_SIMPLEX,0.5,cv::Scalar(0,0,0));
    }
    return 0;
}