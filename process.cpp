#include"process.h"

const int anchor0[6]={10,13,16,30,33,23};       //锚框0参数
const int anchor1[6]={30,61,62,45,59,119};      //锚框1参数
const int anchor2[6]={116,90,156,198,373,326};  //锚框2参数

struct probarray
{
    float conf;
    int index;
};

vector<string> labels;

int readlines(const char *filepath,vector<string>& label_vector,int maxlines)
{
    ifstream file(filepath);
    if(!file.is_open())
    {
        cerr<<"file" << filepath << "failed\n";
        return -1;
    }
    string line;
    while(getline(file,line))
    {
        label_vector.emplace_back(line);
        if(label_vector.size() >= static_cast<size_t>(maxlines))    //格式转换
        {
            break;
        }
    }
    return label_vector.size();
}

int load_labelname(const char *filepath,vector<string>& label_vector,int maxlines)
{
    int line_num = readlines(filepath,label_vector,OBJ_CLASS_NUM);
    // if(line_num > 0)
    // {
    //     cout<<"read "<<line_num<<" labels"<<endl;
    // }
    return 0;
}

//反量化
static float qnt_int8_to_f32(int8_t int_num,int32_t zp,float scale)
{
    float float_num = (float)(int_num - zp)*scale;
    return float_num;
}

inline static int32_t __limit_num(float val,int min, int max)
{
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}
//量化
static int8_t qnt_f32_to_int8(float float_num,int32_t zp,float scale)
{
    float float_qnt_num = (float_num / scale) + zp;
    int8_t int_num = (int8_t)__limit_num(float_qnt_num,-128,127);//防止超出范围
    return int_num;
}

//使用sort给数组排序
static int sort_descending(vector<probarray>& arr)
{
    sort(arr.begin(),arr.end(),[](const probarray& a,const probarray& b)
    {
        return a.conf > b.conf;
    });
    return 0;
}

static float cal_iou(float xmin0,float ymin0,float xmax0,float ymax0,float xmin1,float ymin1,float xmax1,float ymax1)
{
    float w = fmax(0.f,fmin(xmax0,xmax1) - fmax(xmin0,xmin1)+1.0);
    float h = fmax(0.f,fmin(ymax0,ymax1) - fmax(ymin0,ymin1)+1.0);//像素点处理后要+1
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0)*(ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0)*(ymax1 - ymin1 + 1.0) - i;
    float iou = u <= 0.f? 0.f : (i/u);  
    return iou;
}

static int nms(int validcount,vector<float>& boxes,vector<int>& classid,vector<int>& index,int current_class,float nms_thres)
{
    for(int i=0; i < validcount; i++)
    {
        if(index[i] == -1 || classid[i] != current_class)
        {
            continue;
        }

        int n = index[i];
        int maxid = i;

        for(int j = i+1; j < validcount; j++)
        {
            int m = index[j];
            if(m == -1 || classid[j] != current_class)
            {
                continue;
            } 
            float xmin0 = boxes[n * 4];
            float ymin0 = boxes[n * 4 + 1]; 
            float xmax0 = boxes[n * 4 + 2] + xmin0; 
            float ymax0 = boxes[n * 4 + 3] + ymin0;

            float xmin1 = boxes[m * 4];
            float ymin1 = boxes[m * 4 + 1]; 
            float xmax1 = boxes[m * 4 + 2] + xmin1; 
            float ymax1 = boxes[m * 4 + 3] + ymin1;

            float iou = cal_iou(xmin0,ymin0,xmax0,ymax0,xmin1,ymin1,xmax1,ymax1);

            if(iou > nms_thres)
            {
                index[i]=-1; 
            }
        }
    }
    return 0;
}

int process(int8_t *input,int *anchor,int grid_h,int grid_w,int model_height,int model_width,int stride,
    vector<float>& boxes,vector<float>& objprobs,vector<int>& classid,float box_threshold,int32_t zp,float scale)
{
    int validcount = 0;
    int grid_len = grid_h*grid_w;
    float threshold = box_threshold;
    for(int a=0;a<3;a++)
    {
        for(int i=0;i<grid_h;i++)
        {
            for(int j=0;j<grid_w;j++)
            {
                int8_t obj_threshold = input[(a*BOX_NUM+4)*grid_len + i*grid_w+j];
                float obj_threshold_f = qnt_int8_to_f32(obj_threshold,zp,scale);
                if(obj_threshold_f >= threshold)
                {
                    //printf("obj_threshold_f = %f\n",obj_threshold_f);
                    //int box_offt = (a*BOX_NUM)*grid_len+i*grid_w+j;
                    int8_t *box_p = input + (a*BOX_NUM)*grid_len+i*grid_w+j;
                    
                    float box_x = (qnt_int8_to_f32(*box_p,zp,scale))*2-0.5;
                    float box_y = (qnt_int8_to_f32(*(box_p+1*grid_len),zp,scale))*2-0.5;
                    float box_w = (qnt_int8_to_f32(*(box_p+2*grid_len),zp,scale))*2.0;
                    float box_h = (qnt_int8_to_f32(*(box_p+3*grid_len),zp,scale))*2.0;
                    
                    box_x = (box_x+j)*(float)stride;
                    box_y = (box_y+i)*(float)stride;
                    box_w = box_w*box_w*(float)anchor[a*2];
                    box_h = box_h*box_h*(float)anchor[a*2+1];
                    //坐标转移到网格左上角
                    box_x = box_x - (box_w/2.0);
                    box_y = box_y - (box_h/2.0);

                    boxes.emplace_back(box_x);
                    boxes.emplace_back(box_y);
                    boxes.emplace_back(box_w);
                    boxes.emplace_back(box_h);

                    //printf("box_x = %f\nbox_y = %f\nbox_w = %f\nbox_h = %f\n",box_x,box_y,box_w,box_h);

                    int8_t maxclassprob = *(box_p + 5*grid_len);
                    int maxclassid = 0;
                    for(int k = 1; k<OBJ_CLASS_NUM;k++)
                    {
                        int8_t prob = *(box_p+(5+k)*grid_len);
                        if(prob > maxclassprob)
                        {
                            maxclassprob = prob;
                            maxclassid = k;
                        }
                    }
                    objprobs.emplace_back(qnt_int8_to_f32(maxclassprob,zp,scale));
                    classid.emplace_back(maxclassid);
                    validcount++;
                }
                
            }
        }
    }

    return  validcount;
}

//后处理
int post_process(int8_t *output0,int8_t *output1,int8_t *output2,int model_height,int model_width,float box_threshold,
float nms_threshold,float scale_w,float scale_h,vector<int32_t> &qnt_zps,vector<float> &qnt_scales,detect_result_group_t &group)
{   
    //初始化
    static int init = -1;
    int ret;
    if(init == -1)
    {   
        ret = load_labelname(LABEL_PATH,labels,OBJ_CLASS_NUM);
        // if(ret == 0)
        // {
        //     for(string&s : labels)  
        //     {
        //         cout<<"label name : "<< s << endl;
        //     }
        // }
        // else
        // {
        //     cout<<" label name failed"<<endl;
        // }
        init = 0;
    }
    //output解析
    vector<float> detect_boxes;
    vector<float> objprobs;
    vector<int>   classid;
    //output0
    int stride0 = 8;    //分格
    int grid_h0 = model_height/stride0;
    int grid_w0 = model_width/stride0;
    int validcount0 = 0;
    validcount0 = process(output0,(int*)anchor0,grid_h0,grid_w0,model_height,model_width,
    stride0,detect_boxes,objprobs,classid,box_threshold,qnt_zps[0],qnt_scales[0]);
    //output1
    int stride1 = 16;    //分格
    int grid_h1 = model_height/stride1;
    int grid_w1 = model_width/stride1;
    int validcount1 = 0;
    validcount1 = process(output1,(int*)anchor1,grid_h1,grid_w1,model_height,model_width,
    stride1,detect_boxes,objprobs,classid,box_threshold,qnt_zps[1],qnt_scales[1]);
    //output2
    int stride2 = 32;    //分格
    int grid_h2 = model_height/stride2;
    int grid_w2 = model_width/stride2;
    int validcount2 = 0;
    validcount2 = process(output2,(int*)anchor2,grid_h2,grid_w2,model_height,model_width,
    stride2,detect_boxes,objprobs,classid,box_threshold,qnt_zps[2],qnt_scales[2]);

    //printf("validcount0:%d,validcount1:%d,validcount2:%d\n",validcount0,validcount1,validcount2);

    int validcount = validcount0 + validcount1 + validcount2;
    if(validcount < 0)
    {
        return 0;
    }
    
    vector<probarray> pro_arr;

    for(int i=0;i<validcount;i++)
    {
        probarray temp;
        temp.index= i;
        temp.conf= objprobs[i];
        pro_arr.emplace_back(temp);
    }
   
    //排序
    sort_descending(pro_arr);

    vector<int> indexarray;
    objprobs.clear();
    indexarray.clear();
    for(int i=0;i<validcount;i++)
    {
        objprobs.emplace_back(pro_arr[i].conf);
        indexarray.emplace_back(pro_arr[i].index);
    }

    set<int> class_set(begin(classid),end(classid));
    for(const int& id : class_set)
    {
        //printf("class is %s\n",labels[id].c_str());
        nms(validcount,detect_boxes,classid,indexarray,id,nms_threshold);
    }

    int count = 0;
    group.box_count = 0;
    for(int i=0;i<validcount;i++)
    {
        if(indexarray[i] == -1 || count >= OBJ_NAME_MAX_SIZE)
        {
            continue;
        }
        int n = indexarray[i];

        float x0       = detect_boxes[4 * n];
        float y0       = detect_boxes[4 * n + 1];
        float x1       = detect_boxes[4 * n + 2] + x0;
        float y1       = detect_boxes[4 * n + 3] + y0;
        int id         = classid[n];
        float box_conf = objprobs[i];
        //printf("cpp sort get objprobs[%d],index = %d,probles = %f,class is %s\n",i,indexarray[i],objprobs[i],labels[id].c_str());

        group.result[count].box.left_top_x      = (int)(__limit_num(x0,0,model_width)/scale_w);
        group.result[count].box.left_top_y      = (int)(__limit_num(y0,0,model_height)/scale_h);
        group.result[count].box.right_bottom_x  = (int)(__limit_num(x1,0,model_width)/scale_w);
        group.result[count].box.right_bottom_y  = (int)(__limit_num(y1,0,model_height)/scale_h);    //边缘裁剪并还原到原图上的坐标
        group.result[count].box_conf            = box_conf;
        
        const char *label_temp                  = labels[id].c_str();
        strncpy(group.result[count].label_name,label_temp,LABEL_NAME_MAX_SIZE);

        count++;
    }
    group.box_count = count;

    return 0;
}
