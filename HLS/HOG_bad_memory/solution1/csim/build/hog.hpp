#include "stdint.h"
#include "ap_int.h"
#include "hls_math.h"
#include "ap_fixed.h"
#include <string.h>
#include <hls_opencv.h>

using namespace std;
using namespace cv;


struct obj
{
    int x,y;
    float scale,score;
};

std::vector<obj> SVM_Detection_visual_apr(Mat orginalImage,int pixelPerCell,int cellPerBlock,float scale);
