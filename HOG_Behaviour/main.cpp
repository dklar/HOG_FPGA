#include <iostream>
#include <unistd.h>
#include <stdexcept>
#include <iomanip>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

inline int MODULO(int a, int b) {
	int res = a % b;
	return res < 0 ? res + b : res;
}

inline float MODULO(float a, float b) {
	float res = fmod(a,b);
	return res < 0 ? res + b : res;
}

Mat gradientY(Mat pictureIn)
{
    Mat gradient(pictureIn.size(), CV_16SC1, cv::Scalar(0));
    try
    {
        for (int y = 1; y < pictureIn.rows - 1; y++)
        {
            for (int x = 0; x < pictureIn.cols; x++)
            {
                gradient.at<int16_t>(y, x) =
                    pictureIn.at<uint8_t>(y + 1, x) -
                    pictureIn.at<uint8_t>(y - 1, x);
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        cout << "Error at gradientY calculation";
    }
    return gradient;
}

Mat gradientX(Mat pictureIn)
{
    Mat gradient(pictureIn.size(), CV_16SC1, cv::Scalar(0));
    try
    {
        for (int y = 0; y < pictureIn.rows; y++)
        {
            for (int x = 0; x < pictureIn.cols - 2; x++)
            {
                gradient.at<int16_t>(y, x + 1) =
                    pictureIn.at<uint8_t>(y, x + 2) -
                    pictureIn.at<uint8_t>(y, x);
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        cout << "Error at gradientX calculation";
    }
    return gradient;
}

float cell_hog(Mat magnitude,Mat orientation,int start, int stop,int oStart, int oEnd,int r_index, int c_index,int pixelPerCell=8)
{
    int cri = 0;
    int cci = 0;
    float total = 0;

    for (int y = start; y < stop; y++)
    {
        cri = r_index + y;
        if (cri >= 0 && cri < magnitude.rows)
        {
            for (int x = start; x < stop; x++)
            {
                cci = c_index + x;
                bool c1 = cci < 0;
                bool c2 = cci >= magnitude.cols;
                bool c3 = orientation.at<int>(cri,cci) >= oStart;
                bool c4 = orientation.at<int>(cri,cci) < oEnd;
                if (c1 || c2 || c3 || c4)
                {
                }
                else
                {
                    total += magnitude.at<int>(cri,cci) ;
                }
            }
        }
    }
    return total / (pixelPerCell * pixelPerCell);
}

/**
 * @brief 
 * 
 * @param magnitude Magnitude matrix
 * @param orientation Orientation matrix bins 0 to 8
 * @param binNr actual bin number
 * @param row_offset 
 * @param col_offset 
 * @return float 
 */
float cell_hog1(Mat magnitude,Mat orientation, int binNr,int row_offset, int col_offset)
{
    int cri = 0;
    int cci = 0;
    float total = 0;
    int pixelPerCell=8;
    for (int y = 0; y < pixelPerCell; y++)
    {
        cri = row_offset + y;
        if (cri >= 0 && cri < magnitude.rows)
        {
            for (int x = 0; x < pixelPerCell; x++)
            {
                cci = col_offset + x;
                bool c1 = cci > 0;
                bool c2 = cci < magnitude.cols;
                bool c3 = orientation.at<int>(cri,cci) == binNr;
                if (c1 && c2 && c3)
                    total += magnitude.at<int>(cri,cci) ;
            }
        }
    }
    return total / (pixelPerCell * pixelPerCell);
}

void normalizeCell(float *cellValues, int len, std::string mode = "L1")
{
    if (mode == "L1")
    {
        float sum = 0;
        for (int i = 0; i < len; i++)
        {
            sum += abs(cellValues[i]);
        }
        for (int i = 0; i < len; i++)
        {
            cellValues[i] = cellValues[i] / sum;
        }
    }
    if (mode == "L2"){
        float sum = 0;
        float eps =0.;
        for (int i = 0; i < len; i++)
        {
            sum += sqrt(cellValues[i]*cellValues[i]+eps*eps);
        }
        for (int i = 0; i < len; i++)
        {
            cellValues[i] = cellValues[i] / sum;
        }
    }
}

Mat normalize_self(Mat in, float min, float max)
{
    Mat returnMat(in.size(), CV_8UC1, Scalar(0));
    for (int y = 1; y < in.rows - 1; y++)
    {
        for (int x = 1; x < in.cols - 1; x++)
        {
            float Inew = (in.at<float>(y, x) - min) * (255 / (max - min));

            if (Inew < 0)
                Inew = 0;
            int newValue = (uint8_t)nearbyintf32(Inew);
            if (newValue > 40 && newValue < 155)
                newValue += 100; //small cheat
            returnMat.at<uint8_t>(y, x) = (uint8_t)nearbyintf32(newValue);
        }
    }
    return returnMat;
}

/**
 * @brief Draws the HOG Values
 * 
 * @param image Image to adapt HOG
 * @param nrOrientation Number of Orientation per Bin (Normally 9)
 * @param pixelPerCell Pixel per cell (Normally 8)
 * @param cellPerBlock Cells per block (normally 2)
 * @return cv::Mat  The Matrix to show
 */
Mat Draw_HOG_Values(Mat image,int nrOrientation,int pixelPerCell,int cellPerBlock){
    Mat gx = gradientX(image);
    Mat gy = gradientY(image);


    Mat mag(image.size(),CV_32SC1,cv::Scalar(0));
    Mat ort(image.size(),CV_32SC1,cv::Scalar(0));
    for (int y = 0; y < image.rows; y++){
        for (int x = 0; x < image.cols; x++){
            int valueY = gy.at<int16_t>(y,x);
            int valueX = gx.at<int16_t>(y,x);
            int o = MODULO( (int) nearbyintf32(atan2f(valueY,valueX)*180./3.1415926f),180);
            int m =         (int) nearbyintf32(sqrtf(valueY*valueY+valueX*valueX));
            mag.at<int>(y,x) = m;
            ort.at<int>(y,x) = o;
        }
    }


    int step = 180 / nrOrientation;
    int cellPerHeight = image.rows / pixelPerCell;
    int cellPerWidth = image.cols / pixelPerCell;
    int NumberBlocksX= cellPerWidth / cellPerBlock;
    int NumberBlocksY = cellPerHeight / cellPerBlock;
    float hist[cellPerHeight][cellPerWidth][nrOrientation];
    float Blockhistogram[NumberBlocksY][NumberBlocksX][nrOrientation * cellPerBlock*cellPerBlock];

    cout << "================HOG-Algorithmus==================\n";
    cout << "================  Values used  ==================\n";
    cout << "Image width                 : " << image.cols <<"\n";
    cout << "Image height                : " << image.rows <<"\n";
    cout << "Cell width/height in pixel  : " << pixelPerCell <<"\n";
    cout << "Block width/height in cells : " << cellPerBlock <<"\n";
    cout << "Angular bins on 180°        : " << nrOrientation <<"\n";
    cout << "Cells  in X direction       : " << cellPerWidth <<"\n";
    cout << "Cells  in Y direction       : " << cellPerHeight <<"\n";
    cout << "Blocks in X direction       : " << NumberBlocksX <<"\n";
    cout << "Blocks in Y direction       : " << NumberBlocksY <<"\n";
    cout << "================HOG-Algorithmus==================\n";

    for (int i = 0; i < nrOrientation; i++)
    {
        int orientation_start = i * step;
        int orientation_end = (i + 1) * step;
        for (int yCell = 0; yCell < cellPerHeight; yCell++)
        {
            for (int xCell = 0; xCell < cellPerWidth; xCell++)
            {
                int cellPositionY = yCell * pixelPerCell;
                int cellPositionX = xCell * pixelPerCell;
                float value = cell_hog(mag, ort, 0, pixelPerCell,orientation_end, orientation_start,cellPositionY, cellPositionX);
                hist[yCell][xCell][i] = value;
            }
        }
    }
    

    //====================================================================
    //              Copy histogram (cells) into Block array
    //====================================================================
    for (int y = 0; y < NumberBlocksY; y++)
    {
        for (int x = 0; x < NumberBlocksX; x++)
        {
            for (int i = 0; i < nrOrientation; i++)
            {
                Blockhistogram[y][x][i] = hist[y * 2][x * 2][i];
            }
            for (int i = 0; i < nrOrientation; i++)
            {
                Blockhistogram[y][x][i+9] = hist[y * 2][x * 2 + 1][i];
            }
            for (int i = 0; i < nrOrientation; i++)
            {
                Blockhistogram[y][x][i+2*9] = hist[y * 2 + 1][x * 2][i];
            }
            for (int i = 0; i < nrOrientation; i++)
            {
                Blockhistogram[y][x][i+3*9] = hist[y * 2 + 1][x * 2 + 1][i];
            }
        }
    }
    

    //====================================================================
    //                      Normalise Block array
    //        2x2 Cells are one block, each cell has 9 angular bins
    //        which is resulting in 36 values (9bin*4cells)
    //====================================================================
    for (int y = 0; y < NumberBlocksY; y++)
        for (int x = 0; x < NumberBlocksX; x++)
            normalizeCell(Blockhistogram[y][x],36);//4 Block times 9 angular bins


    //==========================================================
    //|                      Visualization                     |
    //|             Show vectors as image on Screen            |
    //==========================================================

    Mat HOGimage(image.size(),CV_32FC1,Scalar(0));
    int radius = pixelPerCell / 2 - 1;
    std::vector<float> dy(nrOrientation);
    std::vector<float> dx(nrOrientation);
    for (int i = 0; i < nrOrientation; i++){
        float alpha = (3.1415926 * (i+0.5)) /(float) nrOrientation;
        dy[i] = radius * cos(alpha);
        dx[i] = radius * sin(alpha);
    }
    float min=+100;
    float max=-100;
    for (int y = 0; y < cellPerHeight; y++)
    {
        for (int x = 0; x < cellPerWidth; x++)
        {
            for (int i = 0; i < nrOrientation; i++)
            {
                float strength = hist[y][x][i];
                max = strength>max?strength:max;
                min = strength<min?strength:min;
                int p1 = y * pixelPerCell + pixelPerCell/2;
                int p2 = x * pixelPerCell + pixelPerCell/2;
                Point sPoint(p2 - dx[i],p1 + dy[i]);
                Point ePoint(p2 + dx[i],p1 - dy[i]);
                line(HOGimage,ePoint,sPoint,Scalar(strength));
            }
        }
    }
    return normalize_self(HOGimage,min,max);


}

struct obj
{
    int x,y;
    float scale;
    int w,h;
};

/**
 * @brief Replace atan2 function for sorting angular bins in 9 bins
 * 
 * @param x Argument 1
 * @param y Argument 2
 * @return int Bin (0 to 8)
 */
int atan2_apr(int y, int x)
{
    if (x == 0 && y == 0)
    {
        return 0;
    }
    if (x == 0)
    {
        // +/- pi/2
        return 4;
    }
    if (y == 0 && x < 0)
    {
        // +/- pi
        return 8;
    }
    float q = (((float)y) / (float)(x));

    if (q < 0)
    {
        if (q >= -0.3639)
        { //-0..-20°=340°
            return 8;
        }
        if (q >= -0.839)
        { //-20...-40°=320
            return 7;
        }
        if (q >= -1.732)
        { //-40..-60°=320
            return 6;
        }
        if (q >= -5.671)
        { //-60..-80°=320
            return 5;
        }
        else
        {
            //-100°=320
            //half of bin
            return 4;
        }
    }
    else
    {

        if (q <= 0.3639)
        {
            //0..20
            return 0;
        }
        if (q <= 0.839)
        {
            //20..40
            return 1;
        }
        if (q <= 1.732)
        {
            //40..60
            return 2;
        }
        if (q <= 5.671)
        {
            //60..80
            return 3;
        }
        else
        {
            //half of bin
            return 4;
        }
    }

}

/**
 * @brief Detects objects (Humans) on the image by using HOG with
 * a SVM modell
 * 
 * @param image Color Image
 * @param pixelPerCell Pixel per cell (Normally 8)
 * @param cellPerBlock Cells per block (normally 2)
 */
std::vector<obj> SVM_Detection_visual(Mat orginalImage,int pixelPerCell,int cellPerBlock,float scale){
    int nrOrientation = 9;
    int height = orginalImage.rows;
    int width = orginalImage.cols;
    static const int BlocksPerWindowX = 4;
    static const int BlocksPerWindowY = 8;
    int step = 180 / nrOrientation;
    int cellPerHeight = orginalImage.rows / pixelPerCell;
    int cellPerWidth = orginalImage.cols / pixelPerCell;
    int NumberBlocksX= cellPerWidth / cellPerBlock;
    int NumberBlocksY = cellPerHeight / cellPerBlock;
    float hist[cellPerHeight][cellPerWidth][nrOrientation];
    float Blockhistogram[NumberBlocksY][NumberBlocksX][nrOrientation * cellPerBlock * cellPerBlock];

    Mat image;
    cvtColor(orginalImage,image,COLOR_BGR2GRAY);

    Mat gx = gradientX(image);
    Mat gy = gradientY(image);
    Mat mag(image.size(),CV_32SC1,cv::Scalar(0));
    Mat ort(image.size(),CV_32SC1,cv::Scalar(0));

    try
    {
        for (int y = 0; y < image.rows; y++)
        {
            for (int x = 0; x < image.cols; x++)
            {
                int valueY = gy.at<int16_t>(y, x);
                int valueX = gx.at<int16_t>(y, x);
                float t1 = atan2f(valueY, valueX);
                float t2 = nearbyintf32(t1* 180. / 3.1415926f);
                int t3 = MODULO((int)t2,180);
                int o = t3;
                int t_apr = atan2_apr(valueY,valueX);
                int o2 = MODULO(t_apr,180);

                int m = (int)nearbyintf32(sqrtf(valueY * valueY + valueX * valueX));
                mag.at<int>(y, x) = m;
                ort.at<int>(y, x) = o;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        cout << "Error magnitude\n";
        cout << e.what() << '\n';
    }


    cout << "================HOG-Algorithm==================\n";
    cout << "================  Values used  ==================\n";
    cout << "Image width                 : " << image.cols <<"\n";
    cout << "Image height                : " << image.rows <<"\n";
    cout << "Cell width/height in pixel  : " << pixelPerCell <<"\n";
    cout << "Block width/height in cells : " << cellPerBlock <<"\n";
    cout << "Angular bins on 180°        : " << nrOrientation <<"\n";
    cout << "Cells  in X direction       : " << cellPerWidth <<"\n";
    cout << "Cells  in Y direction       : " << cellPerHeight <<"\n";
    cout << "Blocks in X direction       : " << NumberBlocksX <<"\n";
    cout << "Blocks in Y direction       : " << NumberBlocksY <<"\n";
    cout << "================HOG-Algorithm==================\n";

    for (int i = 0; i < nrOrientation; i++)
    {
        int orientation_start = i * step;
        int orientation_end = (i + 1) * step;
        for (int yCell = 0; yCell < cellPerHeight; yCell++)
        {
            for (int xCell = 0; xCell < cellPerWidth; xCell++)
            {
                int cellPositionY = yCell * pixelPerCell;
                int cellPositionX = xCell * pixelPerCell;
                float value = cell_hog(mag, ort, 0, pixelPerCell,orientation_end, orientation_start,cellPositionY, cellPositionX);
                hist[yCell][xCell][i] = value;
            }
        }
    }
    

    //====================================================================
    //              Copy histogram (cells) into Block array
    //====================================================================
    for (int y = 0; y < NumberBlocksY; y++)
    {
        for (int x = 0; x < NumberBlocksX; x++)
        {
            for (int i = 0; i < nrOrientation; i++)
            {
                Blockhistogram[y][x][i] = hist[y * 2][x * 2][i];
            }
            for (int i = 0; i < nrOrientation; i++)
            {
                Blockhistogram[y][x][i+9] = hist[y * 2][x * 2 + 1][i];
            }
            for (int i = 0; i < nrOrientation; i++)
            {
                Blockhistogram[y][x][i+2*9] = hist[y * 2 + 1][x * 2][i];
            }
            for (int i = 0; i < nrOrientation; i++)
            {
                Blockhistogram[y][x][i+3*9] = hist[y * 2 + 1][x * 2 + 1][i];
            }
        }
    }
    

    //====================================================================
    //                      Normalise Block array
    //        2x2 Cells are one block, each cell has 9 angular bins
    //        which is resulting in 36 values (9bin*4cells)
    //====================================================================
    for (int y = 0; y < NumberBlocksY; y++)
        for (int x = 0; x < NumberBlocksX; x++)
            normalizeCell(Blockhistogram[y][x],36);//4 Block times 9 angular bins


    float Intercept = -0.27716787;
    float classifier[] = {
    0.15061,-0.20561,-0.10163,-0.14403,0.19473,-0.08734,-0.24039,-0.25564,0.08457,0.36947,-0.05390,0.20649,-0.01303,0.45472,-0.02749,-0.24547,-0.18111,0.11844,0.12782,0.01664,-0.17207,
    -0.17627,0.07035,-0.04300,0.06975,-0.12144,0.37833,0.14161,-0.56742,-0.33982,-0.38881,0.02782,-0.07950,-0.14770,0.18130,0.72184,-0.38000,0.07637,-0.14601,-0.08344,-0.05547,-0.22044,
    -0.25144,-0.41544,0.01831,-0.21176,0.20102,0.25869,0.67906,0.62725,0.14415,0.28656,-0.25994,-0.24768,0.06255,0.02820,-0.36748,-0.19115,-0.26138,-0.28978,-0.24855,-0.32726,0.15861,
    0.03096,0.23729,0.28550,0.24516,-0.03944,-0.02135,0.20046,0.36416,-0.16344,-0.20553,-0.15893,0.28224,0.05471,0.36603,0.18732,0.67779,0.16401,-0.02762,-0.42013,-0.45084,-0.27834,
    -0.28720,-0.00184,-0.05780,0.28143,-0.35335,-0.09157,-0.06391,0.13131,-0.25694,0.14464,-0.15389,0.08866,0.48076,0.44160,0.20401,-0.13376,0.21584,-0.41280,-0.45226,-0.52122,-0.16046,
    0.02111,0.09096,0.37880,0.21958,0.05910,-0.07997,-0.39157,-0.00270,-0.09378,0.10730,-0.26975,-0.05690,0.29329,-0.09263,-0.16304,-0.17994,0.00844,-0.12897,-0.02429,-0.07713,0.45436,
    0.20842,0.27925,0.00467,-0.42610,-0.52203,-0.14553,-0.04626,-0.35423,0.23710,0.15086,-0.04346,-0.27850,-0.11501,0.38754,0.00248,0.06437,0.03442,0.70349,-0.01881,0.07163,-0.02426,
    -0.22356,-0.51733,-0.16107,-0.11250,-0.12637,0.59468,-0.05524,-0.15345,0.43118,0.02020,0.02260,0.12573,0.05908,0.16399,0.19277,0.04157,-0.07942,0.23395,-0.13868,-0.32628,-0.16790,
    -0.34938,-0.33748,-0.03287,0.19630,0.57111,0.76942,0.31399,0.30375,-0.42569,-0.57085,-0.25585,-0.31212,-0.32095,0.03749,0.46881,1.02457,0.40816,0.31363,0.04806,0.18779,-0.11784,
    -0.54192,0.25266,0.36688,0.40857,0.16239,0.15959,0.34200,0.13361,-0.37649,-0.91048,0.04758,0.25972,0.36890,-0.09775,-0.19897,-0.04124,-0.20666,-0.41490,-0.83093,-0.00181,-0.12350,
    -0.24326,-0.46679,-0.09433,0.09589,0.01681,-0.39245,-0.18670,0.47895,0.47463,0.29624,-0.31533,0.14360,0.00931,0.22141,0.41607,-0.33976,0.16809,0.27587,-0.22486,-0.04234,0.53767,
    0.60031,0.33885,0.15610,-0.75500,0.11175,0.16269,-0.11693,-0.61298,-0.13583,-0.05323,-0.00537,-0.29931,-0.87642,-0.17756,-0.34468,-0.17831,-0.07495,-0.16798,0.28318,0.27215,-0.31653,
    -0.03165,0.00180,0.11173,-0.48382,0.21129,-0.06078,0.52308,-0.29042,-0.08423,0.26731,-0.03818,0.00699,-0.24770,-0.40917,0.01404,-0.10611,-0.22628,0.26215,-0.72884,-0.00891,-0.16323,
    0.04899,0.25358,0.10733,0.40549,0.15591,-0.12071,0.20064,-0.14178,0.11236,-0.03662,-0.08727,-0.07374,0.23346,0.03154,0.11459,-0.01679,0.43656,0.45554,-0.26024,0.25936,-0.42192,
    -0.06730,-0.08524,0.05621,0.28535,0.65342,0.19554,-0.29537,-0.36509,-0.24600,-0.34438,-0.20751,0.05593,-0.06606,0.23209,0.08011,-0.19281,-0.41702,-0.14640,-0.10145,-0.15571,-0.07888,
    -0.13398,0.52477,0.29432,-0.03701,-0.29970,0.03003,0.01977,0.28589,-0.20321,0.10609,0.09034,-0.03765,-0.23231,-0.48166,0.12497,0.01405,-0.00276,-0.08392,-0.32610,0.33277,-0.07657,
    0.16183,0.03023,0.04917,0.58044,0.29158,-0.50042,0.16151,0.41716,0.22528,0.08827,-0.03275,0.03031,-0.28248,-0.06440,-0.21791,-0.24336,0.00229,0.18047,-0.07494,0.10416,0.03155,
    -0.14887,-0.08804,-0.40550,-0.51060,0.07286,0.03047,-0.24818,-0.35993,-0.09781,0.34755,0.16061,-0.12724,0.28584,0.28517,0.22053,-0.27460,-0.40289,-0.22043,0.18408,0.77551,0.46806,
    -0.66208,-0.01800,-0.14252,-0.17918,-0.21854,-0.10444,0.14792,-0.03886,-0.43014,0.09054,0.45242,0.09505,-0.29814,-0.40875,-0.22355,0.17812,0.49863,0.39534,-0.25537,-0.19830,0.05335,
    -0.51140,-0.07249,-0.07475,0.04318,0.47229,0.14670,-0.02334,-0.15635,0.14532,-0.44056,0.17488,-0.27462,0.30511,0.48641,-0.02011,-0.00975,0.07081,-0.11006,-0.20908,-0.49847,-0.26718,
    0.20621,0.26726,-0.07913,0.30009,0.18053,0.15382,-0.13132,-0.19727,0.11259,-0.11135,0.20701,0.03820,-0.41658,-0.07070,0.10078,0.02533,-0.30398,-0.10642,-0.10758,-0.06905,0.06586,
    0.00600,0.42179,0.16618,0.05644,-0.26175,-0.09805,0.13550,0.12049,0.02455,-0.29799,-0.14326,0.14258,-0.11303,-0.20884,0.07735,0.15876,-0.06612,-0.02191,0.05371,0.52068,0.16004,
    0.14457,-0.06113,-0.06517,0.06802,-0.20135,-0.11288,0.05443,0.26112,0.53092,0.07352,0.24866,-0.12122,0.15583,0.08337,-0.40760,-0.43088,0.06348,0.29296,-0.30276,-0.02320,0.04421,
    0.29193,0.00818,-0.10986,0.49606,0.18800,0.09269,0.20676,0.49858,0.14835,-0.14732,-0.39255,-0.81161,-0.59078,-0.26908,-0.17609,0.04167,0.38418,0.04761,-0.09426,-0.12132,-0.49113,
    -0.29468,0.20861,0.11970,-0.39876,-0.51486,-0.17617,-0.13875,-0.02885,-0.17142,-0.10563,0.16641,0.39566,-0.29917,-0.47238,0.00445,0.14234,0.27861,0.15078,0.13399,-0.28596,-0.25886,
    -0.08182,0.30080,-0.02163,0.12925,-0.24667,-0.35877,0.11010,0.51346,0.47190,0.02668,0.43639,-0.10089,-0.07690,0.29355,-0.12767,-0.18511,0.02103,-0.24042,-0.19460,-0.39227,0.03134,
    0.14912,0.22485,0.42643,-0.18469,-0.21335,-0.01793,-0.07313,0.22889,-0.02137,0.06345,-0.16291,-0.29852,-0.62934,-0.09357,0.05287,-0.01932,0.05007,-0.13338,0.26845,0.63566,-0.19010,
    -0.58931,-0.00569,0.20760,-0.08547,0.57405,0.29075,-0.01138,0.03307,0.20709,0.02372,0.50727,-0.00658,0.06195,0.32517,0.24052,-0.12602,-0.06914,0.13853,0.15352,0.08118,0.31882,
    -0.18266,-0.14605,0.21579,0.23375,0.00905,-0.39921,0.13071,0.13289,-0.03781,-0.15431,0.26713,0.10011,0.13044,0.03641,-0.00193,-0.10720,-0.34150,-0.46871,-0.22639,-0.26516,-0.10351,
    -0.32943,-0.05940,-0.35913,-0.44365,0.10816,0.11617,-0.00936,0.27891,0.40703,0.65446,-0.09919,-0.43323,-0.24003,-0.16397,0.09566,0.07637,0.20645,-0.15749,-0.13737,-0.02991,-0.60966,
    -0.14596,-0.02431,-0.15543,-0.36722,0.09034,-0.02738,0.20796,0.41362,-0.12467,0.40979,0.13986,0.45794,0.12285,-0.04368,-0.12271,-0.35731,-0.02682,-0.34338,-0.69400,-0.35977,-0.09580,
    0.15665,-0.32232,0.05140,-0.02558,-0.06492,-0.66363,0.21039,0.21137,0.41275,0.31768,0.00031,-0.14851,-0.09921,-0.11464,-0.03597,-0.36547,0.10003,-0.07141,-0.05687,0.16305,-0.07601,
    -0.35207,-0.04872,-0.14556,0.40774,0.31475,0.47697,0.54694,0.34266,0.02010,-0.03953,-0.16871,-0.06126,-0.05739,0.25351,0.15888,-0.03199,0.16820,-0.38947,-0.15067,0.14320,-0.19638,
    -0.10653,0.04062,0.38182,-0.29394,0.03662,0.08070,0.01702,0.18690,0.28779,0.02859,-0.13039,-0.00270,-0.26646,0.24240,-0.22116,-0.22987,-0.19927,-0.16697,-0.08593,0.03427,0.09025,
    -0.06307,0.14658,-0.06994,-0.08160,0.02640,0.14281,-0.13023,-0.03030,-0.14793,0.13766,0.08536,0.19843,-0.00153,0.01022,0.01325,0.56548,0.25720,0.00958,0.07236,-0.16010,-0.07393,
    -0.09191,-0.04373,0.12530,-0.26480,-0.03632,-0.04771,-0.45205,0.06751,-0.02064,-0.13759,-0.22002,0.10632,0.11628,0.28730,-0.16975,-0.16026,0.13469,0.03926,-0.14201,-0.05112,-0.12142,
    0.29757,0.18516,0.07361,-0.26536,-0.43267,-0.10271,0.00684,0.04872,0.16270,0.31249,0.28133,-0.11200,-0.19961,-0.30906,-0.31662,-0.18837,0.00696,-0.17716,-0.23390,0.32704,-0.07771,
    -0.36227,-0.21626,-0.30273,-0.04051,0.31339,0.45018,0.31290,0.10049,-0.09166,0.21943,0.11854,-0.10925,-0.03131,0.11052,-0.03589,0.07520,-0.11690,-0.09860,-0.15463,0.12612,-0.21317,
    -0.08545,-0.05939,-0.08082,-0.03977,-0.06608,0.15571,0.40719,-0.30750,-0.24599,-0.39935,-0.02439,0.02710,0.25368,0.22386,0.14483,0.05384,-0.05795,-0.28935,0.16080,0.19549,0.38870,
    -0.10474,-0.08749,-0.31065,0.04668,0.07316,-0.22841,0.11908,0.03424,0.20779,-0.09982,0.00679,0.15967,-0.20571,-0.47201,-0.33090,-0.09902,-0.13393,0.13444,-0.26548,-0.25157,-0.05952,
    -0.08272,0.95364,0.34405,0.10028,-0.06727,0.18999,0.17795,-0.09180,-0.04128,-0.10997,0.16280,-0.13532,-0.30875,-0.02821,0.07572,-0.03610,-0.07586,0.01200,0.32248,0.52892,0.30525,
    -0.43220,-0.29697,-0.12674,-0.11908,0.32083,0.18181,-0.00200,-0.27043,-0.21426,-0.09474,0.04169,-0.20662,0.09165,0.00830,0.11143,-0.08142,-0.32678,-0.07101,0.22733,0.29215,-0.08420,
    0.18198,0.26595,0.25444,-0.05491,0.26780,-0.02195,-0.14210,0.08659,-0.19947,-0.26252,-0.15424,0.04475,-0.07190,0.30827,0.05048,-0.50238,-0.18053,0.04793,0.23361,0.33925,0.01546,
    -0.00413,-0.45915,-0.35239,-0.47182,-0.16145,0.44889,-0.30472,0.00225,0.09260,-0.22201,-0.38504,-0.21501,-0.22184,-0.02860,-0.02520,0.75408,-0.02458,0.14477,-0.24554,0.06405,-0.25778,
    0.01304,0.36213,0.21060,0.55997,-0.01901,0.11666,-0.35189,-0.04290,-0.48853,0.08970,0.29589,0.26144,0.04527,-0.23390,-0.40229,-0.04941,0.24285,-0.36349,0.17021,0.39827,0.43848,
    -0.14537,-0.07425,-0.35775,-0.39872,-0.02763,0.05717,-0.01448,0.04824,0.25318,-0.00531,0.16651,-0.24214,-0.32379,0.00242,-0.28020,-0.08105,0.25842,0.37101,0.36010,-0.04202,-0.17224,
    -0.48635,0.06277,-0.05232,0.10491,0.15703,0.33872,0.22254,-0.27217,-0.12258,-0.11055,0.25332,0.23127,0.15156,0.17089,0.33239,-0.26404,-0.03557,-0.25429,-0.29501,-0.30487,-0.12740,
    -0.31089,-0.35073,-0.00445,-0.17294,-0.11791,-0.15263,-0.27723,0.10177,0.25559,0.15128,0.30017,0.05138,0.06214,-0.28811,0.16254,0.18888,0.45565,0.31401,-0.07378,-0.05957,-0.08782,
    0.28390,0.18803,0.03068,0.03604,0.54997,0.22911,-0.07103,-0.24963,-0.19509,0.00547,0.07460,0.07179,-0.00057,0.03624,0.29547,-0.14266,-0.26967,-0.34310,-0.05041,-0.17195,-0.12933,
    -0.10928,0.10303,0.01006,-0.02048,0.10774,-0.01338,-0.44393,0.11551,0.02271,0.15203,0.07276,0.21622,-0.31621,-0.38448,0.03270,0.41474,0.22027,0.04806,-0.03155,0.31918,0.09351,
    -0.09233,0.09898,0.43914,-0.04123,0.06489,-0.06991,-0.06090,-0.34402,0.03891,0.13354,0.13767,0.01621,0.23984,0.10163,0.01560,-0.09772,-0.04034,0.01300,-0.12675,-0.00813,-0.00105,
    -0.48172,-0.23557,-0.32899,-0.11371,0.30259,0.01102,-0.29775,-0.38425,-0.23002,-0.03520,0.22928,-0.17529,0.22704,0.06825,-0.20619,-0.07644,-0.01136,0.09134,-0.09683,-0.22659,-0.11945,
    0.06035,-0.06768,-0.34614,0.36296,-0.04446,0.13432,-0.48388,-0.12781,-0.29079,0.03867,0.57512,0.13945,-0.02086,0.02222,0.11141,-0.05064,0.11688,-0.31853,0.24792,0.24054,-0.00482,
    0.08242,-0.17084,-0.15152,-0.36671,-0.01655,0.05739,0.14437,0.33486,0.18020,0.11963,-0.06385,-0.15868,0.35744,0.08016,-0.22290,-0.04115,0.45233,0.12410,-0.31995,0.13341,-0.35013,
    -0.46431,-0.12802,0.17499,0.19542,0.10341,0.11448,-0.18916,-0.09603,-0.09633,-0.23894,-0.01583,0.02400,-0.00281,-0.00409,0.14047,0.09977,-0.24355,-0.09462};


    //====================================================================
    //              Slide the window over the input
    //              picture and calculate the score,
    //              if the value is bigger then a given
    //              threshold, the current window position 
    //              contains the detected object
    //====================================================================

    std::vector<obj> objects;

    for (int windowY = 0; windowY < NumberBlocksY - BlocksPerWindowY; windowY++)
    {//slide the window over the picture
        for (int windowX = 0; windowX < NumberBlocksX - BlocksPerWindowX; windowX++)
        {
            float sum = Intercept;
            for (int y = 0; y < BlocksPerWindowY; y++)
            {
                for (int x = 0; x < BlocksPerWindowX; x++)
                {
                    for (int i = 0; i < 36; i++)
                    {
                        sum += Blockhistogram[y + windowY][x + windowX][i] * classifier[y * 144 + x * 36 + i];//y*4*36+x*36
                    }
                }
            }

            std::string text = std::to_string(windowY) + "|" + std::to_string(windowX) + " = " + std::to_string(sum);
            std::cout << "\r" << text << std::flush;
            if (sum > 0.9)
            {
                obj tmp;
                tmp.x = windowX ;
                tmp.y = windowY ;
                tmp.w =   32 ;
                tmp.h =  128 ;
                tmp.scale = scale;
                objects.push_back(tmp);
            }
        }
    }
    return objects;

}

std::vector<obj> SVM_Detection_visual_apr(Mat orginalImage,int pixelPerCell,int cellPerBlock,float scale){
    const int nrOrientation = 9;
    int height = orginalImage.rows;
    int width = orginalImage.cols;
    static const int BlocksPerWindowX = 4;
    static const int BlocksPerWindowY = 8;
    int step = 180 / nrOrientation;
    int cellPerHeight = orginalImage.rows / pixelPerCell;
    int cellPerWidth = orginalImage.cols / pixelPerCell;
    int NumberBlocksX= cellPerWidth / cellPerBlock;
    int NumberBlocksY = cellPerHeight / cellPerBlock;
    float hist[cellPerHeight][cellPerWidth][nrOrientation];
    float Blockhistogram[NumberBlocksY][NumberBlocksX][nrOrientation * cellPerBlock * cellPerBlock];

    Mat image;
    if (orginalImage.channels()>1){
        cvtColor(orginalImage,image,COLOR_BGR2GRAY);
    }else{
        image = orginalImage;
    }
    

    Mat gx = gradientX(image);
    Mat gy = gradientY(image);
    Mat mag(image.size(),CV_32SC1,cv::Scalar(0));
    Mat ort(image.size(),CV_32SC1,cv::Scalar(0));

    for (int y = 0; y < image.rows; y++)
    {
        for (int x = 0; x < image.cols; x++)
        {
            int valueY = gy.at<int16_t>(y, x);
            int valueX = gx.at<int16_t>(y, x);
            int t_apr = atan2_apr(valueY, valueX);
            //int m = (int)nearbyintf32(sqrtf(valueY * valueY + valueX * valueX));
            int m = abs(valueX) + abs(valueY);
            mag.at<int>(y, x) = m;
            ort.at<int>(y, x) = t_apr;
        }
    }

    for (int i = 0; i < nrOrientation; i++)
    {
        for (int yCell = 0; yCell < cellPerHeight; yCell++)
        {
            for (int xCell = 0; xCell < cellPerWidth; xCell++)
            {
                int cellPositionY = yCell * pixelPerCell;
                int cellPositionX = xCell * pixelPerCell;
                float value = cell_hog1(mag, ort, i,cellPositionY, cellPositionX);
                hist[yCell][xCell][i] = value;
            }
        }
    }

    //====================================================================
    //              Copy histogram (cells) into Block array
    //====================================================================
    for (int y = 0; y < NumberBlocksY; y++)
    {
        for (int x = 0; x < NumberBlocksX; x++)
        {
            for (int i = 0; i < nrOrientation; i++)
            {
                Blockhistogram[y][x][i] = hist[y * 2][x * 2][i];
            }
            for (int i = 0; i < nrOrientation; i++)
            {
                Blockhistogram[y][x][i+9] = hist[y * 2][x * 2 + 1][i];
            }
            for (int i = 0; i < nrOrientation; i++)
            {
                Blockhistogram[y][x][i+2*9] = hist[y * 2 + 1][x * 2][i];
            }
            for (int i = 0; i < nrOrientation; i++)
            {
                Blockhistogram[y][x][i+3*9] = hist[y * 2 + 1][x * 2 + 1][i];
            }
        }
    }

    // for (int y = 0; y < NumberBlocksY; y++)
    // {
    //     for (int x = 0; x < NumberBlocksX; x++)
    //     {
    //         cout << "(";
    //         for (int i = 0; i < nrOrientation * cellPerBlock * cellPerBlock; i++)
    //         {

    //             cout << fixed << setprecision(4) << std::setfill('0') << std::setw(7);
    //             cout << Blockhistogram[y][x][i] << " , ";
    //         }
    //         cout << ")\n";
    //     }
    // }

    //====================================================================
    //                      Normalise Block array
    //        2x2 Cells are one block, each cell has 9 angular bins
    //        which is resulting in 36 values (9bin*4cells)
    //====================================================================
    for (int y = 0; y < NumberBlocksY; y++)
        for (int x = 0; x < NumberBlocksX; x++)
            normalizeCell(Blockhistogram[y][x],36);//4 Block times 9 angular bins




    float Intercept = -0.27716787;
    float classifier[] = {
    0.15061,-0.20561,-0.10163,-0.14403,0.19473,-0.08734,-0.24039,-0.25564,0.08457,0.36947,-0.05390,0.20649,-0.01303,0.45472,-0.02749,-0.24547,-0.18111,0.11844,0.12782,0.01664,-0.17207,
    -0.17627,0.07035,-0.04300,0.06975,-0.12144,0.37833,0.14161,-0.56742,-0.33982,-0.38881,0.02782,-0.07950,-0.14770,0.18130,0.72184,-0.38000,0.07637,-0.14601,-0.08344,-0.05547,-0.22044,
    -0.25144,-0.41544,0.01831,-0.21176,0.20102,0.25869,0.67906,0.62725,0.14415,0.28656,-0.25994,-0.24768,0.06255,0.02820,-0.36748,-0.19115,-0.26138,-0.28978,-0.24855,-0.32726,0.15861,
    0.03096,0.23729,0.28550,0.24516,-0.03944,-0.02135,0.20046,0.36416,-0.16344,-0.20553,-0.15893,0.28224,0.05471,0.36603,0.18732,0.67779,0.16401,-0.02762,-0.42013,-0.45084,-0.27834,
    -0.28720,-0.00184,-0.05780,0.28143,-0.35335,-0.09157,-0.06391,0.13131,-0.25694,0.14464,-0.15389,0.08866,0.48076,0.44160,0.20401,-0.13376,0.21584,-0.41280,-0.45226,-0.52122,-0.16046,
    0.02111,0.09096,0.37880,0.21958,0.05910,-0.07997,-0.39157,-0.00270,-0.09378,0.10730,-0.26975,-0.05690,0.29329,-0.09263,-0.16304,-0.17994,0.00844,-0.12897,-0.02429,-0.07713,0.45436,
    0.20842,0.27925,0.00467,-0.42610,-0.52203,-0.14553,-0.04626,-0.35423,0.23710,0.15086,-0.04346,-0.27850,-0.11501,0.38754,0.00248,0.06437,0.03442,0.70349,-0.01881,0.07163,-0.02426,
    -0.22356,-0.51733,-0.16107,-0.11250,-0.12637,0.59468,-0.05524,-0.15345,0.43118,0.02020,0.02260,0.12573,0.05908,0.16399,0.19277,0.04157,-0.07942,0.23395,-0.13868,-0.32628,-0.16790,
    -0.34938,-0.33748,-0.03287,0.19630,0.57111,0.76942,0.31399,0.30375,-0.42569,-0.57085,-0.25585,-0.31212,-0.32095,0.03749,0.46881,1.02457,0.40816,0.31363,0.04806,0.18779,-0.11784,
    -0.54192,0.25266,0.36688,0.40857,0.16239,0.15959,0.34200,0.13361,-0.37649,-0.91048,0.04758,0.25972,0.36890,-0.09775,-0.19897,-0.04124,-0.20666,-0.41490,-0.83093,-0.00181,-0.12350,
    -0.24326,-0.46679,-0.09433,0.09589,0.01681,-0.39245,-0.18670,0.47895,0.47463,0.29624,-0.31533,0.14360,0.00931,0.22141,0.41607,-0.33976,0.16809,0.27587,-0.22486,-0.04234,0.53767,
    0.60031,0.33885,0.15610,-0.75500,0.11175,0.16269,-0.11693,-0.61298,-0.13583,-0.05323,-0.00537,-0.29931,-0.87642,-0.17756,-0.34468,-0.17831,-0.07495,-0.16798,0.28318,0.27215,-0.31653,
    -0.03165,0.00180,0.11173,-0.48382,0.21129,-0.06078,0.52308,-0.29042,-0.08423,0.26731,-0.03818,0.00699,-0.24770,-0.40917,0.01404,-0.10611,-0.22628,0.26215,-0.72884,-0.00891,-0.16323,
    0.04899,0.25358,0.10733,0.40549,0.15591,-0.12071,0.20064,-0.14178,0.11236,-0.03662,-0.08727,-0.07374,0.23346,0.03154,0.11459,-0.01679,0.43656,0.45554,-0.26024,0.25936,-0.42192,
    -0.06730,-0.08524,0.05621,0.28535,0.65342,0.19554,-0.29537,-0.36509,-0.24600,-0.34438,-0.20751,0.05593,-0.06606,0.23209,0.08011,-0.19281,-0.41702,-0.14640,-0.10145,-0.15571,-0.07888,
    -0.13398,0.52477,0.29432,-0.03701,-0.29970,0.03003,0.01977,0.28589,-0.20321,0.10609,0.09034,-0.03765,-0.23231,-0.48166,0.12497,0.01405,-0.00276,-0.08392,-0.32610,0.33277,-0.07657,
    0.16183,0.03023,0.04917,0.58044,0.29158,-0.50042,0.16151,0.41716,0.22528,0.08827,-0.03275,0.03031,-0.28248,-0.06440,-0.21791,-0.24336,0.00229,0.18047,-0.07494,0.10416,0.03155,
    -0.14887,-0.08804,-0.40550,-0.51060,0.07286,0.03047,-0.24818,-0.35993,-0.09781,0.34755,0.16061,-0.12724,0.28584,0.28517,0.22053,-0.27460,-0.40289,-0.22043,0.18408,0.77551,0.46806,
    -0.66208,-0.01800,-0.14252,-0.17918,-0.21854,-0.10444,0.14792,-0.03886,-0.43014,0.09054,0.45242,0.09505,-0.29814,-0.40875,-0.22355,0.17812,0.49863,0.39534,-0.25537,-0.19830,0.05335,
    -0.51140,-0.07249,-0.07475,0.04318,0.47229,0.14670,-0.02334,-0.15635,0.14532,-0.44056,0.17488,-0.27462,0.30511,0.48641,-0.02011,-0.00975,0.07081,-0.11006,-0.20908,-0.49847,-0.26718,
    0.20621,0.26726,-0.07913,0.30009,0.18053,0.15382,-0.13132,-0.19727,0.11259,-0.11135,0.20701,0.03820,-0.41658,-0.07070,0.10078,0.02533,-0.30398,-0.10642,-0.10758,-0.06905,0.06586,
    0.00600,0.42179,0.16618,0.05644,-0.26175,-0.09805,0.13550,0.12049,0.02455,-0.29799,-0.14326,0.14258,-0.11303,-0.20884,0.07735,0.15876,-0.06612,-0.02191,0.05371,0.52068,0.16004,
    0.14457,-0.06113,-0.06517,0.06802,-0.20135,-0.11288,0.05443,0.26112,0.53092,0.07352,0.24866,-0.12122,0.15583,0.08337,-0.40760,-0.43088,0.06348,0.29296,-0.30276,-0.02320,0.04421,
    0.29193,0.00818,-0.10986,0.49606,0.18800,0.09269,0.20676,0.49858,0.14835,-0.14732,-0.39255,-0.81161,-0.59078,-0.26908,-0.17609,0.04167,0.38418,0.04761,-0.09426,-0.12132,-0.49113,
    -0.29468,0.20861,0.11970,-0.39876,-0.51486,-0.17617,-0.13875,-0.02885,-0.17142,-0.10563,0.16641,0.39566,-0.29917,-0.47238,0.00445,0.14234,0.27861,0.15078,0.13399,-0.28596,-0.25886,
    -0.08182,0.30080,-0.02163,0.12925,-0.24667,-0.35877,0.11010,0.51346,0.47190,0.02668,0.43639,-0.10089,-0.07690,0.29355,-0.12767,-0.18511,0.02103,-0.24042,-0.19460,-0.39227,0.03134,
    0.14912,0.22485,0.42643,-0.18469,-0.21335,-0.01793,-0.07313,0.22889,-0.02137,0.06345,-0.16291,-0.29852,-0.62934,-0.09357,0.05287,-0.01932,0.05007,-0.13338,0.26845,0.63566,-0.19010,
    -0.58931,-0.00569,0.20760,-0.08547,0.57405,0.29075,-0.01138,0.03307,0.20709,0.02372,0.50727,-0.00658,0.06195,0.32517,0.24052,-0.12602,-0.06914,0.13853,0.15352,0.08118,0.31882,
    -0.18266,-0.14605,0.21579,0.23375,0.00905,-0.39921,0.13071,0.13289,-0.03781,-0.15431,0.26713,0.10011,0.13044,0.03641,-0.00193,-0.10720,-0.34150,-0.46871,-0.22639,-0.26516,-0.10351,
    -0.32943,-0.05940,-0.35913,-0.44365,0.10816,0.11617,-0.00936,0.27891,0.40703,0.65446,-0.09919,-0.43323,-0.24003,-0.16397,0.09566,0.07637,0.20645,-0.15749,-0.13737,-0.02991,-0.60966,
    -0.14596,-0.02431,-0.15543,-0.36722,0.09034,-0.02738,0.20796,0.41362,-0.12467,0.40979,0.13986,0.45794,0.12285,-0.04368,-0.12271,-0.35731,-0.02682,-0.34338,-0.69400,-0.35977,-0.09580,
    0.15665,-0.32232,0.05140,-0.02558,-0.06492,-0.66363,0.21039,0.21137,0.41275,0.31768,0.00031,-0.14851,-0.09921,-0.11464,-0.03597,-0.36547,0.10003,-0.07141,-0.05687,0.16305,-0.07601,
    -0.35207,-0.04872,-0.14556,0.40774,0.31475,0.47697,0.54694,0.34266,0.02010,-0.03953,-0.16871,-0.06126,-0.05739,0.25351,0.15888,-0.03199,0.16820,-0.38947,-0.15067,0.14320,-0.19638,
    -0.10653,0.04062,0.38182,-0.29394,0.03662,0.08070,0.01702,0.18690,0.28779,0.02859,-0.13039,-0.00270,-0.26646,0.24240,-0.22116,-0.22987,-0.19927,-0.16697,-0.08593,0.03427,0.09025,
    -0.06307,0.14658,-0.06994,-0.08160,0.02640,0.14281,-0.13023,-0.03030,-0.14793,0.13766,0.08536,0.19843,-0.00153,0.01022,0.01325,0.56548,0.25720,0.00958,0.07236,-0.16010,-0.07393,
    -0.09191,-0.04373,0.12530,-0.26480,-0.03632,-0.04771,-0.45205,0.06751,-0.02064,-0.13759,-0.22002,0.10632,0.11628,0.28730,-0.16975,-0.16026,0.13469,0.03926,-0.14201,-0.05112,-0.12142,
    0.29757,0.18516,0.07361,-0.26536,-0.43267,-0.10271,0.00684,0.04872,0.16270,0.31249,0.28133,-0.11200,-0.19961,-0.30906,-0.31662,-0.18837,0.00696,-0.17716,-0.23390,0.32704,-0.07771,
    -0.36227,-0.21626,-0.30273,-0.04051,0.31339,0.45018,0.31290,0.10049,-0.09166,0.21943,0.11854,-0.10925,-0.03131,0.11052,-0.03589,0.07520,-0.11690,-0.09860,-0.15463,0.12612,-0.21317,
    -0.08545,-0.05939,-0.08082,-0.03977,-0.06608,0.15571,0.40719,-0.30750,-0.24599,-0.39935,-0.02439,0.02710,0.25368,0.22386,0.14483,0.05384,-0.05795,-0.28935,0.16080,0.19549,0.38870,
    -0.10474,-0.08749,-0.31065,0.04668,0.07316,-0.22841,0.11908,0.03424,0.20779,-0.09982,0.00679,0.15967,-0.20571,-0.47201,-0.33090,-0.09902,-0.13393,0.13444,-0.26548,-0.25157,-0.05952,
    -0.08272,0.95364,0.34405,0.10028,-0.06727,0.18999,0.17795,-0.09180,-0.04128,-0.10997,0.16280,-0.13532,-0.30875,-0.02821,0.07572,-0.03610,-0.07586,0.01200,0.32248,0.52892,0.30525,
    -0.43220,-0.29697,-0.12674,-0.11908,0.32083,0.18181,-0.00200,-0.27043,-0.21426,-0.09474,0.04169,-0.20662,0.09165,0.00830,0.11143,-0.08142,-0.32678,-0.07101,0.22733,0.29215,-0.08420,
    0.18198,0.26595,0.25444,-0.05491,0.26780,-0.02195,-0.14210,0.08659,-0.19947,-0.26252,-0.15424,0.04475,-0.07190,0.30827,0.05048,-0.50238,-0.18053,0.04793,0.23361,0.33925,0.01546,
    -0.00413,-0.45915,-0.35239,-0.47182,-0.16145,0.44889,-0.30472,0.00225,0.09260,-0.22201,-0.38504,-0.21501,-0.22184,-0.02860,-0.02520,0.75408,-0.02458,0.14477,-0.24554,0.06405,-0.25778,
    0.01304,0.36213,0.21060,0.55997,-0.01901,0.11666,-0.35189,-0.04290,-0.48853,0.08970,0.29589,0.26144,0.04527,-0.23390,-0.40229,-0.04941,0.24285,-0.36349,0.17021,0.39827,0.43848,
    -0.14537,-0.07425,-0.35775,-0.39872,-0.02763,0.05717,-0.01448,0.04824,0.25318,-0.00531,0.16651,-0.24214,-0.32379,0.00242,-0.28020,-0.08105,0.25842,0.37101,0.36010,-0.04202,-0.17224,
    -0.48635,0.06277,-0.05232,0.10491,0.15703,0.33872,0.22254,-0.27217,-0.12258,-0.11055,0.25332,0.23127,0.15156,0.17089,0.33239,-0.26404,-0.03557,-0.25429,-0.29501,-0.30487,-0.12740,
    -0.31089,-0.35073,-0.00445,-0.17294,-0.11791,-0.15263,-0.27723,0.10177,0.25559,0.15128,0.30017,0.05138,0.06214,-0.28811,0.16254,0.18888,0.45565,0.31401,-0.07378,-0.05957,-0.08782,
    0.28390,0.18803,0.03068,0.03604,0.54997,0.22911,-0.07103,-0.24963,-0.19509,0.00547,0.07460,0.07179,-0.00057,0.03624,0.29547,-0.14266,-0.26967,-0.34310,-0.05041,-0.17195,-0.12933,
    -0.10928,0.10303,0.01006,-0.02048,0.10774,-0.01338,-0.44393,0.11551,0.02271,0.15203,0.07276,0.21622,-0.31621,-0.38448,0.03270,0.41474,0.22027,0.04806,-0.03155,0.31918,0.09351,
    -0.09233,0.09898,0.43914,-0.04123,0.06489,-0.06991,-0.06090,-0.34402,0.03891,0.13354,0.13767,0.01621,0.23984,0.10163,0.01560,-0.09772,-0.04034,0.01300,-0.12675,-0.00813,-0.00105,
    -0.48172,-0.23557,-0.32899,-0.11371,0.30259,0.01102,-0.29775,-0.38425,-0.23002,-0.03520,0.22928,-0.17529,0.22704,0.06825,-0.20619,-0.07644,-0.01136,0.09134,-0.09683,-0.22659,-0.11945,
    0.06035,-0.06768,-0.34614,0.36296,-0.04446,0.13432,-0.48388,-0.12781,-0.29079,0.03867,0.57512,0.13945,-0.02086,0.02222,0.11141,-0.05064,0.11688,-0.31853,0.24792,0.24054,-0.00482,
    0.08242,-0.17084,-0.15152,-0.36671,-0.01655,0.05739,0.14437,0.33486,0.18020,0.11963,-0.06385,-0.15868,0.35744,0.08016,-0.22290,-0.04115,0.45233,0.12410,-0.31995,0.13341,-0.35013,
    -0.46431,-0.12802,0.17499,0.19542,0.10341,0.11448,-0.18916,-0.09603,-0.09633,-0.23894,-0.01583,0.02400,-0.00281,-0.00409,0.14047,0.09977,-0.24355,-0.09462};


    //====================================================================
    //              Slide the window over the input
    //              picture and calculate the score,
    //              if the value is bigger then a given
    //              threshold, the current window position 
    //              contains the detected object
    //====================================================================

    std::vector<obj> objects;
    //slide the window over the picture
    for (int windowY = 0; windowY < NumberBlocksY - BlocksPerWindowY; windowY++)
    {
        for (int windowX = 0; windowX < NumberBlocksX - BlocksPerWindowX; windowX++)
        {
            float sum = Intercept;
            for (int y = 0; y < BlocksPerWindowY; y++)
            {
                for (int x = 0; x < BlocksPerWindowX; x++)
                {
                    for (int i = 0; i < 36; i++)
                    {
                        sum += Blockhistogram[y + windowY][x + windowX][i] * classifier[y * 144 + x * 36 + i];//y*4*36+x*36
                    }
                }
            }

            std::string text = std::to_string(windowY) + "|" + std::to_string(windowX) + " = " + std::to_string(sum);
            std::cout << "\r" << text << std::flush;
            if (sum > 0.9)
            {
                cout << "\nAdded at (" << windowX << " ; " << windowY << ") with " << sum << "\n";
                obj tmp;
                tmp.x = windowX ;
                tmp.y = windowY ;
                tmp.w =   32 ;
                tmp.h =  128 ;
                tmp.scale = scale;
                objects.push_back(tmp);
            }
        }
    }
    return objects;

}

/**
 * @brief Detects objects (Humans) on the image by using HOG with
 * a SVM modell
 * 
 * @param image Color Image
 * @param pixelPerCell Pixel per cell (Normally 8)
 * @param cellPerBlock Cells per block (normally 2)
 * @return cv::Mat  The Matrix to show
 */
Mat SVM_Detection(Mat orginalImage,int pixelPerCell,int cellPerBlock){
    int nrOrientation = 9;
    int height = orginalImage.rows;
    int width = orginalImage.cols;
    int BlocksPerWindowX = 4;
    int BlocksPerWindowY = 8;

    Mat image;
    if (orginalImage.channels()>1){
        cvtColor(orginalImage,image,COLOR_BGR2GRAY);
    }else{
        image = orginalImage;
    }
    Mat gx = gradientX(image);
    Mat gy = gradientY(image);


    Mat mag(image.size(),CV_32SC1,cv::Scalar(0));
    Mat ort(image.size(),CV_32SC1,cv::Scalar(0));
    for (int y = 0; y < image.rows; y++){
        for (int x = 0; x < image.cols; x++){
            int valueY = gy.at<int16_t>(y,x);
            int valueX = gx.at<int16_t>(y,x);
            int o = MODULO( (int) nearbyintf32(atan2f(valueY,valueX)*180./3.1415926f),180);
            int m =         (int) nearbyintf32(sqrtf(valueY*valueY+valueX*valueX));
            mag.at<int>(y,x) = m;
            ort.at<int>(y,x) = o;
        }
    }


    int step = 180 / nrOrientation;
    int cellPerHeight = image.rows / pixelPerCell;
    int cellPerWidth = image.cols / pixelPerCell;
    int NumberBlocksX= cellPerWidth / cellPerBlock;
    int NumberBlocksY = cellPerHeight / cellPerBlock;
    float hist[cellPerHeight][cellPerWidth][nrOrientation];
    float Blockhistogram[NumberBlocksY][NumberBlocksX][nrOrientation * cellPerBlock*cellPerBlock];

    cout << "================HOG-Algorithm==================\n";
    cout << "================  Values used  ==================\n";
    cout << "Image width                 : " << image.cols <<"\n";
    cout << "Image height                : " << image.rows <<"\n";
    cout << "Cell width/height in pixel  : " << pixelPerCell <<"\n";
    cout << "Block width/height in cells : " << cellPerBlock <<"\n";
    cout << "Angular bins on 180°        : " << nrOrientation <<"\n";
    cout << "Cells  in X direction       : " << cellPerWidth <<"\n";
    cout << "Cells  in Y direction       : " << cellPerHeight <<"\n";
    cout << "Blocks in X direction       : " << NumberBlocksX <<"\n";
    cout << "Blocks in Y direction       : " << NumberBlocksY <<"\n";
    cout << "================HOG-Algorithm==================\n";

    for (int i = 0; i < nrOrientation; i++)
    {
        int orientation_start = i * step;
        int orientation_end = (i + 1) * step;
        for (int yCell = 0; yCell < cellPerHeight; yCell++)
        {
            for (int xCell = 0; xCell < cellPerWidth; xCell++)
            {
                int cellPositionY = yCell * pixelPerCell;
                int cellPositionX = xCell * pixelPerCell;
                float value = cell_hog(mag, ort, 0, pixelPerCell,orientation_end, orientation_start,cellPositionY, cellPositionX);
                hist[yCell][xCell][i] = value;
            }
        }
    }
    

    //====================================================================
    //              Copy histogram (cells) into Block array
    //====================================================================
    for (int y = 0; y < NumberBlocksY; y++)
    {
        for (int x = 0; x < NumberBlocksX; x++)
        {
            for (int i = 0; i < nrOrientation; i++)
            {
                Blockhistogram[y][x][i] = hist[y * 2][x * 2][i];
            }
            for (int i = 0; i < nrOrientation; i++)
            {
                Blockhistogram[y][x][i+9] = hist[y * 2][x * 2 + 1][i];
            }
            for (int i = 0; i < nrOrientation; i++)
            {
                Blockhistogram[y][x][i+2*9] = hist[y * 2 + 1][x * 2][i];
            }
            for (int i = 0; i < nrOrientation; i++)
            {
                Blockhistogram[y][x][i+3*9] = hist[y * 2 + 1][x * 2 + 1][i];
            }
        }
    }
    

    //====================================================================
    //                      Normalise Block array
    //        2x2 Cells are one block, each cell has 9 angular bins
    //        which is resulting in 36 values (9bin*4cells)
    //====================================================================
    for (int y = 0; y < NumberBlocksY; y++)
        for (int x = 0; x < NumberBlocksX; x++)
            normalizeCell(Blockhistogram[y][x],36);//4 Block times 9 angular bins




    float Intercept = -0.27716787;
    float classifier[] = {
    0.15061,-0.20561,-0.10163,-0.14403,0.19473,-0.08734,-0.24039,-0.25564,0.08457,0.36947,-0.05390,0.20649,-0.01303,0.45472,-0.02749,-0.24547,-0.18111,0.11844,0.12782,0.01664,-0.17207,
    -0.17627,0.07035,-0.04300,0.06975,-0.12144,0.37833,0.14161,-0.56742,-0.33982,-0.38881,0.02782,-0.07950,-0.14770,0.18130,0.72184,-0.38000,0.07637,-0.14601,-0.08344,-0.05547,-0.22044,
    -0.25144,-0.41544,0.01831,-0.21176,0.20102,0.25869,0.67906,0.62725,0.14415,0.28656,-0.25994,-0.24768,0.06255,0.02820,-0.36748,-0.19115,-0.26138,-0.28978,-0.24855,-0.32726,0.15861,
    0.03096,0.23729,0.28550,0.24516,-0.03944,-0.02135,0.20046,0.36416,-0.16344,-0.20553,-0.15893,0.28224,0.05471,0.36603,0.18732,0.67779,0.16401,-0.02762,-0.42013,-0.45084,-0.27834,
    -0.28720,-0.00184,-0.05780,0.28143,-0.35335,-0.09157,-0.06391,0.13131,-0.25694,0.14464,-0.15389,0.08866,0.48076,0.44160,0.20401,-0.13376,0.21584,-0.41280,-0.45226,-0.52122,-0.16046,
    0.02111,0.09096,0.37880,0.21958,0.05910,-0.07997,-0.39157,-0.00270,-0.09378,0.10730,-0.26975,-0.05690,0.29329,-0.09263,-0.16304,-0.17994,0.00844,-0.12897,-0.02429,-0.07713,0.45436,
    0.20842,0.27925,0.00467,-0.42610,-0.52203,-0.14553,-0.04626,-0.35423,0.23710,0.15086,-0.04346,-0.27850,-0.11501,0.38754,0.00248,0.06437,0.03442,0.70349,-0.01881,0.07163,-0.02426,
    -0.22356,-0.51733,-0.16107,-0.11250,-0.12637,0.59468,-0.05524,-0.15345,0.43118,0.02020,0.02260,0.12573,0.05908,0.16399,0.19277,0.04157,-0.07942,0.23395,-0.13868,-0.32628,-0.16790,
    -0.34938,-0.33748,-0.03287,0.19630,0.57111,0.76942,0.31399,0.30375,-0.42569,-0.57085,-0.25585,-0.31212,-0.32095,0.03749,0.46881,1.02457,0.40816,0.31363,0.04806,0.18779,-0.11784,
    -0.54192,0.25266,0.36688,0.40857,0.16239,0.15959,0.34200,0.13361,-0.37649,-0.91048,0.04758,0.25972,0.36890,-0.09775,-0.19897,-0.04124,-0.20666,-0.41490,-0.83093,-0.00181,-0.12350,
    -0.24326,-0.46679,-0.09433,0.09589,0.01681,-0.39245,-0.18670,0.47895,0.47463,0.29624,-0.31533,0.14360,0.00931,0.22141,0.41607,-0.33976,0.16809,0.27587,-0.22486,-0.04234,0.53767,
    0.60031,0.33885,0.15610,-0.75500,0.11175,0.16269,-0.11693,-0.61298,-0.13583,-0.05323,-0.00537,-0.29931,-0.87642,-0.17756,-0.34468,-0.17831,-0.07495,-0.16798,0.28318,0.27215,-0.31653,
    -0.03165,0.00180,0.11173,-0.48382,0.21129,-0.06078,0.52308,-0.29042,-0.08423,0.26731,-0.03818,0.00699,-0.24770,-0.40917,0.01404,-0.10611,-0.22628,0.26215,-0.72884,-0.00891,-0.16323,
    0.04899,0.25358,0.10733,0.40549,0.15591,-0.12071,0.20064,-0.14178,0.11236,-0.03662,-0.08727,-0.07374,0.23346,0.03154,0.11459,-0.01679,0.43656,0.45554,-0.26024,0.25936,-0.42192,
    -0.06730,-0.08524,0.05621,0.28535,0.65342,0.19554,-0.29537,-0.36509,-0.24600,-0.34438,-0.20751,0.05593,-0.06606,0.23209,0.08011,-0.19281,-0.41702,-0.14640,-0.10145,-0.15571,-0.07888,
    -0.13398,0.52477,0.29432,-0.03701,-0.29970,0.03003,0.01977,0.28589,-0.20321,0.10609,0.09034,-0.03765,-0.23231,-0.48166,0.12497,0.01405,-0.00276,-0.08392,-0.32610,0.33277,-0.07657,
    0.16183,0.03023,0.04917,0.58044,0.29158,-0.50042,0.16151,0.41716,0.22528,0.08827,-0.03275,0.03031,-0.28248,-0.06440,-0.21791,-0.24336,0.00229,0.18047,-0.07494,0.10416,0.03155,
    -0.14887,-0.08804,-0.40550,-0.51060,0.07286,0.03047,-0.24818,-0.35993,-0.09781,0.34755,0.16061,-0.12724,0.28584,0.28517,0.22053,-0.27460,-0.40289,-0.22043,0.18408,0.77551,0.46806,
    -0.66208,-0.01800,-0.14252,-0.17918,-0.21854,-0.10444,0.14792,-0.03886,-0.43014,0.09054,0.45242,0.09505,-0.29814,-0.40875,-0.22355,0.17812,0.49863,0.39534,-0.25537,-0.19830,0.05335,
    -0.51140,-0.07249,-0.07475,0.04318,0.47229,0.14670,-0.02334,-0.15635,0.14532,-0.44056,0.17488,-0.27462,0.30511,0.48641,-0.02011,-0.00975,0.07081,-0.11006,-0.20908,-0.49847,-0.26718,
    0.20621,0.26726,-0.07913,0.30009,0.18053,0.15382,-0.13132,-0.19727,0.11259,-0.11135,0.20701,0.03820,-0.41658,-0.07070,0.10078,0.02533,-0.30398,-0.10642,-0.10758,-0.06905,0.06586,
    0.00600,0.42179,0.16618,0.05644,-0.26175,-0.09805,0.13550,0.12049,0.02455,-0.29799,-0.14326,0.14258,-0.11303,-0.20884,0.07735,0.15876,-0.06612,-0.02191,0.05371,0.52068,0.16004,
    0.14457,-0.06113,-0.06517,0.06802,-0.20135,-0.11288,0.05443,0.26112,0.53092,0.07352,0.24866,-0.12122,0.15583,0.08337,-0.40760,-0.43088,0.06348,0.29296,-0.30276,-0.02320,0.04421,
    0.29193,0.00818,-0.10986,0.49606,0.18800,0.09269,0.20676,0.49858,0.14835,-0.14732,-0.39255,-0.81161,-0.59078,-0.26908,-0.17609,0.04167,0.38418,0.04761,-0.09426,-0.12132,-0.49113,
    -0.29468,0.20861,0.11970,-0.39876,-0.51486,-0.17617,-0.13875,-0.02885,-0.17142,-0.10563,0.16641,0.39566,-0.29917,-0.47238,0.00445,0.14234,0.27861,0.15078,0.13399,-0.28596,-0.25886,
    -0.08182,0.30080,-0.02163,0.12925,-0.24667,-0.35877,0.11010,0.51346,0.47190,0.02668,0.43639,-0.10089,-0.07690,0.29355,-0.12767,-0.18511,0.02103,-0.24042,-0.19460,-0.39227,0.03134,
    0.14912,0.22485,0.42643,-0.18469,-0.21335,-0.01793,-0.07313,0.22889,-0.02137,0.06345,-0.16291,-0.29852,-0.62934,-0.09357,0.05287,-0.01932,0.05007,-0.13338,0.26845,0.63566,-0.19010,
    -0.58931,-0.00569,0.20760,-0.08547,0.57405,0.29075,-0.01138,0.03307,0.20709,0.02372,0.50727,-0.00658,0.06195,0.32517,0.24052,-0.12602,-0.06914,0.13853,0.15352,0.08118,0.31882,
    -0.18266,-0.14605,0.21579,0.23375,0.00905,-0.39921,0.13071,0.13289,-0.03781,-0.15431,0.26713,0.10011,0.13044,0.03641,-0.00193,-0.10720,-0.34150,-0.46871,-0.22639,-0.26516,-0.10351,
    -0.32943,-0.05940,-0.35913,-0.44365,0.10816,0.11617,-0.00936,0.27891,0.40703,0.65446,-0.09919,-0.43323,-0.24003,-0.16397,0.09566,0.07637,0.20645,-0.15749,-0.13737,-0.02991,-0.60966,
    -0.14596,-0.02431,-0.15543,-0.36722,0.09034,-0.02738,0.20796,0.41362,-0.12467,0.40979,0.13986,0.45794,0.12285,-0.04368,-0.12271,-0.35731,-0.02682,-0.34338,-0.69400,-0.35977,-0.09580,
    0.15665,-0.32232,0.05140,-0.02558,-0.06492,-0.66363,0.21039,0.21137,0.41275,0.31768,0.00031,-0.14851,-0.09921,-0.11464,-0.03597,-0.36547,0.10003,-0.07141,-0.05687,0.16305,-0.07601,
    -0.35207,-0.04872,-0.14556,0.40774,0.31475,0.47697,0.54694,0.34266,0.02010,-0.03953,-0.16871,-0.06126,-0.05739,0.25351,0.15888,-0.03199,0.16820,-0.38947,-0.15067,0.14320,-0.19638,
    -0.10653,0.04062,0.38182,-0.29394,0.03662,0.08070,0.01702,0.18690,0.28779,0.02859,-0.13039,-0.00270,-0.26646,0.24240,-0.22116,-0.22987,-0.19927,-0.16697,-0.08593,0.03427,0.09025,
    -0.06307,0.14658,-0.06994,-0.08160,0.02640,0.14281,-0.13023,-0.03030,-0.14793,0.13766,0.08536,0.19843,-0.00153,0.01022,0.01325,0.56548,0.25720,0.00958,0.07236,-0.16010,-0.07393,
    -0.09191,-0.04373,0.12530,-0.26480,-0.03632,-0.04771,-0.45205,0.06751,-0.02064,-0.13759,-0.22002,0.10632,0.11628,0.28730,-0.16975,-0.16026,0.13469,0.03926,-0.14201,-0.05112,-0.12142,
    0.29757,0.18516,0.07361,-0.26536,-0.43267,-0.10271,0.00684,0.04872,0.16270,0.31249,0.28133,-0.11200,-0.19961,-0.30906,-0.31662,-0.18837,0.00696,-0.17716,-0.23390,0.32704,-0.07771,
    -0.36227,-0.21626,-0.30273,-0.04051,0.31339,0.45018,0.31290,0.10049,-0.09166,0.21943,0.11854,-0.10925,-0.03131,0.11052,-0.03589,0.07520,-0.11690,-0.09860,-0.15463,0.12612,-0.21317,
    -0.08545,-0.05939,-0.08082,-0.03977,-0.06608,0.15571,0.40719,-0.30750,-0.24599,-0.39935,-0.02439,0.02710,0.25368,0.22386,0.14483,0.05384,-0.05795,-0.28935,0.16080,0.19549,0.38870,
    -0.10474,-0.08749,-0.31065,0.04668,0.07316,-0.22841,0.11908,0.03424,0.20779,-0.09982,0.00679,0.15967,-0.20571,-0.47201,-0.33090,-0.09902,-0.13393,0.13444,-0.26548,-0.25157,-0.05952,
    -0.08272,0.95364,0.34405,0.10028,-0.06727,0.18999,0.17795,-0.09180,-0.04128,-0.10997,0.16280,-0.13532,-0.30875,-0.02821,0.07572,-0.03610,-0.07586,0.01200,0.32248,0.52892,0.30525,
    -0.43220,-0.29697,-0.12674,-0.11908,0.32083,0.18181,-0.00200,-0.27043,-0.21426,-0.09474,0.04169,-0.20662,0.09165,0.00830,0.11143,-0.08142,-0.32678,-0.07101,0.22733,0.29215,-0.08420,
    0.18198,0.26595,0.25444,-0.05491,0.26780,-0.02195,-0.14210,0.08659,-0.19947,-0.26252,-0.15424,0.04475,-0.07190,0.30827,0.05048,-0.50238,-0.18053,0.04793,0.23361,0.33925,0.01546,
    -0.00413,-0.45915,-0.35239,-0.47182,-0.16145,0.44889,-0.30472,0.00225,0.09260,-0.22201,-0.38504,-0.21501,-0.22184,-0.02860,-0.02520,0.75408,-0.02458,0.14477,-0.24554,0.06405,-0.25778,
    0.01304,0.36213,0.21060,0.55997,-0.01901,0.11666,-0.35189,-0.04290,-0.48853,0.08970,0.29589,0.26144,0.04527,-0.23390,-0.40229,-0.04941,0.24285,-0.36349,0.17021,0.39827,0.43848,
    -0.14537,-0.07425,-0.35775,-0.39872,-0.02763,0.05717,-0.01448,0.04824,0.25318,-0.00531,0.16651,-0.24214,-0.32379,0.00242,-0.28020,-0.08105,0.25842,0.37101,0.36010,-0.04202,-0.17224,
    -0.48635,0.06277,-0.05232,0.10491,0.15703,0.33872,0.22254,-0.27217,-0.12258,-0.11055,0.25332,0.23127,0.15156,0.17089,0.33239,-0.26404,-0.03557,-0.25429,-0.29501,-0.30487,-0.12740,
    -0.31089,-0.35073,-0.00445,-0.17294,-0.11791,-0.15263,-0.27723,0.10177,0.25559,0.15128,0.30017,0.05138,0.06214,-0.28811,0.16254,0.18888,0.45565,0.31401,-0.07378,-0.05957,-0.08782,
    0.28390,0.18803,0.03068,0.03604,0.54997,0.22911,-0.07103,-0.24963,-0.19509,0.00547,0.07460,0.07179,-0.00057,0.03624,0.29547,-0.14266,-0.26967,-0.34310,-0.05041,-0.17195,-0.12933,
    -0.10928,0.10303,0.01006,-0.02048,0.10774,-0.01338,-0.44393,0.11551,0.02271,0.15203,0.07276,0.21622,-0.31621,-0.38448,0.03270,0.41474,0.22027,0.04806,-0.03155,0.31918,0.09351,
    -0.09233,0.09898,0.43914,-0.04123,0.06489,-0.06991,-0.06090,-0.34402,0.03891,0.13354,0.13767,0.01621,0.23984,0.10163,0.01560,-0.09772,-0.04034,0.01300,-0.12675,-0.00813,-0.00105,
    -0.48172,-0.23557,-0.32899,-0.11371,0.30259,0.01102,-0.29775,-0.38425,-0.23002,-0.03520,0.22928,-0.17529,0.22704,0.06825,-0.20619,-0.07644,-0.01136,0.09134,-0.09683,-0.22659,-0.11945,
    0.06035,-0.06768,-0.34614,0.36296,-0.04446,0.13432,-0.48388,-0.12781,-0.29079,0.03867,0.57512,0.13945,-0.02086,0.02222,0.11141,-0.05064,0.11688,-0.31853,0.24792,0.24054,-0.00482,
    0.08242,-0.17084,-0.15152,-0.36671,-0.01655,0.05739,0.14437,0.33486,0.18020,0.11963,-0.06385,-0.15868,0.35744,0.08016,-0.22290,-0.04115,0.45233,0.12410,-0.31995,0.13341,-0.35013,
    -0.46431,-0.12802,0.17499,0.19542,0.10341,0.11448,-0.18916,-0.09603,-0.09633,-0.23894,-0.01583,0.02400,-0.00281,-0.00409,0.14047,0.09977,-0.24355,-0.09462};


    //====================================================================
    //              Slide the window over the input
    //              picture and calculate the score,
    //              if the value is bigger then a given
    //              threshold, the current window position 
    //              contains the detected object
    //====================================================================
    for (int windowY = 0; windowY < NumberBlocksY-BlocksPerWindowY; windowY++)
    {
        for (int windowX = 0; windowX < NumberBlocksX-BlocksPerWindowX; windowX++)
        {
            float sum = Intercept;
            for (int y = 0; y < 8; y++)
            {
                for (int x = 0; x < 4; x++)
                {
                    for (int i = 0; i < 36; i++)
                    {
                        sum += Blockhistogram[y+windowY][x+windowX][i] * classifier[y * 144 + x * 36 + i];
                    }
                }
            }

            std::string text = std::to_string(windowY) + "|" + std::to_string(windowX) + " = " +std::to_string(sum);
            Mat cloneImage = orginalImage.clone();

            putText(cloneImage, std::to_string((int)sum), cv::Point(windowX*16+32,windowY*16+128),FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0,0,255), 1);
            std::cout <<"\r" << text << std::flush;
            if (sum>1){
                cv::rectangle(cloneImage,cv::Rect(windowX*16,windowY*16,32,128),cv::Scalar(0,0,255),1);
                cv::rectangle(orginalImage,cv::Rect(windowX*16,windowY*16,32,128),cv::Scalar(255,0,0),1);
                cout << "\tPerson detected "<<sum<<"\n";
                usleep(50000);
            }else{
                cv::rectangle(cloneImage,cv::Rect(windowX*16,windowY*16,32,128),cv::Scalar(0,255,0),1);
            }
            imshow("int",cloneImage);
            waitKey(1);
            usleep(25000);
        }
    }
}

void test(){
    string imagePath = "/home/dennis/Databases/HOG_database/Graz02_personen/person_188.bmp";
    Mat image = imread(imagePath,IMREAD_GRAYSCALE);
    Mat image2 = imread(imagePath);
    Mat imgScaled,imgScaled2;
    std:vector<obj> allObjects;

    for (int i = 10; i > 1; i--){
        int resizeWidth  = (int)image.cols*i / 10.;
        int resizeHeight = (int)image.rows*i / 10.; 
        cv::resize(image, imgScaled, Size(resizeWidth,resizeHeight), 0., 0., cv::INTER_LINEAR);
        cv::resize(image2, imgScaled2, Size(resizeWidth,resizeHeight), 0., 0., cv::INTER_LINEAR);
        std::cout << "\nNew Size "<< imgScaled.rows << " x " << imgScaled.cols<< " scale "<< i / 10.<<std::endl;
        std::vector<obj> obj_vector = SVM_Detection_visual(imgScaled2,8,2,i/10.);
        //std::vector<obj> obj_vector = SVM_Detection_visual_apr(imgScaled2,8,2,i/10.);
        for (obj o : obj_vector){
            allObjects.push_back(o);
        }
    }
    for (obj o : allObjects){
        int xPos = o.x*16/o.scale;
        int yPos = o.y*16/o.scale;
        int sWidth = 32/o.scale;
        int sHeight = 128/o.scale;
        cout << "Original : x = " << std::setfill('0') << std::setw(3)<< o.x  << " y = " << std::setfill('0') << std::setw(3)<< o.y  << " h = " << o.h << " w = " << o.w<< " scale = " << o.scale << "\n";
        cout << "Scaled   : x = " << xPos << " y = " << yPos << " h = "  <<  sWidth << " w = " << sHeight<<"\n";
        cv::Rect human(xPos,yPos,sWidth,sHeight);
        cv::rectangle(image2,human,cv::Scalar(255,0,0),1);
    }
    imshow("Detected",image2);
    waitKey();
}

int main(int, char**) {
    string imagePath = "/home/dennis/Databases/HOG_database/Graz02_personen/person_188.bmp";
    Mat image = imread(imagePath,IMREAD_GRAYSCALE);
    Mat hog = Draw_HOG_Values(image,9,8,2);
    imwrite("orginal.jpg",image);
    imwrite("hog.jpg",hog);
    test();
    return 0;
}
