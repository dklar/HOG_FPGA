#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <iomanip>
#include <fstream>

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

Mat gradientY(Mat pictureIn){
    Mat gradient(pictureIn.size(),CV_16SC1,cv::Scalar(0));
    for (int y = 1; y < pictureIn.rows-1; y++)
    {
        for (int x = 0; x < pictureIn.cols; x++)
        {
            gradient.at<int16_t>(y,x) = 
                            pictureIn.at<uint8_t>(y+1,x) -
                            pictureIn.at<uint8_t>(y-1,x);
        }
    }
    return gradient;
}

Mat gradientX(Mat pictureIn){
    Mat gradient(pictureIn.size(),CV_16SC1,cv::Scalar(0));
    for (int y = 0; y < pictureIn.rows; y++)
    {
        for (int x = 0; x < pictureIn.cols-2; x++)
        {
            gradient.at<int16_t>(y,x+1) = 
                        pictureIn.at<uint8_t>(y,x+2) -
                        pictureIn.at<uint8_t>(y,x);
        }
    }
    return gradient;
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

float cell_hog(Mat magnitude,Mat orientation, 
            int start, int stop,int oStart, int oEnd,
            int r_index, int c_index,int pixelPerCell=8)
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

/**
 * @brief Save the calculated values to a file
 * 
 * @param image Image to adapt HOG
 * @param filename Filename to save
 * @param nrOrientation Number of Orientation per Bin (Normally 9)
 * @param pixelPerCell Pixel per cell (Normally 8)
 * @param cellPerBlock Cells per block (normally 2)
 */
void Save_HOG_Values(Mat image,string filename,int nrOrientation,int pixelPerCell,int cellPerBlock){
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
    cout << "Size of feature vector      : " << NumberBlocksX *NumberBlocksY *36 <<" values\n";
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

    ofstream file;
    file.open(filename,ios::app);
    for (int y = 0; y < NumberBlocksY; y++){
        for (int x = 0; x < NumberBlocksX; x++){
            for (int i = 0; i < 36; i++){
                file << Blockhistogram[y][x][i]<<";";
            }
        }
    }
    file << "\n";
    file.close();
}

void Save_HOG_Values_apr(Mat image,string filename,int nrOrientation,int pixelPerCell,int cellPerBlock){
    Mat gx = gradientX(image);
    Mat gy = gradientY(image);

    Mat mag(image.size(),CV_32SC1,cv::Scalar(0));
    Mat ort(image.size(),CV_32SC1,cv::Scalar(0));

    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            int valueY = gy.at<int16_t>(y, x);
            int valueX = gx.at<int16_t>(y, x);
            int t_apr = atan2_apr(valueY, valueX);
            int m = abs(valueX) + abs(valueY);
            mag.at<int>(y, x) = m;
            ort.at<int>(y, x) = t_apr;
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
    cout << "Size of feature vector      : " << NumberBlocksX *NumberBlocksY *36 <<" values\n";
    cout << "================HOG-Algorithmus==================\n";
    for (int i = 0; i < nrOrientation; i++) {
        for (int yCell = 0; yCell < cellPerHeight; yCell++) {
            for (int xCell = 0; xCell < cellPerWidth; xCell++) {
                int cellPositionY = yCell * pixelPerCell;
                int cellPositionX = xCell * pixelPerCell;
                float value = cell_hog1(mag, ort, i, cellPositionY, cellPositionX);
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


    ofstream file;
    file.open(filename,ios::app);
    for (int y = 0; y < NumberBlocksY; y++){
        for (int x = 0; x < NumberBlocksX; x++){
            for (int i = 0; i < 36; i++){
                file << Blockhistogram[y][x][i]<<";";
            }
        }
    }
    file << "\n";
    file.close();
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
    cout << "Size of feature vector      : " << NumberBlocksX *NumberBlocksY *36 <<" values\n";
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
    //|                      Visualisation                     |
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