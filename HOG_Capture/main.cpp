#include <iostream>
#include <fstream>
#include <cstdint>
#include <cmath>
#include <vector>
#include <iomanip>
#include <unistd.h>
#include <fstream>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;


bool selectObject = false;
Rect selection;
Point origin;
Mat image, canvas;
String win_name = "Select area with objects...";
vector<Rect> selections;


cv::Point startpoint(0,0);

static bool showSelections()
{
    for (size_t i = 0; i< selections.size(); i++)
    {
        rectangle(canvas, selections[i], Scalar(0, 255, 0), 2);
    }
    imshow(win_name, canvas);
    return true;
}

static void onMouse(int event, int x, int y, int, void*)
{
    switch (event)
    {
    case EVENT_LBUTTONDOWN:
        origin = Point(x, y);
        selectObject = true;
        cout << "Mouse down" << x <<" "<<y<<endl;
        break;
    case EVENT_LBUTTONUP:
    {
        selectObject = false;
        selections.push_back(selection);
        showSelections();
        cout << "Mouse release" << x <<" "<<y<<endl;
        break; 
    }
    }

    if (selectObject)
    {
        selection.x = MIN(x, origin.x);
        selection.y = MIN(y, origin.y);
        selection.width = std::abs(x - origin.x) + 1;
        selection.height = std::abs(y - origin.y) + 1;
        selection &= Rect(0, 0, image.cols, image.rows);

        if (selection.width > 0 && selection.height > 0)
        {
            Mat canvascopy;
            canvas.copyTo(canvascopy);
            Mat roi = canvascopy(selection);
            bitwise_not(roi, roi);
            imshow(win_name, canvascopy);
        }
    }
}

int main(int argc, char *argv[])
{
    cout << "Select objects...\nPress [ESC] for close and save\nd for delete last object\ns for save objects\n";

    String file_name_picture    = argv[1];
    String file_name_table      = argv[2];

    cout << "picture to read    : " << file_name_picture <<std::endl;
    cout << "database to write  : " << file_name_table << endl;

    ofstream myfile;
    myfile.open(file_name_table, std::ios::app);

    image = imread(file_name_picture);
    image.copyTo(canvas);

    namedWindow(win_name);
    setMouseCallback(win_name, onMouse);

    imshow(win_name, image);

    while (true)
    {
        int key = waitKey(0);
        if (key == 27)
        {
            for (size_t i = 0; i < selections.size(); i++)
            {
                String saveVal = file_name_picture + "|" + to_string(selections[i].x) + "|" + to_string(selections[i].y) + "|" + to_string(selections[i].width) + "|" + to_string(selections[i].height);
                myfile << saveVal << std::endl;
                std::cout << "wrote area to file..\n";
            }
            myfile.close();
            break;
        }
        if (key == 's'){
            for (size_t i = 0; i < selections.size(); i++){
                String saveVal = file_name_picture + "|" + to_string(selections[i].x) + "|" + to_string(selections[i].y)
                            + "|" + to_string(selections[i].width) + "|" + to_string(selections[i].height);
                myfile << saveVal << std::endl;
                std::cout << "wrote area to file..\n";
            }
        }

        if ((key == 'd') & (selections.size() > 0)){
            selections.erase(selections.end() - 1);
            image.copyTo(canvas);
            showSelections();
        }
    }

    return 0;
}