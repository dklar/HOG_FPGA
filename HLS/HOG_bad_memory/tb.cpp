#include "hog.hpp"
#include <hls_opencv.h>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;
using namespace cv;

vector<string> split(string s,string delimiter){
    vector<string> result;
    size_t pos = 0;
    std::string token;
    while ((pos = s.find(delimiter)) != std::string::npos) {
        token = s.substr(0, pos);
        result.push_back(token);
        s.erase(0, pos + delimiter.length());
    }
    if (s!="") result.push_back(s);
    return result;
}


void hog(){
	string imagePath = "/home/dennis/Downloads/person/person_188.bmp";
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

        //std::vector<obj> obj_vector = SVM_Detection_visual(imgScaled2,8,2,i/10.);
        std::vector<obj> obj_vector = SVM_Detection_visual_apr(imgScaled2,8,2,i/10.);
        for (obj o : obj_vector){
            allObjects.push_back(o);
        }
    }

    for (obj o : allObjects){

        int xPos = o.x*16/o.scale;
        int yPos = o.y*16/o.scale;
        int sWidth = 32/o.scale;
        int sHeight = 128/o.scale;
        //cout << "Original : x = " << std::setfill('0') << std::setw(3)<< o.x  << " y = " << std::setfill('0') << std::setw(3)<< o.y  << " h = " << o.h << " w = " << o.w<< " scale = " << o.scale << "\n";
        //cout << "Scaled   : x = " << xPos << " y = " << yPos << " h = "  <<  sWidth << " w = " << sHeight<<"\n";
        cv::Rect human(xPos,yPos,sWidth,sHeight);
        cv::rectangle(image2,human,cv::Scalar(255,0,0),1);
    }
    imwrite("result.jpg",image2);


	//hog_picture_old(int16_t *Gx, int16_t *Gy, int nrOrientation,float scale,objects);
}

void hog(string imagePath,string name ){
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

        std::vector<obj> obj_vector = SVM_Detection_visual_apr(imgScaled2,8,2,i/10.);
        for (obj o : obj_vector){
            allObjects.push_back(o);
        }
    }

    for (obj o : allObjects){

        int xPos = o.x*16/o.scale;
        int yPos = o.y*16/o.scale;
        int sWidth = 64/o.scale;
        int sHeight = 128/o.scale;
        cv::Rect human(xPos,yPos,sWidth,sHeight);
        cv::rectangle(image2,human,cv::Scalar(255,0,0),1);
    }
    imwrite(name,image2);


	//hog_picture_old(int16_t *Gx, int16_t *Gy, int nrOrientation,float scale,objects);
}

void test_2(){
	string filepath = "/home/dennis/Schreibtisch/HOG_train_Aug/HOG_Capture/positive.txt";
	ifstream f(filepath.c_str());
	string lineString;
	int i = 0;
	while (std::getline(f, lineString)) {
	  vector<string> args = split(lineString,"|");
	  string fileName = args[0];
	  cout << "Test with " << fileName<<"\n";
	  string resultName = to_string(i)+".jpg";
	  hog(fileName,resultName);
	  i++;
	}

}

void test_acc(){
	string filepath = "/home/dennis/Schreibtisch/HOG_train_Aug/HOG_Capture/positive.txt";
	ifstream f(filepath.c_str());
	string lineString;
	int i = 0;
	while (std::getline(f, lineString)) {
	  vector<string> args = split(lineString,"|");
	  string fileName = args[0];
	  cout << "Test with " << fileName<<"\n";
	  string resultName = to_string(i)+"_acc.jpg";
	  hog(fileName,resultName);
	  i++;
	}
}

int main(){

	//hog();
	test_2();
	return 0;
}
