#include "hog.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

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

void readSampleData(string path){
    for (int i = 0; i < 3000; i++)
    {
        string name = path + "pos_" + to_string(i)+".jpg";
        fstream f(name.c_str());
        if (f.good()){
            Mat image = imread(name,IMREAD_GRAYSCALE);
            Save_HOG_Values_apr(image,"positive_apr_L2.data",9,8,2,"L2");
            Save_HOG_Values(image,"positive_acc_L2.data",9,8,2,"L2");
        }else{
            break;
        }
    }
    for (int i = 0; i < 3000; i++)
    {
        string name = path + "neg_" + to_string(i)+".jpg";
        fstream f(name.c_str());
        if (f.good()){
            Mat image = imread(name,IMREAD_GRAYSCALE);
            Save_HOG_Values_apr(image,"negative_apr_L2.data",9,8,2,"L2");
            Save_HOG_Values(image,"negative_acc_L2.data",9,8,2,"L2");
        }else{
            break;
        }
    }

}

int main(int, char**) {
    std::cout << "Delete old files...\n";
    remove("positive_acc.data");
    remove("positive_apr.data");
    remove("negative_acc.data");
    remove("negative_apr.data");
    std::cout << "Create new ones...\n";
    //string path = "/home/dennis/Schreibtisch/HOG_train_Aug/HOG_Capture/Database/";
    string path = "/home/dennis/Schreibtisch/HOG_FPGA/HOG_Capture/Database/";
    readSampleData(path);
    std::cout << "Done...\n";
    return 0;
}
