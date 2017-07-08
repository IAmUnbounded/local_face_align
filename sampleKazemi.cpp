/*
By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install,
copy or use the software.

                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2013, OpenCV Foundation, all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall copyright holders or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/
#include "face_alignment.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "tbb/tbb.h"
using namespace std;
using namespace cv;
using namespace cv::face;
void readSplit(ifstream& is, splitr &vec)
{
    is.read((char*)&vec, sizeof(splitr));
}
void readLeaf(ifstream& is, vector<Point2f> &leaf)
{
    unsigned long size;
    is.read((char*)&size, sizeof(size));
    leaf.resize(size);
    is.read((char*)&leaf[0], leaf.size() * sizeof(Point2f));
}

bool load(){
    ifstream f("model1.dat",ios::binary);
    size_t len;
    f.read((char*)&len, sizeof(size_t));
    char* temp = new char[len+1];
    f.read(temp, len);
    temp[len] = '\0';
    string s(temp);
    delete [] temp;
    cout<<s<<endl;
    if(s.compare("cascade_depth")!=0){
        cout<<"data not saved properly"<<endl;
        return false;
    }
    unsigned long cascade_size;
    f.read((char*)&cascade_size,sizeof(cascade_size));
    cout<<cascade_size<<endl;
    vector< vector<regtree> > forests;
    forests.resize(cascade_size);
    f.read((char*)&len, sizeof(size_t));
    char* temp1 = new char[len+1];
    f.read(temp1, len);
    temp1[len] = '\0';
    s =string(temp1);
    delete [] temp1;
    cout<<s<<endl;
    if(s.compare("num_trees")!=0){
        //cout<<s<<endl;
        cout<<"data not saved properly"<<endl;
        return false;
    }
    unsigned long num_trees;
    f.read((char*)&num_trees,sizeof(num_trees));
    for(unsigned long i=0;i<cascade_size;i++){
        for(unsigned long j=0;j<num_trees;j++){
            regtree tree;
            f.read((char*)&len, sizeof(size_t));
            char* temp2 = new char[len+1];
            f.read(temp2, len);
            temp2[len] = '\0';
            s =string(temp2);
            delete [] temp2;
            cout<<s<<endl;
            if(s.compare("num_nodes")!=0){
                cout<<s<<endl;
                cout<<"data not loaded properly"<<endl;
                return false;
            }
            unsigned long num_nodes;
            f.read((char*)&num_nodes,sizeof(num_nodes));
            tree.nodes.resize(num_nodes);
            for(unsigned long k=0;k<num_nodes;k++){
                f.read((char*)&len, sizeof(size_t));
                char* temp3 = new char[len+1];
                f.read(temp3, len);
                temp3[len] = '\0';
                s =string(temp3);
                delete [] temp3;
                cout<<s<<endl;
                tree_node node;
                if(s.compare("split")==0){
                    splitr split;
                    readSplit(f,split);
                    node.split = split;
                    cout<<split.index1<<" "<<split.index2<<" "<<split.thresh<<endl;
                    node.leaf.clear();
                }
                else if(s.compare("leaf")==0){
                    vector<Point2f> leaf;
                    readLeaf(f,leaf);
                    node.leaf = leaf;
                    for(unsigned long z=0;z<leaf.size();z++)
                        cout<<leaf[z]<<" ";
                    waitKey(0);
                }
                else{
                    cout<<s<<endl;
                    cout<<"Data not loaded properly"<<endl;
                    return false;
                }
                tree.nodes.push_back(node);
            }
            forests[i].push_back(tree);
        }
    }
    f.close();
    return true;
}

int main(int argc,char** argv){
    //Give the path to the directory containing all the files containing data
   CommandLineParser parser(argc, argv,
        "{ help h usage ? |      | give the following arguments in following format }"
        "{ annotations a  |.     | (required) path to annotations txt file [example - /data/annotations.txt] }"
        "{ config c       |      | (required) path to configuration xml file containing parameters for training.[example - /data/config.xml] }"
        "{ width w        |  460 | The width which you want all images to get to scale the annotations. large images are slow to process [default = 460] }"
        "{ height h       |  460 | The height which you want all images to get to scale the annotations. large images are slow to process [default = 460] }"
        "{ face_cascade f |      | Path to the face cascade xml file which you want to use as a detector}"
    );
    clock_t begin = clock();
    // Read in the input arguments
    if (parser.has("help")){
        parser.printMessage();
        cerr << "TIP: Use absolute paths to avoid any problems with the software!" << endl;
        return 0;
    }
    string directory(parser.get<string>("annotations"));
    //default initialisation
    Size scale(460,460);
    scale = Size(parser.get<int>("width"),parser.get<int>("height"));
    if (directory.empty()){
        parser.printMessage();
        cerr << "TIP: Use absolute paths to avoid any problems with the software!" << endl;
        return -1;
    }
    string configfile_name(parser.get<string>("config"));
    if (configfile_name.empty()){
        parser.printMessage();
        cerr << "TIP: Use absolute paths to avoid any problems with the software!" << endl;
        return -1;
    }
    string cascade_name(parser.get<string>("face_cascade"));
    if (cascade_name.empty()){
        parser.printMessage();
        cerr << "TIP: Use absolute paths to avoid any problems with the software!" << endl;
        return -1;
    }
    //create a vector to store names of files in which annotations
    // and image names are found
    /*The format of the file containing annotations should be of following format
        /data/abc/abc.jpg
        123.45,345.65
        321.67,543.89

        The above format is similar to HELEN dataset which is used for training model 
     */
    vector<String> filenames;
    //reading the files from the given directory
    glob(directory,filenames);
    //create a vector to store image names 
    vector<String> imagenames;
    //create object to get landmarks
    vector< vector<Point2f> > trainlandmarks;
    //create a pointer to call the base class
    //pass the face cascade xml file which you want to pass as a detector
    CascadeClassifier face_cascade;
    face_cascade.load(cascade_name);
    Ptr<FacemarkKazemi> facemark= createFacemarkKazemi(face_cascade);
    //gets landmarks and corresponding image names in both the vectors
    facemark->getData(filenames,trainlandmarks,imagenames);
    //vector to store images
    vector<Mat> trainimages;
    for(unsigned long i=0;i<imagenames.size();i++){
        string imgname = imagenames[i].substr(0, imagenames[i].size()-1);
        string img =string(imgname);
        Mat src = imread(img);
        if(src.empty()){
            cerr<<string("Image"+img+"not found\n.Aborting...")<<endl;
            return 0;
        }
        trainimages.push_back(src);
    }
    cout<<"Got data"<<endl;
    //Now scale data according to the size selected
    facemark->scaleData(trainlandmarks,trainimages,scale);
    //calculate mean shape
    cout<<"Scaled data"<<endl;
    facemark->calcMeanShape(trainlandmarks,trainimages);
    cout<<"Got mean shape"<<endl;
    /*Now train data using training function which is yet to be built*/
    cout<<"Training started .. . ."<<endl;
    facemark->train(trainimages,trainlandmarks,configfile_name);
    cout<<"Training complete"<<endl;
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout<<elapsed_secs<<endl;
    load();
    return 0; 
}
