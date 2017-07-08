/*
By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2000-2015, Intel Corporation, all rights reserved.
Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
Copyright (C) 2009-2015, NVIDIA Corporation, all rights reserved.
Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
Copyright (C) 2015, OpenCV Foundation, all rights reserved.
Copyright (C) 2015, Itseez Inc., all rights reserved.
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
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

#include "face_alignment.hpp"
#include "tbb/tbb.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <map>
#include <vector>
#include <cmath>
#include <vector>
#include <queue>
#include <algorithm>

using namespace std;
namespace cv{
namespace face{
//This function initialises the training parameters.
bool FacemarkKazemiImpl::setTrainingParameters(string filename){
    cout << "Reading Training Parameters " << endl;
    FileStorage fs;
    fs.open(filename, FileStorage::READ);
    if (!fs.isOpened())
    {   String error_message = "Error while opening configuration file.Aborting..";
        CV_Error(Error::StsBadArg, error_message);
        return false;
    }
    int cascade_depth_;
    int tree_depth_;
    int num_trees_per_cascade_level_;
    float learning_rate_;
    int oversampling_amount_;
    int num_test_coordinates_;
    float lambda_;
    int num_test_splits_;
    fs["cascade_depth"]>> cascade_depth_;
    fs["tree_depth"]>> tree_depth_;
    fs["num_trees_per_cascade_level"] >> num_trees_per_cascade_level_;
    fs["learning_rate"] >> learning_rate_;
    fs["oversampling_amount"] >> oversampling_amount_;
    fs["num_test_coordinates"] >> num_test_coordinates_;
    fs["lambda"] >> lambda_;
    fs["num_test_splits"] >> num_test_splits_;
    cascade_depth = (unsigned long)cascade_depth_;
    tree_depth = (unsigned long) tree_depth_;
    num_trees_per_cascade_level = (unsigned long) num_trees_per_cascade_level_;
    learning_rate = (float) learning_rate_;
    oversampling_amount = (unsigned long) oversampling_amount_;
    num_test_coordinates = (unsigned  long) num_test_coordinates_;
    lambda = (float) lambda_;
    num_test_splits = (unsigned long) num_test_splits_;
    fs.release();
    cout<<"Parameters loaded"<<endl;
    return true;
}
void FacemarkKazemiImpl::getTestCoordinates (vector< vector<Point2f> >& pixel_coordinates,float min_x,
                                        float min_y, float max_x , float max_y)
{
    for (unsigned long i = 0; i < cascade_depth; ++i){
        vector<Point2f> temp;
        RNG rng(time(0));
        for (unsigned long j = 0; j < num_test_coordinates; ++j)
        {
            Point2f pt;
            pt.x = (float)rng.uniform(min_x,max_x);
            pt.y = (float)rng.uniform(min_y,max_y);
            temp.push_back(pt);
        }
        pixel_coordinates.push_back(temp);
    }
}

bool FacemarkKazemiImpl :: getRelativePixels(training_sample& sample_,vector<Point2f> pixel_coordinates_){
    vector<Point2f> sample = sample_.current_shape; 
    if(sample.size()!=meanshape.size()){
        String error_message = "Error while finding relative shape. Aborting....";
        CV_Error(Error::StsBadArg, error_message);
        return false;
    }
    float minx=8000.0,maxx=0.0,miny=8000.0,maxy=0.0;
    Point2f srcTri[3];
    srcTri[0] = Point2f(minmeanx , minmeany );
    srcTri[1] = Point2f( maxmeanx, minmeany );
    srcTri[2] = Point2f( minmeanx, maxmeany );
     for (vector<Point2f>::iterator it = sample.begin(); it != sample.end(); it++)
    {
        Point2f pt1;
        pt1.x=(*it).x;
        if(pt1.x<minx)
            minx=pt1.x;
        if(pt1.x>maxx)
            maxx=pt1.x;
        pt1.y=(*it).y;
        if(pt1.y<miny)
            miny=pt1.y;
        if(pt1.y>maxx)
            maxy=pt1.y;
    }
    Point2f dstTri[3];
    dstTri[0] = Point2f(minx , miny );
    dstTri[1] = Point2f( maxx, miny );
    dstTri[2] = Point2f( minx, maxy );
    Mat warp_mat( 2, 3, CV_32FC1 );
    //get affine transform to calculate warp matrix
    warp_mat = getAffineTransform( srcTri, dstTri );
    //loop to initialize initial shape
    for (vector<Point2f>::iterator it = pixel_coordinates_.begin(); it != pixel_coordinates_.end(); it++) {
        //unsigned long index = getNearestLandmark(*it);
        Point2f pt = (*it);
        Mat C = (Mat_<double>(3,1) << pt.x, pt.y, 1);
        Mat D =warp_mat*C;
        pt.x=float(abs(D.at<double>(0,0)));
        pt.y=float(abs(D.at<double>(1,0)));
        sample_.pixel_coordinates.push_back(pt);
    }
    return true;
}
bool FacemarkKazemiImpl::getPixelIntensities(training_sample &sample){
    if(sample.pixel_coordinates.size()==0){
        String error_message = "No pixel coordinates found. Aborting.....";
        CV_Error(Error::StsBadArg, error_message);
        return false;
    }
    Mat img = sample.image;
    //convert image to graysscale
    cvtColor(img,img,COLOR_BGR2GRAY);
    //for(unsigned long i=0;i<sample.pixel_coordinates.size();i++)
        //cv::circle(img,sample.pixel_coordinates[i],5,cv::Scalar(0,0,255),CV_FILLED);
    //namedWindow("pappu");
    //imshow("pappu",img);
    //waitKey(0);
    for(unsigned long j=0;j<sample.pixel_coordinates.size();j++){
        sample.pixel_intensities.push_back((int)img.at<uchar>(sample.pixel_coordinates[j]));
    }
    return true;
}
unsigned long FacemarkKazemiImpl::  getNearestLandmark(Point2f pixel)
{
    if(meanshape.empty()) {
            // throw error if no data (or simply return -1?)
            String error_message = "The data is not loaded properly by train function. Aborting...";
            CV_Error(Error::StsBadArg, error_message);
            return false;
    }
    float dist=float(1000000009.00);
    unsigned long index =0;
    for(unsigned long i=0;i<meanshape.size();i++){
        Point2f pt = pixel-meanshape[i];
        if(sqrt(pt.x*pt.x+pt.y*pt.y)<dist){
            dist=sqrt(pt.x*pt.x+pt.y*pt.y);
            index = i;
        }
    }
    return index;
}
vector<regtree> FacemarkKazemiImpl::gradientBoosting(vector<training_sample>& samples,vector<Point2f> pixel_coordinates){
    vector<regtree> forest;
    for(unsigned long i=0;i<num_trees_per_cascade_level;i++){
            cout<<"Fit "<<i<<" trees"<<endl;
            regtree tree;
            buildRegtree(tree,samples,pixel_coordinates);
            forest.push_back(tree);
    }
    return forest;
}
bool FacemarkKazemiImpl:: getRelativeShape(training_sample& sample){
    if(sample.actual_shape.size()!=sample.current_shape.size()){
        String error_message = "Error while finding relative shape. Aborting....";
        CV_Error(Error::StsBadArg, error_message);
        return false;
    }
    float minx=8000.0,maxx=0.0,miny=8000.0,maxy=0.0;
    Point2f srcTri[3];
    for (vector<Point2f>::iterator it = sample.current_shape.begin(); it != sample.current_shape.end(); it++)
    {
        Point2f pt1;
        pt1.x=(*it).x;
        if(pt1.x<minx)
            minx=pt1.x;
        if(pt1.x>maxx)
            maxx=pt1.x;
        pt1.y=(*it).y;
        if(pt1.y<miny)
            miny=pt1.y;
        if(pt1.y>maxx)
            maxy=pt1.y;
    }
    //source points to find warp matrix
    srcTri[0] = Point2f(minx , miny );
    srcTri[1] = Point2f( maxx, miny );
    srcTri[2] = Point2f( minx, maxy );
    minx=8000.0;maxx=0.0;miny=8000.0;maxy=0.0;
    for (vector<Point2f>::iterator it = sample.actual_shape.begin(); it != sample.actual_shape.end(); it++)
    {
        Point2f pt1;
        pt1.x=(*it).x;
        if(pt1.x<minx)
            minx=pt1.x;
        if(pt1.x>maxx)
            maxx=pt1.x;
        pt1.y=(*it).y;
        if(pt1.y<miny)
            miny=pt1.y;
        if(pt1.y>maxx)
            maxy=pt1.y;
    }
    Point2f dstTri[3];
    dstTri[0] = Point2f(minx , miny );
    dstTri[1] = Point2f( maxx, miny );
    dstTri[2] = Point2f( minx, maxy );
    Mat warp_mat( 2, 3, CV_32FC1 );
    //get affine transform to calculate warp matrix
    warp_mat = getAffineTransform( srcTri, dstTri );
    //loop to initialize initial shape
    for (vector<Point2f>::iterator it = sample.current_shape.begin(); it !=sample.current_shape.end(); it++) {
        Point2f pt = (*it);
        Mat C = (Mat_<double>(3,1) << pt.x, pt.y, 1);
        Mat D =warp_mat*C;
        pt.x=float(abs(D.at<double>(0,0)));
        pt.y=float(abs(D.at<double>(1,0)));
        (*it)=pt;
    }
    return true;
}
class getDiffShape : public ParallelLoopBody
{
    public:
        getDiffShape(vector<training_sample>* samples_) :
        samples(samples_)
        {
        }
        virtual void operator()( const cv::Range& range) const
        {
            for (size_t j = range.start; j < range.end; ++j){
                (*samples)[j].shapeResiduals.resize((*samples)[j].current_shape.size());
                for(unsigned long k=0;k<(*samples)[j].current_shape.size();k++)
                    (*samples)[j].shapeResiduals[k]=(*samples)[j].actual_shape[k]-(*samples)[j].current_shape[k];
            }
        }
    private:
        vector<training_sample>* samples;
};
bool FacemarkKazemiImpl::createTrainingSamples(vector<training_sample> &samples,vector<Mat> images,vector< vector<Point2f> > landmarks){
    RNG rng;
    unsigned long in=0;
    samples.resize(oversampling_amount*images.size());
    for(unsigned long i=0;i<images.size();i++){
        for(unsigned long j=0;j<oversampling_amount;j++){
        //make the splits generated from randomly_generate_split function
            unsigned long rindex =(unsigned long)rng.uniform(0,(int)images.size()-1);
            samples[in].image=images[i];
            samples[in].actual_shape = landmarks[i]; 
            if(i!=rindex){
                samples[in].current_shape = landmarks[rindex];
                in++;
            }
            else{
                samples[in].current_shape=meanshape;
                in++;
            }
        }
    }
    tbb::parallel_for(
    tbb::blocked_range<std::vector<training_sample>::iterator>(samples.begin(),samples.end()),
    [&] (tbb::blocked_range<std::vector<training_sample>::iterator> samples) {
    for (std::vector<training_sample>::iterator it = samples.begin(); it != samples.end(); it++) {
        getRelativeShape((*it));
    }
    });
    parallel_for_(Range(0,samples.size()),getDiffShape(&samples));
    return true;
}

void FacemarkKazemiImpl :: writeSplit( FileStorage& fs, splitr split)
{
    fs<<"split"<<"{";
    int index_1 = split.index1;
    int index_2 = split.index2;
    float thresh_ = split.thresh;
    fs<<"index1"<<index_1;
    fs<<"index2"<<index_2;
    fs<<"thresh"<<thresh_;
    fs<<"}";
}
void FacemarkKazemiImpl :: writeLeaf( FileStorage& fs,vector<Point2f> leaf){
    fs<<"leaf"<<"{";
    for(unsigned long i =0 ;i<leaf.size();i++){
        string s = string("X");
        stringstream ss;
        ss<<i;
        s = s+ss.str();
        fs<<s<<leaf[i].x;
        s= string("Y");
        s = s+ ss.str();
        fs<<s<<leaf[i].y;
    }
    fs<<"}";
}
void FacemarkKazemiImpl :: writeNode( FileStorage& fs, vector<tree_node> nodes,int i,unsigned long size){
    if(i>=(int)size)
        return ;
    string s = string("node");
    stringstream ss;
    ss<<i;
    s = s+ss.str();
    fs<<s<<"{";
    if(nodes[i].leaf.empty())
        writeSplit(fs,nodes[i].split);
    else
        writeLeaf(fs,nodes[i].leaf);
    fs<<"}";
    writeNode(fs,nodes,2*i+1,size);
    writeNode(fs,nodes,2*i+2,size);
}

void FacemarkKazemiImpl :: writeTree( FileStorage& fs, regtree tree,int tree_no )
{   string s = string("tree");
    stringstream ss;
    ss<<tree_no;
    s = s+ss.str();
    fs << s << "{";
    unsigned long root =0;
    writeNode(fs,tree.nodes,root,(unsigned long)tree.nodes.size());
    fs << "}";
}
void FacemarkKazemiImpl :: writeCascade( FileStorage& fs,vector<regtree> forest,int cascade_depth_){
    string s = string("cascade");
    stringstream ss;
    ss<<cascade_depth_;
    s = s+ss.str();
    fs<<s<<"{";
    for(unsigned long i=0;i<forest.size();i++){
        writeTree(fs,forest[i],i);
    }
    fs<<"}";
}
void writeLeaf(ofstream& os, const vector<Point2f> &leaf)
{
    unsigned long size = leaf.size();
    os.write((char*)&size, sizeof(size));
    os.write((char*)&leaf[0], leaf.size() * sizeof(Point2f));
}


void writeSplit(ofstream& os, splitr split)
{
    os.write((char*)&split, sizeof(split));
}




void writeTree(ofstream &f,regtree tree)
{
    string s("num_nodes");
    size_t len = s.size();
    f.write((char*)&len, sizeof(size_t));
    f.write(s.c_str(), len);
    unsigned long num_nodes = tree.nodes.size();
    f.write((char*)&num_nodes,sizeof(num_nodes));
    for(unsigned long i=0;i<tree.nodes.size();i++){
        if(tree.nodes[i].leaf.empty()){
            string s("split");
            size_t len = s.size();
            f.write((char*)&len, sizeof(size_t));
            f.write(s.c_str(), len);
            writeSplit(f,tree.nodes[i].split);
        }
        else{
            string s("leaf");
            size_t len = s.size();
            f.write((char*)&len, sizeof(size_t));
            f.write(s.c_str(), len);
            writeLeaf(f,tree.nodes[i].leaf);
        }
    }
}
bool save(vector< vector<regtree> > forest){
    ofstream f("model1.dat",ios::binary);
    if(!f.is_open()){
        cout<<"file not written.aborting.."<<endl;
        return false;
    }
    string s("cascade_depth");
    size_t len = s.size();
    f.write((char*)&len, sizeof(size_t));
    f.write(s.c_str(), len);
    unsigned long cascade_size = forest.size();
    f.write((char*)&cascade_size,sizeof(cascade_size));
    s = string("num_trees");
    len = s.size();
    f.write((char*)&len, sizeof(size_t));
    f.write(s.c_str(), len);
    unsigned long num_trees = forest[0].size();
    f.write((char*)&num_trees,sizeof(num_trees));
    for(unsigned long i=0;i<forest.size();i++){
        for(unsigned long j=0;j<forest[i].size();j++){
            writeTree(f,forest[i][j]);
        }
    }
    return true;
}


bool FacemarkKazemiImpl::train(vector<Mat>& images, vector< vector<Point2f> >& landmarks,string filename){
    if(images.size()!=landmarks.size()){
        // throw error if no data (or simply return -1?)
        String error_message = "The data is not loaded properly. Aborting training function....";
        CV_Error(Error::StsBadArg, error_message);
        return false;
    }
    if(!setTrainingParameters(filename)){
        String error_message = "Error while loading training parameters";
        CV_Error(Error::StsBadArg, error_message);
        return false;
    }
    FileStorage fs("Facemark_Kazemi.xml", FileStorage::WRITE);
    if (!fs.isOpened())
    {
        cerr << "Failed to open file to save model"<< endl;
        return false;
    }
    vector<training_sample> samples;
    vector< vector<Point2f> > pixel_coordinates;
    getTestCoordinates(pixel_coordinates,minmeanx,minmeany,maxmeanx,maxmeany);
    createTrainingSamples(samples,images,landmarks);
    images.clear();
    landmarks.clear();
    vector< vector<regtree> > forests;
    cout<<"Total Samples :"<<samples.size()<<endl;
    namedWindow("pappu");
    for(unsigned long k=0;k<samples.size();k++){
        Mat src = samples[k].image.clone();
        for(unsigned long i=0;i<samples[k].current_shape.size();i++){
            cv::circle(src,samples[k].current_shape[i],5,cv::Scalar(0,0,255),CV_FILLED);
        }
        imshow("pappu",src);
        waitKey(0);
    }
    for(unsigned long i=0;i<cascade_depth;i++){
        tbb::parallel_for(
        tbb::blocked_range<std::vector<training_sample>::iterator>(samples.begin(),samples.end()),
        [&] (tbb::blocked_range<std::vector<training_sample>::iterator> samples) {
        for (std::vector<training_sample>::iterator it = samples.begin(); it != samples.end(); it++) {
            getRelativePixels((*it),pixel_coordinates[i]);
            //getPixelIntensities((*it));
        }
        });
        for (std::vector<training_sample>::iterator it = samples.begin(); it != samples.end(); it++) {
            getPixelIntensities((*it));
        }
        cout<<"got pixel intensities"<<endl;
        cout<<"Training "<<i<<" regressor"<<endl;
        vector<regtree> forest = gradientBoosting(samples,pixel_coordinates[i]);
        writeCascade(fs,forest,i);
        forests.push_back(forest);
    }
    save(forests);
    /*for(unsigned long k=0;k<samples.size();k++){
        Mat src = samples[k].image.clone();
        for(unsigned long i=0;i<samples[k].current_shape.size();i++){
            cv::circle(src,samples[k].current_shape[i],5,cv::Scalar(0,0,255),CV_FILLED);
        }
        imshow("pappu",src);
        waitKey(0);
    }
    destroyWindow("pappu");*/
    fs.release();
    return true;
}
/*bool FacemarkKazemiImpl::getShape(vector<Rect> face, vector< vector<Point2f> >& shapes){
    if(image.empty()){
        String error_message = "No image found.Aborting..";
        CV_Error(Error::StsBadArg, error_message);
        return false;
    }
    vector< vector<regtree> > forests;
    vector< vector<Point2f> > pixel_coordinates;
    if(!load("model.bin",forests,pixel_coordinates)){
        cout<<"Training model not saved"<<endl;
        return false;
    }
    if(meanshape.empty()){
        cout<<"No meanshape found"<<endl;
        cout<<endl;
    }
    vector< vector<Point2f> > shapes;
    getMeanShapeRelative(face,shapes);
    tree_node curr_node;
    for(unsigned long k=0;k<faces.size();k++){
        for(unsigned long i=0;i<forests.size();i++){
            for(unsigned long j=0;j<forests[i].size();j++){
                regtree tree = forest[i][j];
                curr_node = tree.nodes[0];
                while(curr_node.leaf.size()==0){
                    if ((float)pixel_intensities[curr_node.split.index1] - (float)pixel_intensities[curr_node.split.index2] > curr_node.split.thresh)
                    {
                        curr_node=tree.nodes[left(curr_node)];
                    }
                    else
                        curr_node=tree.nodes[right(curr_node)];
                }
                for(unsigned long p=0;p<shapes[k].size();p++)
                    shapes[p][k]=shapes[p][k]-tree.node.leaf[k];
            }
        }
    }
    return true;
}*/
}//cv
}//face
