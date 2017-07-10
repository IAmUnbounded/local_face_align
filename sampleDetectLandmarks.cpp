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
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"

using namespace std;
using namespace cv;
using namespace cv::face;

int main(int argc,char** argv){
    //Give the path to the directory containing all the files containing data
   CommandLineParser parser(argc, argv,
        "{ help h usage ?    |      | give the following arguments in following format }"
        "{ model_filename f  |.     | (required) path to binary file storing the trained model which is to be loaded [example - /data/file.dat]}"
        "{ image i           |      | (required) path to image in which face landmarks have to be detected.[example - /data/image.jpg] }"
        "{ face_cascade c    |      | Path to the face cascade xml file which you want to use as a detector}"
    );
    clock_t begin = clock();
    // Read in the input arguments
    if (parser.has("help")){
        parser.printMessage();
        cerr << "TIP: Use absolute paths to avoid any problems with the software!" << endl;
        return 0;
    }
    string filename(parser.get<string>("model_filename"));
    string image(parser.get<string>("image"));
    Mat img = imread(image);
    if (filename.empty()||img.empty()){
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
    //pass the face cascade xml file which you want to pass as a detector
    CascadeClassifier face_cascade;
    face_cascade.load(cascade_name);
    Ptr<FacemarkKazemi> facemark= createFacemarkKazemi(face_cascade);
    cout<<filename<<endl;
    facemark->load(filename);
    cout<<"Loaded model"<<endl;
    //vector to store the faces detected in the image
    vector<Rect> faces;
    //Detect faces in the current image
    resize(img,img,Size(460,460));
    facemark->getFaces(img,faces);
    //vector to store the landmarks of all the faces in the image
    vector< vector<Point2f> > shapes;
    Mat src = img.clone();
    facemark->getMeanShapeRelative(faces,shapes);
    for(int i=0;i<faces.size();i++){
        for(unsigned long k=0;k<shapes[i].size();k++)
            cv::circle(src,shapes[i][k],5,cv::Scalar(0,0,255),CV_FILLED);
    }
    namedWindow("papputu");
    imshow("papputu",src);
    waitKey(0);
    if(facemark->getShape(img,faces,shapes))
    {
        for( size_t i = 0; i < faces.size(); i++ )
        {
            cv::rectangle(img,faces[i],Scalar( 255, 0, 0 ));
        }
        for(int i=0;i<faces.size();i++){
            for(unsigned long k=0;k<shapes[i].size();k++)
                cv::circle(img,shapes[i][k],5,cv::Scalar(0,0,255),CV_FILLED);
        }
        namedWindow("pappu");
        imshow("pappu",img);
        waitKey(0);
    }
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout<<elapsed_secs<<endl;
    return 0; 
}
