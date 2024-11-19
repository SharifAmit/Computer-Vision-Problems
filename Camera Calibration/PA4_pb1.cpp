#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <fstream>
#define 	CV_TERMCRIT_ITER   1
#define 	CV_TERMCRIT_EPS   2
#define 	CV_CALIB_CB_NORMALIZE_IMAGE   2
#define 	CV_CALIB_CB_ADAPTIVE_THRESH   1
#define 	CV_CALIB_CB_FAST_CHECK   8

using namespace cv;
using namespace std;


vector<Point2f> pixel_point(vector<Point2f> corner_pts,string filename){
    ifstream inFile;
    inFile.open(filename);
    float x,y;
    while (inFile >> x >> y)
    {
        corner_pts.push_back(Point2f(x,y));
    }
    inFile.close();
    return corner_pts;

}

vector<Point3f> world_points(vector<Point3f> world_points,string filename){
    ifstream inFile;
    inFile.open(filename);
    float x,y;
    while (inFile >> x >> y)
    {
        world_points.push_back(Point3f(x,y,0));
    }
    inFile.close();
    return world_points;

}

vector<Point2f> new_pixel_point(vector<Point3f> world_points,Mat cameraMatrix, Mat distCoeffs, Mat R, Mat T) {
    vector<Point2f> new_pixel_point;
    for(int i=0;i<96;i++){
        double x = (R.at<double>(0,0)*world_points[i].x) + (R.at<double>(0,1)*world_points[i].y)  + (R.at<double>(0,2)*world_points[i].z) + T.at<double>(0);
        double y = (R.at<double>(1,0)*world_points[i].x) + (R.at<double>(1,1)*world_points[i].y)  + (R.at<double>(1,2)*world_points[i].z) + T.at<double>(1);
        double z = (R.at<double>(2,0)*world_points[i].x) + (R.at<double>(2,1)*world_points[i].y)  + (R.at<double>(2,2)*world_points[i].z) + T.at<double>(2);
        double x_i = x/z;
        double y_i = y/z;

        double r_2 = pow (x_i, 2.0) + pow (y_i, 2.0);
        double r_4 = pow(r_2,2.0);
        double r_6 = pow(r_2,3.0);


        double k1, k2, p1, p2, k3, k4, k5, k6;
        k1 = distCoeffs.at<double>(0);
        k2 = distCoeffs.at<double>(1);
        p1 = distCoeffs.at<double>(2);
        p2 = distCoeffs.at<double>(3);
        k3 = distCoeffs.at<double>(4);
        k4 = distCoeffs.at<double>(5);
        k5 = distCoeffs.at<double>(6);
        k6 = distCoeffs.at<double>(7);


        double f_x, f_y, c_x, c_y;
        f_x =cameraMatrix.at<double>(0,0);
        f_y =cameraMatrix.at<double>(1,1);
        c_x =cameraMatrix.at<double>(0,2);
        c_y =cameraMatrix.at<double>(1,2);

        // RADIAL Distortions & Tangenial Distortion
        double rad_numer= 1 + (k1*r_2) + (k2*r_4) + (k3*r_6);
        double rad_denom = 1 + (k4*r_2) + (k5*r_4) + (k6*r_6);
        double rad = rad_numer/rad_denom;
        //cout << rad << endl;
        double x_ii = (x_i*rad) + (2*p1*x_i*y_i) + p2*(r_2 + (2*pow(x_i,2.0)));
        double y_ii = (y_i*rad) + p1*(r_2 + (2*pow(y_i,2.0))) + (2*p2*x_i*y_i);



        //camera focal length
        double u = (f_x*x_ii)+ c_x;
        double v = (f_y*y_ii)+ c_y;
        //cout << u << endl;
        //cout << v << endl;

        //new_pixel_point.push_back(Point2f(0,0));

        new_pixel_point.push_back(Point2f(u,v));
    }
    return new_pixel_point;
}
Mat convert_R_matrix(Mat R){
    double accum;
    for (int i = 0; i < 3; ++i) {
        accum += R.at<double>(i) * R.at<double>(i);
    }
    double theta = sqrt(accum);
    vector<vector<double> > idty( 3, vector<double>( 3, 0 ));
    for( int i = 0; i < 3; ++i )
        idty[i][i] = cos(theta);

    vector<double> R_vec;
    for(int i = 0; i < 3; ++i ){
        R_vec.push_back(R.at<double>(i));
    }
    for(int i = 0; i < 3; ++i ){
        R_vec[i]=R_vec[i]/theta;
    }
    vector<vector<double>> R_R;
    vector<double> dum_R;
    for(int i = 0; i < 3; ++i ){

        for(int j = 0; j < 3; ++j ){
            dum_R.push_back((1-cos(theta))*R_vec[i]*R_vec[j]);
        }
        R_R.push_back(dum_R);
        dum_R.clear();
    }
    vector<vector<double>> sinR {{0,-1*R_vec[2],R_vec[1]},{R_vec[2],0,-1*R_vec[0]},{-1*R_vec[1],R_vec[0],0}};
    for(int i = 0; i < 3; ++i ){
        for(int j = 0; j < 3; ++j ){
            sinR[i][j] = sinR[i][j]*sin(theta);

        }

    }
    Mat R_matrix(3,3,CV_64F);
    for(int i = 0; i < 3; ++i ){
        for(int j = 0; j < 3; ++j ){
            R_matrix.at<double>(i,j) = sinR[i][j] + R_R[i][j] + idty[i][j];
        }
    }
    return R_matrix;
}
int main()
{
    vector<vector<Point3f> > objpoints;
    vector<vector<Point2f> > imgpoints;

    vector<String> images;
    string path = "./Data/*.bmp";

    glob(path, images);

    Mat frame, gray;

    bool success;
    string output_file[15] = {"cap_out01.png","cap_out02.png","cap_out03.png","cap_out04.png","cap_out05.png","cap_out06.png","cap_out07.png","cap_out08.png","cap_out09.png",
                            "cap_out10.png","cap_out11.png","cap_out12.png","cap_out13.png","cap_out14.png","cap_out15.png"};
    string pixel_file[15] = { "Data/pixel01.txt", "Data/pixel02.txt", "Data/pixel03.txt", "Data/pixel04.txt", "Data/pixel05.txt", "Data/pixel06.txt", "Data/pixel07.txt", "Data/pixel08.txt",
                        "Data/pixel09.txt", "Data/pixel10.txt", "Data/pixel11.txt", "Data/pixel12.txt","Data/pixel13.txt", "Data/pixel14.txt", "Data/pixel15.txt"};
    string world_file[15] = { "Data/world01.txt", "Data/world02.txt", "Data/world03.txt", "Data/world04.txt", "Data/world05.txt", "Data/world06.txt", "Data/world07.txt", "Data/world08.txt",
                        "Data/world09.txt", "Data/world10.txt", "Data/world11.txt", "Data/world12.txt","Data/world13.txt", "Data/world14.txt", "Data/world15.txt"};
    for(int i{0}; i<15; i++)
    {
        vector<Point3f> obj_points;

        vector<Point2f> img_points;


        string filename = pixel_file[i];
        img_points = pixel_point(img_points,filename);

        string world_filename = world_file[i];
        obj_points = world_points(obj_points,world_filename);
        imgpoints.push_back(img_points);
        objpoints.push_back(obj_points);
    }

    ofstream myfile("output.txt");
    Mat cameraMatrix,distCoeffs,R,T;
    int flag = 16384;
    frame = imread(images[0]);
    cvtColor(frame,gray,COLOR_BGR2GRAY);
    double rms = calibrateCamera(objpoints, imgpoints, Size(gray.rows,gray.cols), cameraMatrix, distCoeffs, R, T,flag);
    myfile << "rms " << rms << endl;
    myfile << "cameraMatrix : " << cameraMatrix << endl;
    cout << "cameraMatrix : " << cameraMatrix << endl;
    myfile << "distCoeffs : " << distCoeffs << endl;
    cout << "distCoeffs : " << distCoeffs << endl;
    myfile << "Translation vector : " << T<< endl;
    cout << "Translation vector : " << T<< endl;
    myfile << "Rotation vector : " << R<< endl;


    double rms_avg = 0;
    for(int i=0; i<15; i++){

        Mat R_Mat = convert_R_matrix(R.row(i));
        myfile << "Rotation Matrix " << to_string(i+1) << ":"<< R_Mat<< endl;
        cout << "Rotation Matrix " << to_string(i+1) << ":"<< R_Mat<< endl;
        frame = imread(images[i]);
        cvtColor(frame,gray,COLOR_BGR2GRAY);
        vector<Point2f> new_img_points = new_pixel_point(objpoints[i],cameraMatrix, distCoeffs, R_Mat, T.row(i));
        for(int i=0; i<new_img_points.size(); i++){
            circle(frame, Point(new_img_points[i].x,new_img_points[i].y),2, Scalar(0,0,255),-1, 8,0);
        }
        double err = 0;
        for(int j=0; j<imgpoints[i].size(); j++){
                err += sqrt( pow(imgpoints[i][j].x-new_img_points[j].x ,2) + pow(imgpoints[i][j].y-new_img_points[j].y,2));
        }
        err = err/96.0;
        rms_avg += err;
        myfile << "err_" << to_string(i+1) << ": "<< err << endl;
        cout << "err_" << to_string(i+1) << ": "<< err << endl;
        imwrite(output_file[i],frame);
    }
    myfile << "err_avg: " << rms_avg/15 << endl;
    cout << "err_avg: " << rms_avg/15 << endl;
    return 0;

}
