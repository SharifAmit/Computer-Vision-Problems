#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <vector>
#include <assert.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>



using namespace std;
using namespace cv;

typedef vector<double> Array;
typedef vector<Array> Matrix;

vector<int> center_x;
vector<int> center_y;
vector<int> circle_radius;

vector<int> edge_x;
vector<int> edge_y;


Matrix img2matrix(Mat img){
    Matrix img_mat(img.rows, Array(img.cols));
    for(int i=0; i<img.rows;i++){
        for(int j=0; j<img.cols;j++){
            img_mat[i][j] = img.at<uchar>(i,j);
        }
    }
    return img_mat;
}
Mat matrix2img(Matrix padded2d){
    Mat paddedimg(padded2d.size(), padded2d[0].size(), CV_8U);
    for(int i=0; i<padded2d.size();i++){
        for(int j=0; j<padded2d[0].size();j++){
            paddedimg.at<uchar>(i,j) = padded2d[i][j];
        }
    }
    return paddedimg;
}


Matrix thresholding(Matrix unthresholded,int threshold){
    Matrix thresholded(unthresholded.size(), Array(unthresholded[0].size()));
    for(int i=0; i<unthresholded.size();i++){
        for(int j=0; j<unthresholded[0].size();j++){
            if (unthresholded[i][j]>threshold){
                thresholded[i][j] = 255;
            }
            else {
                thresholded[i][j] = 0;
            }
        }
    }
    return thresholded;
}



Matrix overlap_circles(Matrix source, int radius){
    Matrix overlap_circle(source.size(), Array(source[0].size()));
    for(int i=0; i<source.size();i++){
        for(int j=0; j<source[0].size();j++){
            overlap_circle[i][j]=0;
        }
    }
    for(int i=0; i<source.size();i++){
        for(int j=0; j<source[0].size();j++)
            if(source[i][j]==255){
                for(int c=0; c<360; c++){
                    int a = i - radius*cos((c*M_PI)/180.0);
                    int b = j - radius*sin((c*M_PI)/180.0);
                    if (a<0 || a>=source.size()){
                        continue;
                    }
                    if (b<0 || b>=source[0].size()){
                        continue;
                    }
                    else{
                        overlap_circle[a][b] = overlap_circle[a][b]+1;
                        //error_calc[a][b][i][j] +=1;
                    }
                }
            }
    }
    return overlap_circle;
}



void max_id(Matrix oc,int radius){
    Array x_y(3);
    int temp =170;
    for(int i=0; i<oc.size();i++){
        for(int j=0; j<oc[0].size();j++){
            if(oc[i][j]>temp){

                    temp = oc[i][j];
                    x_y[0]=i;
                    x_y[1]=j;
                    x_y[2]=temp;

                    if (x_y[2]>170){
                        //cout << x_y[0] << " "<< x_y[1] << " " << x_y[2] << endl;
                        center_x.push_back(i);
                        center_y.push_back(j);
                        circle_radius.push_back(radius);
                    }

            }
        }
    }
}
void erase_cocentric_circle(){
    int k = center_x.size();
    for(int i=0; i<center_x.size(); i++){
        int checker_x=center_x[i];
        int checker_y=center_y[i];
        int checker_radius = circle_radius[i];
        //cout << "checker_x:" << checker_x << endl;
        //cout << "checker_y:" << checker_y << endl;
        for(int j=0; j<k; j++)
            if(i!=j){
                if((center_x[j]+14>=checker_x && center_y[j]+14>=checker_y) && (center_x[j]-14<=checker_x && center_y[j]-14<=checker_y)){
                    if ( circle_radius[j]<=checker_radius){
                        //cout << "popping:" <<center_x[j] << ","<< center_y[j]<< "," << circle_radius[j]<< endl;
                        center_x.erase (center_x.begin()+j);
                        center_y.erase (center_y.begin()+j);
                        circle_radius.erase (circle_radius.begin()+j);
                        k = k-1;
                        j=0;
                        i=0;
                        checker_x=center_x[i];
                        checker_y=center_y[i];
                        checker_radius = circle_radius[i];
                    }
                    else{
                        continue;
                    }
                    for(int i=0; i<center_x.size(); i++){
                        //cout << center_x[i] << ","<< center_y[i]<< "," << circle_radius[i] <<  "\t";
                    }
                    //cout << endl;
                }
            }
    }
}

void support_edges(Matrix source,int index){
    edge_x.clear();
    edge_y.clear();
    Matrix overlap_circle(source.size(), Array(source[0].size()));
    int k =0;
    for(int i=0; i<source.size();i++){
        for(int j=0; j<source[0].size();j++)
            if(source[i][j]==255){
                for(int c=0; c<360; c++){
                    int a = i - circle_radius[index]*cos((c*M_PI)/180.0);
                    int b = j - circle_radius[index]*sin((c*M_PI)/180.0);
                    if (a==center_x[index] && b==center_y[index]){
                        edge_x.push_back(i);
                        edge_y.push_back(j);
                        k +=1;
                    }
                }
            }
    }
    cout << "no of supports:" << k << endl;
}

float error_calc(int index){
    float d;
    for(int i=0; i<edge_x.size(); i++){
        float x_diff = edge_x[i]-center_x[index];
        float y_diff = edge_y[i]-center_y[index];
        d += abs(sqrt(pow(x_diff,2)+pow(y_diff,2))-circle_radius[index]);
    }
    float E = d/edge_x.size();
    return E;
}

int main()
{
    vector<String> file_names = {"c1","c2","c3","c4","c5","c6","overlap1","overlap2"};
    vector<Scalar> color_palette = {Scalar(0,0,255),Scalar(0,255,0),Scalar(255,0,0),Scalar(0,255,255),Scalar(255,0,255),Scalar(255,255,0)};
    for(int i=0; i<file_names.size();i++){

        String file = "Circles/"+ file_names[i] + ".pgm";
        Mat img = imread(file, IMREAD_UNCHANGED);
        //imshow("c1",img);
        //waitKey(1000);
        Mat canny_img;
        Canny( img, canny_img, 150, 150, 3);
        //imshow("canny",canny_img);
        //waitKey(0);
        Matrix img_mat = img2matrix(canny_img);

        center_x.clear();
        center_y.clear();
        circle_radius.clear();

        int radius_min = 30;
        int radius_max = 70;
        for(int radius=radius_min;radius <=radius_max;radius++){
            Matrix oc =overlap_circles(img_mat,radius);
            max_id(oc,radius);
        }

        erase_cocentric_circle();




        cvtColor(img,img,COLOR_GRAY2BGR);

        Mat mask;
        double alpha = 0.2;
        img.copyTo(mask);
        for(int i=0; i<center_x.size(); i++){
            cout << "x:" << center_x[i] << " y:" << center_y[i] << " r:" << circle_radius[i] << endl;
            circle(mask, Point(center_y[i],center_x[i]),circle_radius[i], color_palette[i],-1, 8,0);
            addWeighted(mask, alpha, img, 1 - alpha, 0, img);
        }
        String save_file = file_names[i]+".png";
        imwrite(save_file,img);
        //imshow("color_circle",img);
        //waitKey(0);

        for(int i=0; i<center_x.size(); i++){
            cout << "Circle " << to_string(i+1) << ":" << endl;
            cout << "-------------------" << endl;
            support_edges(img_mat,i);
            float E = error_calc(i);
            cout << "Error:" << E << endl;
            cout << "-------------------" << endl;
        }
    }
    return 0;
}
