#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;


typedef vector<double> Array;
typedef vector<Array> Matrix;

extern "C" void solve_system(int, int, float**, float*, float*);
//Matrix normalized_mat()

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

Matrix inverse_mapping(Array affine_arr, Matrix img){

    // a11 a12 b1 a21 a22 b2
    double div = 1/((affine_arr[0]*affine_arr[4])-(affine_arr[1]*affine_arr[3]));
    //cout << "  " << endl;
    //cout << "div" << endl;
    //cout << "---" << endl;
    //cout << div << endl;
    Matrix x_prime(40, Array(48));
    Matrix affine_mat(3, Array(3));

    affine_mat[0][0] = affine_arr[4];
    affine_mat[0][1] = -1*(affine_arr[3]);
    affine_mat[0][2] = 0;
    affine_mat[1][0] = -1*(affine_arr[1]);
    affine_mat[1][1] = affine_arr[0];
    affine_mat[1][2] = 0;
    affine_mat[2][0] = (affine_arr[1]*affine_arr[5]) - (affine_arr[4]*affine_arr[2]);
    affine_mat[2][1] = (affine_arr[3]*affine_arr[2]) - (affine_arr[0]*affine_arr[5]);
    affine_mat[2][2] = (affine_arr[0]*affine_arr[4]) - (affine_arr[1]*affine_arr[3]);
    int x, y, one;
    int i = 0;
    int j = 0;
    for(int i=0; i<x_prime.size(); i++){
        for(int j=0; j<x_prime[0].size(); j++){
             x = div*((i*affine_mat[0][0]) + (j*affine_mat[1][0]) + (1*affine_mat[2][0]));
             y = div*((i*affine_mat[0][1]) + (j*affine_mat[1][1]) + (1*affine_mat[2][1]));
             if (x>91){
                x = 91;
             }
             else if (x<0){
                x=0;
             }
             if(y>111){
                y = 111;
             }
             else if (y<0){
                y=0;
             }
             x_prime[i][j] = img[x][y];
             //one = div*((i*affine_mat[0][2]) + (j*affine_mat[1][2]) + (1*affine_mat[2][2]));
             //cout << "x:" << " " << x << endl;
             //cout << "y:" << " " << y << endl;
             //cout << "one:" << " " << one << endl;
        }
    }
    return x_prime;

}

Array pointer_to_array(float *c,int l){
    Array arr(l);
    int k = 0;
    for(int i=1; i<l+1; i++){
         arr[k] = c[i];
         k++;
    }
    return arr;
}

void a_b(float **a, float *b,Matrix img ,int m,int n){
    int k = 92;
    int w = 112;
    //cout << "Value of a" << endl;
    //cout << "----------" << endl;
    for(int i=1; i<m+1; i++){
            for(int j=1; j<n+1; j++){
                

            }
    }
    k = 0;
    //cout << "          " << endl;
    //cout << "Value of b" << endl;
    //cout << "----------" << endl;
    for(int i=1; i<m+1; i++){
         b[i] = fixed_points[k];
         k++;
    //     cout << b[i]<< " ";
    }
    //cout << endl;


}



int main(int argc, char* argv[])
{
     int i;
     int m, n;
     float **a, *x, *b;


     /* Set m to the number of equations */
     /* Set n to the number of unknowns */

     m = 112*92;
     n = 4;

     a = new float* [m+1];
     for(i=0; i<m+1; i++)
       a[i] = new float [n+1];

     x = new float [n+1];

     b = new float [m+1];



     /* Fill matrix a and vector b with the desired values */
     /* The values should be placed at indices 1..m, 1..n */
    //for(int i=0; i<person_1.len)
    for(int k=1;k<3;k++){
        for(int i=1;i<11;i++){4
            String img_name = "S" + to_string(k) + string("/") + to_string(i)+string(".pgm");
            Mat img = imread(img_name, IMREAD_GRAYSCALE);
            Matrix img_mat = img2matrix(img);
            a_b(a,b,img_mat,m,n);
            solve_system(m,n,a,x,b);

            Array x_arr = pointer_to_array(x,n);
            /* ERROR CALCULATION NEEDED

            error_for_4_points = abs(fixed_points - x_arr)/fixed_points * 100

            */
            Array x_prime_pred(x_arr.size());
            for(int i=1;i<m+1;i++){
                for(int j=1;j<n+1;j++){
                    x_prime_pred[i-1] += a[i][j]*x_arr[j-1];
                }
            }

            double sum = 0;
            for(int i=1; i<n+1; i++){
                cout << "          " << endl;
                cout << "error value: " << i << endl;
                cout << "-----------" << endl;
                double diff =  abs(b[i]-x_prime_pred[i-1]);
                sum += diff;
                cout << diff << " ";
            }
            cout << endl;
            cout << "          " << endl;
            cout << "avg error value: " << endl;
            cout << "-----------" << endl;
            cout << sum/6 << endl;
            cout << " " << endl;
            String img_name = "S" + to_string(k) + string("/") + to_string(i)+string(".pgm");
            Mat img = imread(img_name, IMREAD_GRAYSCALE);
            Matrix img_mat = img2matrix(img);
            Matrix new_img = inverse_mapping(x_arr,img_mat);
            Mat converted_img = matrix2img(new_img);
            String filename_img = "S" + to_string(k) + string("_") + to_string(i) +string(".png");
            imwrite(filename_img,converted_img);
            cout << "Status: Done, " << filename_img << endl;
        }
    }
    //imshow("1",converted_img);
    //waitKey(0);
     /* The solution is now in vector x, at indices 1..n */


 }