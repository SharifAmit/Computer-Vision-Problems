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
/*
Matrix inverse_mapping(Array affine_arr, Matrix img){

    // a11 a12 b1 a21 a22 b2
    double div = 1/((affine_arr[0]*affine_arr[4])-(affine_arr[1]*affine_arr[3]));
    //cout << "  " << endl;
    //cout << "div" << endl;
    //cout << "---" << endl;
    //cout << div << endl;
    Matrix x_prime(48, Array(40));
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
             if (x>111){
                x = 111;
             }
             else if (x<0){
                x=0;
             }
             if(y>91){
                y = 91;
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
*/
Array pointer_to_array(float *c,int l){
    Array arr(l);
    for(int i=1; i<l+1; i++){
         arr[i-1] = c[i];
    }
    return arr;
}

Matrix pointer_to_matrix(float **c,int m,int n){
    Matrix arr_mat(m, Array(n));

    for(int i=1; i<m+1; i++){
        for(int j=1; i<n+1; i++){
         arr_mat[i-1][j-1] = c[i][j];
        }
    }
    return arr_mat;
}


void a_b(float **a, float *b,Matrix img,int m,int n){
    int k,l = 0;

    for(int i=1; i<m+1; i++){
        a[i][1] = k;
        a[i][2] = l;
        a[i][3] = k*l;
        a[i][4] = 1;
        l++;
        if(l==40){
            l=0;
            //cout << k << endl;
            k++;
        }
    }
    k = 0;
    l = 0 ;
    //cout << k << endl;
    for(int i=1; i<m+1; i++){
        b[i] = img[k][l];
        //cout << l << endl;
        l++;
        if(l==40){
            l=0;
            //cout << k << endl;
            k++;
        }
    }
    //cout << endl;


}

Matrix twoDnormalize(Matrix unnormalized){
    double max = -9999;
    double min = 99999;
    Matrix normalized (unnormalized.size(), Array(unnormalized[0].size()));
    for(int i=0;i<unnormalized.size();i++){
        for(int j=0; j<unnormalized[0].size();j++){
            if(unnormalized[i][j]>max){
                max = unnormalized[i][j];
            }
            else if(unnormalized[i][j]<=min){
                min = unnormalized[i][j];
            }
        }
    }

    for(int i=0;i<normalized.size();i++){
        for(int j=0; j<normalized[0].size();j++){
            normalized[i][j] = (int)((unnormalized[i][j] - min)*(255/(max-min)));
            //cout << normalized[i][j] << endl;
        }
    }
    return normalized;
}

int main(int argc, char* argv[])
{
     int i;
     int m, n;
     float **a, *x, *b;


     /* Set m to the number of equations */
     /* Set n to the number of unknowns */

     m = 1920;
     n = 4;

     a = new float* [m+1];
     for(i=0; i<m+1; i++)
       a[i] = new float [n+1];

     x = new float [n+1];

     b = new float [m+1];



     /* Fill matrix a and vector b with the desired values */
     /* The values should be placed at indices 1..m, 1..n */


    double average_over_all_images = 0;
    for(int k=1;k<3;k++){
        for(int i=1;i<11;i++){

            String img_name = "S" + to_string(k) + string("_") + to_string(i) +string(".png");
            Mat img = imread(img_name, IMREAD_GRAYSCALE);
            Matrix img_mat = img2matrix(img);

            a_b(a,b,img_mat,m,n);
            solve_system(m,n,a,x,b);
            //cout << x[1] << endl;
            //cout << x[2] << endl;
            //cout << x[3] << endl;
            //cout << x[4] << endl;
            Matrix mask(48, Array(40));

            Array x_prime_pred(1920);
            for(int i=1;i<m+1;i++){
                x_prime_pred[i-1] = a[i][1]*x[1] + a[i][2]*x[2] + a[i][3]*x[3]+ a[i][4]*x[4];
            }

            //Matrix a_mat = pointer_to_matrix(a,m,n);
            int w = 0;
            for(int i=0;i<48;i++){
                for(int j=0;j<40;j++){
                    mask[i][j] = x_prime_pred[w];
                    w++;
                }
            }
            Matrix normalized_mask =  twoDnormalize(mask);
            Mat converted_mask = matrix2img(normalized_mask);
            String filename_mask = "S" + to_string(k) + string("_") + to_string(i) + string("_mask")+string(".png");
            imwrite(filename_mask,converted_mask);
            Matrix out_img(48, Array(40));
            //Array b_arr = pointer_to_array(b,m);
            for(int i=0;i<48;i++){
                for(int j=0;j<40;j++){
                    out_img[i][j] = img_mat[i][j]-mask[i][j];
                }
            }

            //average_over_all_images += sum/8;

            Matrix normalized =  twoDnormalize(out_img);
            //Matrix new_img = inverse_mapping(x_arr,img_mat);
            Mat converted_img = matrix2img(normalized);
            String filename_img = "S" + to_string(k) + string("_") + to_string(i) + string("_light_enhanced")+string(".png");
            imwrite(filename_img,converted_img);
            cout << "Status: Done, " << filename_img << " " << filename_mask << endl;
        }
    }
    //imshow("1",converted_img);
    //waitKey(0);
     /* The solution is now in vector x, at indices 1..n */


 }

