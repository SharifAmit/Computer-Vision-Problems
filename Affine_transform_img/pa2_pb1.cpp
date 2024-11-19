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
             //cout << x <<" "<< y << endl;
             //cout << img[x][y] << endl;
             x_prime[i][j] = img[x][y];
             //cout << "hello" << endl;
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


void a_b(float **a, float *b,int person_1[8] ,int fixed_points[8],int m,int n){
    int k,w = 0;
    //cout << "Value of a" << endl;
    //cout << "----------" << endl;
    for(int i=1; i<m+1; i++){
            w=k;
            for(int j=1; j<n+1; j++){
                if(i%2!=0){
                    if(j<3){
                        a[i][j]= person_1[w];
                        w=k+1;
                        //cout<< a[i][j] << " ";
                    }
                    else if(j==3){
                        a[i][j]= 1;
                        //cout<< a[i][j] << " ";;
                    }
                    else if(j>3){
                        a[i][j]= 0;
                        //cout<< a[i][j] << " ";
                    }
                 }
                 else if(i%2==0){
                    if(j<4){
                        a[i][j]= 0;
                        //cout<< a[i][j] << " ";;
                    }
                    else if(j==4 || j==5){
                        a[i][j]= person_1[w];
                        w=k+1;
                        //cout<< a[i][j] << " ";;
                    }
                    else if(j==6){
                        a[i][j]= 1;
                        //cout<< a[i][j] << " ";;
                    }
                 }

                }
                //cout << endl;
            if(i%2==0){
                k=k+2;
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

     m = 8;
     n = 6;

     a = new float* [m+1];
     for(i=0; i<m+1; i++)
       a[i] = new float [n+1];

     x = new float [n+1];

     b = new float [m+1];


    int person_1[10][8] ={ //Left Eye, Right Eye, Nose tip and Mouth Center
            {25,51,    63,49,    46,68,    45,88},
            {31,44,    75,42,    65,64,    59,86},
            {24,47,    61,44,    45,73,    45,89},
            {15,44,    55,41,    28,62,    31,83},
            {35,44,    76,42,    69,62,    64,83},
            {11,47,    52,43,    25,66,    28,84},
            {24,45,    61,42,    43,58,    43,80},
            {28,45,    64,43,    49,65,    48,84},
            {30,36,    67,35,    53,46,    52,72},
            {34,51,    75,49,    63,75,    57,91}};

    int person_2[10][8] ={//Left Eye, Right Eye, Nose tip and Mouth Center
          {24,52,    59,53,    40,67,    39,89},
          {34,51,    68,52,    55,65,    52,89},
          {20,51,    55,53,    33,66,    33,87},
          {32,53,    65,52,    50,68,    49,89},
          {37,53,    73,52,    61,68,    58,89},
          {16,52,    50,53,    27,68,    28,88},
          {29,53,    61,53,    45,69,    44,89},
          {22,50,    57,52,    35,67,    36,89},
          {33,55,    67,56,    52,69,    49,91},
          {44,53,    77,52,    72,67,    66,88}};

    //int fixed_points[8]={10,8,   30,8,   20,26,   20,37};
    int fixed_points[8]={11,22,   27,21,   20,29,   19,37};

     /* Fill matrix a and vector b with the desired values */
     /* The values should be placed at indices 1..m, 1..n */


    double average_over_all_images = 0;
    for(int k=1;k<3;k++){
        for(int i=1;i<11;i++){
            if (k==1){
                a_b(a,b,person_1[i-1],fixed_points,m,n);
            }
            else if (k==2){
                a_b(a,b,person_2[i-1],fixed_points,m,n);
            }
            solve_system(m,n,a,x,b);
            Array x_arr = pointer_to_array(x,n);

            String img_name = "S" + to_string(k) + string("/") + to_string(i)+string(".pgm");
            Mat img = imread(img_name, IMREAD_GRAYSCALE);
            Matrix img_mat = img2matrix(img);
            Matrix new_img = inverse_mapping(x_arr,img_mat);

            Mat converted_img = matrix2img(new_img);
            String filename_img = "S" + to_string(k) + string("_") + to_string(i) +string(".png");
            imwrite(filename_img,converted_img);
            cout << "Status: Done, " << filename_img << endl;

            Array b_prime(m);
            for(int i=1;i<m+1;i++){
                for(int j=1;j<n+1;j++){
                     b_prime[i-1] += a[i][j]*x[j];
                }
            }
            double sum =0;

            for(int i=0; i<m; i++){
                    sum += abs(b[i+1]-b_prime[i]);
            }
            String img_name1 = "S" + to_string(k) + string("/") + to_string(i);
            cout << "Avg err " << img_name1 << " " << sum/8 << endl;
            average_over_all_images  += sum/8;



        }

    }
    cout << "Avg err all images: " << average_over_all_images/20 << endl;
//    The solution is now in vector x, at indices 1..n */


 }

