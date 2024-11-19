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

Matrix SobelFilterX()
{
    Matrix filter(3, Array(3));
    filter= { { -1, 0, 1 }, {-2, 0, 2 },{ -1, 0, 1 } };
    /*
    for (int i = 0; i < filter_size; i++) {
        for (int j = 0; j < filter_size; j++) {
                filter[i][j]=
        }
    }
    */
    return filter;
}

Matrix SobelFilterY()
{
    Matrix filter(3, Array(3));
    filter= { { -1, -2, 1 }, {0, 0, 0},{ -1, 2, 1 } };
    return filter;
}

Matrix twoDconv(Matrix padded2d,Matrix gk){
    int inp_height = padded2d.size()-gk.size()+1;
    int inp_width = padded2d[0].size()-gk[0].size()+1;
    Matrix twoDconvOutput(inp_height, Array(inp_width));
    //cout <<twoDconvOutput.size() << " " << twoDconvOutput[0].size()<<endl;
    int k,m;
    double element_sum;
    int cnt = 0;

    for (int i=0; i<inp_height; i++)
	{

        for (int j=0; j<inp_width; j++)
        {

            k = i;
            element_sum = 0.0;
            for (int x=0; x<gk.size();x++)
            {

                m = j;
                for (int y=0; y<gk[0].size();y++)
                {
                    element_sum = element_sum + (padded2d[k][m]*gk[x][y]);
                    m = m +1;

                }

            k = k+1;
            twoDconvOutput[i][j] = element_sum;
            }

        }

	}
    return twoDconvOutput;
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

Matrix twoDpadding(Matrix inputArray, int filtersize){
    int n_height = inputArray.size()+filtersize-1;
    int n_width = inputArray[0].size()+filtersize-1;
    Matrix twoDarray(n_height, Array(n_width));
    for (int i=0; i <n_height; i++){
        for(int j=0; j<n_width; j++){
            twoDarray[i][j] = 0.0;
        }
    }
    for (int i = 0; i < inputArray.size(); i++) {
        for(int j=0; j<inputArray[0].size(); j++){
            twoDarray[i+(filtersize/2)][j+(filtersize/2)] = inputArray[i][j];
        }
    }
    return twoDarray;
}
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
Matrix sobelMagnitude(Matrix sobelx,Matrix sobely){
    Matrix magnitude_sobel(sobelx.size(),Array(sobelx[0].size()));
    for(int i=0; i<sobelx.size();i++){
        for(int j=0; j<sobelx[0].size();j++){
            magnitude_sobel[i][j] = sqrt((sobelx[i][j]*sobelx[i][j]) + (sobely[i][j]*sobely[i][j]));
        }
    }
    return magnitude_sobel;
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
int main()
{
    int threshold = 0;
    cout << "Enter Threshold value (0-255)" << endl;
    cin >> threshold ;
    Matrix sobelX = SobelFilterX();
    for(int i = 0; i < sobelX.size(); ++i)
    {
        for (int j = 0; j < sobelX[0].size(); ++j){
            cout<<sobelX[i][j]<<"\t";
        }
        cout<<endl;
    }
    cout << endl;
    Matrix sobelY = SobelFilterY();
    for(int i = 0; i < sobelY.size(); ++i)
    {
        for (int j = 0; j < sobelY[0].size(); ++j){
            cout<<sobelY[i][j]<<"\t";
        }
        cout<<endl;
    }
    int filtersize =3;
    Mat img = imread("lenna.pgm", IMREAD_GRAYSCALE);
    Matrix img_mat = img2matrix(img);

    Matrix padded2d = twoDpadding(img_mat,filtersize);
    Matrix twoDconvOutput_Mx = twoDconv(padded2d,sobelX);
    Matrix normalized = twoDnormalize(twoDconvOutput_Mx);
    Mat convimg = matrix2img(normalized);
    imshow("lenna_sobelX",convimg);
    waitKey(1000);
    String filename_img = "lenna_" + string("sobel_Mx_") +string(".png");
    imwrite(filename_img,convimg);

    Matrix twoDconvOutput_My = twoDconv(padded2d,sobelY);
    normalized = twoDnormalize(twoDconvOutput_My);
    convimg = matrix2img(normalized);
    imshow("lenna_sobelY",convimg);
    waitKey(1000);
    filename_img = "lenna_" + string("sobel_My_") +string(".png");
    imwrite(filename_img,convimg);

    Matrix twoDmagnitude = sobelMagnitude(twoDconvOutput_Mx,twoDconvOutput_My);
    normalized = twoDnormalize(twoDmagnitude);
    convimg = matrix2img(normalized);
    imshow("lenna_magntidue",convimg);
    waitKey(1000);
    filename_img = "lenna_" + string("sobel_Magnitude_unthresholded") +string(".png");
    imwrite(filename_img,convimg);


    Matrix thresholded = thresholding(normalized,threshold);
    convimg = matrix2img(thresholded);
    imshow("lenna_magntidue_thresholded",convimg);
    waitKey(1000);
    filename_img = "lenna_" + string("sobel_Magnitude_thresholded") +string(".png");
    imwrite(filename_img,convimg);

}
