#include <iostream>
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
    filter= { { -1, -2, -1 }, {0, 0, 0},{ 1, 2, 1 } };
    return filter;
}
Matrix Gaussianfilter2D(int height, int width, double sigma)
{
    Matrix filter(height, Array(width));
    double sum=0.0;
    int start  = -1*(height/2);
    int end = (height/2);
    double b;
    for (int i = start; i <= end; i++) {
        for (int j = start; j <= end; j++) {
            b = (i * i + j * j);
            filter[i + end][j + end] = (exp(-(b/ (2*sigma*sigma)))) / (2*M_PI*sigma*sigma);
            sum += filter[i + end][j + end];
        }
    }
    /*
    for (int i=0 ; i<height ; i++) {
        for (int j=0 ; j<width ; j++) {
            filter[i][j] /= sum;
        }
    }
    */
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

Matrix DerivativeOfGaussianX_2D(int height, int width, double sigma)
{
    Matrix filter(height, Array(width));
    double sum=0.0;
    int start  = -1*(height/2);
    int end = (height/2);
    double b;
    for (int i = start; i <= end; i++) {
        for (int j = start; j <= end; j++) {
            b = (i * i + j * j);
            filter[i + end][j + end] = ((-1*i)*exp(-(b/ (2*sigma*sigma)))) / (2*M_PI*sigma*sigma*sigma*sigma);
            //cout << filter[i + end][j + end] << "\t";
            sum += filter[i + end][j + end];
        }
        //cout << endl;
    }
    /*
    for (int i=0 ; i<height ; i++) {
        for (int j=0 ; j<width ; j++) {

            filter[i][j] /= sum;
            cout << filter[i][j] << "\t";
        }
        cout << endl;
    }
    */
    return filter;
}

Matrix DerivativeOfGaussianY_2D(int height, int width, double sigma)
{
    Matrix filter(height, Array(width));
    double sum=0.0;
    int start  = -1*(height/2);
    int end = (height/2);
    double b=0.0;
    for (int i = start; i <= end; i++) {
        for (int j = start; j <= end; j++) {
            b = (i * i + j * j);
            filter[i + end][j + end] = ((-1*j)*exp(-(b/ (2*sigma*sigma)))) / (2*M_PI*sigma*sigma*sigma*sigma);
            //cout << filter[i + end][j + end] << "\t";
            sum += filter[i + end][j + end];
        }
        //cout << endl;
    }
    /*
    for (int i=0 ; i<height ; i++) {
        for (int j=0 ; j<width ; j++) {
            filter[i][j] /= sum;
            cout << filter[i][j] << "\t";
        }
        cout << endl;
    }
    */
    return filter;
}

Matrix element_multiply(Matrix x,Matrix y){
    Matrix output(x.size(),Array(x[0].size()));
    for(int i=0;i<x.size();i++){
        for(int j=0;j<x[0].size();j++){
            output[i][j] = x[i][j]*y[i][j];
        }
    }
    return output;
}
Matrix mat_multiply(Matrix x, Matrix y){
    Matrix output(x.size(),Array(y[0].size()));

    for(int i = 0; i < output.size(); ++i){
        for(int j = 0; j < output[0].size(); ++j){
            for(int k = 0; k < x[0].size(); ++k){
                output[i][j] += x[i][k] * y[k][j];
            }
        }
    }
    return output;
}

Matrix element_sum(Matrix x,Matrix y){
    Matrix output(x.size(),Array(x[0].size()));
    for(int i=0;i<x.size();i++){
        for(int j=0;j<x[0].size();j++){
            output[i][j] = x[i][j]+y[i][j];
        }
    }
    return output;
}
Matrix alpha_multiply(Matrix x, double alpha){
    Matrix output(x.size(),Array(x[0].size()));
    for(int i=0;i<x.size();i++){
        for(int j=0;j<x[0].size();j++){
            output[i][j] = x[i][j]*alpha;
        }
    }
    return output;
}

Matrix element_substract(Matrix x,Matrix y){
    Matrix output(x.size(),Array(x[0].size()));
    for(int i=0;i<x.size();i++){
        for(int j=0;j<x[0].size();j++){
            output[i][j] = x[i][j]-y[i][j];
        }
    }
    return output;
}
double find_max(Matrix x){
    double max_val = 0.0;
    for(int i=0;i<x.size();i++){
        for(int j=0;j<x[0].size();j++){
            if (max_val < x[i][j]){
             max_val = x[i][j];
            }
        }
    }
    return max_val;
}
Matrix thresholded_Rw(Matrix x,double threshold){
    Matrix output(x.size(),Array(x[0].size()));
    for(int i=0;i<x.size();i++){
        for(int j=0;j<x[0].size();j++){
            if (threshold < x[i][j]){
             output[i][j] = x[i][j];
             center_x.push_back(i);
             center_y.push_back(j);
            }
        }
    }
    return output;

}

void non_max_suppress(Matrix x,int filter,double threshold){
    for(int i=0;i<x.size()-filter+1;i++){
        for(int j=0;j<x[0].size()-filter+1;j++){
            double max = 0.0;
            int dummy_x = 0;
            int dummy_y = 0;
            for(int a=i;a<i+filter;a++){
                for(int b=j;b<j+filter;b++){
                    if(max<x[a][b]){
                        max = x[a][b];
                        dummy_x=a;
                        dummy_y=b;
                    }
                }
            }
            for(int a=i;a<i+filter;a++){
                for(int b=j;b<j+filter;b++){
                    if(max!=x[a][b] && max>threshold){
                        x[a][b]=0;
                    }
                }
            }
            bool flag= true;
            for(int i=0;i<center_x.size();i++){
                if(center_x[i]==dummy_x && center_y[i]==dummy_y ){
                    flag=false;
                    break;
                }
            }
            if (flag==true){
                center_x.push_back(dummy_x);
                center_y.push_back(dummy_y);
            }
        }
    }
}

int main()
{
    vector<String> filenames = {"Test1","Test2","Test3"};
    for(int i=0; i<filenames.size(); i++){
        String file = "Harris/"+filenames[i]+ ".pgm";
        Mat img = imread(file, IMREAD_UNCHANGED);
        Matrix img_mat = img2matrix(img);
        double sigma_I = 1.5;
        double gamma = 0.7;
        double alpha = 0.06;
        double sigma_D = sigma_I*gamma;
        int filtersize = 3;

        Matrix padded2d = twoDpadding(img_mat,filtersize);


        Matrix Diff_G_X = DerivativeOfGaussianX_2D(filtersize,filtersize,sigma_D);
        Matrix I_x = twoDconv(padded2d,Diff_G_X);
        //Matrix normalized_I_x = twoDnormalize(I_x);
        //Mat convimg_I_x = matrix2img(normalized_I_x);
        //imshow("I_x",convimg_I_x);
        //waitKey(0);

        Matrix Diff_G_Y = DerivativeOfGaussianY_2D(filtersize,filtersize,sigma_D);
        Matrix I_y = twoDconv(padded2d,Diff_G_Y);
        //Matrix normalized_I_y = twoDnormalize(I_y);
        //Mat convimg_I_y = matrix2img(normalized_I_y);
        //imshow("I_y",convimg_I_y);
        //waitKey(0);


        Matrix I_x2 = element_multiply(I_x,I_x);
        //Matrix normalized_I_x2 = twoDnormalize(I_x2);
        //Mat convimg_I_x2 = matrix2img(normalized_I_x2);
        //imshow("I_x2",convimg_I_x2);
        //waitKey(0);


        Matrix I_y2 = element_multiply(I_y,I_y);
        //Matrix normalized_I_y2 = twoDnormalize(I_y2);
        //Mat convimg_I_y2 = matrix2img(normalized_I_y2);
        //imshow("I_y2",convimg_I_y2);
        //waitKey(0);

        Matrix I_xy = element_multiply(I_x,I_y);
        //Matrix normalized_I_xy = twoDnormalize(I_xy);
        //Mat convimg_I_xy = matrix2img(normalized_I_xy);
        //imshow("I_xy",convimg_I_xy);
        //waitKey(0);


        Matrix Gauss2d = Gaussianfilter2D(filtersize,filtersize,sigma_I);

        Matrix padded2d_I_x2 = twoDpadding(I_x2,filtersize);
        Matrix G_I_x2 = twoDconv(padded2d_I_x2,Gauss2d);
        //Matrix normalized_G_I_x2 = twoDnormalize(G_I_x2);
        //Mat convimg_G_I_x2 = matrix2img(normalized_G_I_x2);
        //imshow("G_I_x2",convimg_G_I_x2);
        //waitKey(0);


        Matrix padded2d_I_y2 = twoDpadding(I_y2,filtersize);
        Matrix G_I_y2 = twoDconv(padded2d_I_y2,Gauss2d);
        //Matrix normalized_G_I_y2 = twoDnormalize(G_I_y2);
        //Mat convimg_G_I_y2 = matrix2img(normalized_G_I_y2);
        //imshow("G_I_y2",convimg_G_I_y2);
        //waitKey(0);

        Matrix padded2d_I_xy = twoDpadding(I_xy,filtersize);
        Matrix G_I_xy = twoDconv(padded2d_I_xy,Gauss2d);
        //Matrix normalized_G_I_xy = twoDnormalize(G_I_xy);
        //Mat convimg_G_I_xy = matrix2img(normalized_G_I_xy);
        //imshow("G_I_xy",convimg_G_I_xy);
        //waitKey(0);

        Matrix cross_1 = element_multiply(G_I_x2,G_I_y2);
        //Matrix normalized_cross_1 = twoDnormalize(cross_1);
        //Mat convimg_cross_1 = matrix2img(normalized_cross_1);
        //imshow("cross_1",convimg_cross_1);
        //waitKey(0);

        Matrix cross_2 = element_multiply(G_I_xy,G_I_xy);
        //Matrix normalized_cross_2 = twoDnormalize(cross_2);
        //Mat convimg_cross_2 = matrix2img(normalized_cross_2);
        //imshow("cross_2",convimg_cross_2);
        //waitKey(0);

        Matrix det_M = element_substract(cross_1,cross_2);
        //Matrix normalized_det_M = twoDnormalize(det_M);
        //Mat convimg_det_M = matrix2img(normalized_det_M);
        //imshow("det_M",convimg_det_M);
        //waitKey(0);

        Matrix trace = element_sum(G_I_x2,G_I_y2);
        Matrix trace_2 = element_multiply(trace,trace);
        trace_2  = alpha_multiply(trace_2,alpha);
        //Matrix normalized_trace_2 = twoDnormalize(trace_2);
        //Mat convimg_trace_2= matrix2img(normalized_trace_2);
        //imshow("trace_2",convimg_trace_2);
        //waitKey(0);


        Matrix R_w = element_substract(det_M,trace_2);
        Matrix normalized_R_w = twoDnormalize(R_w);
        Mat convimg_R_w = matrix2img(normalized_R_w);
        //imshow("R_w",convimg_R_w);
        //waitKey(0);
        String r_w_file_name = filenames[i]+"_Corner_Response_"+".png";
        imwrite(r_w_file_name,convimg_R_w);


        double max_val = find_max(R_w);
        cout << "max value:" << max_val << endl;
        double threshold = 0.01;
        max_val = max_val*threshold;

        cout << "threshold 1%:" << max_val << endl;


        center_x.clear();
        center_y.clear();

        Matrix R_w_T = thresholded_Rw(R_w, max_val);
        //Matrix normalized_R_w_T = twoDnormalize(R_w_T);
        //Mat convimg_R_w_T = matrix2img(normalized_R_w_T);
        //imshow("R_w_T",convimg_R_w_T);
        //waitKey(0);

        cout << "no_x_t:" << center_x.size() << endl;
        cout << "no_y_t:" <<center_y.size() << endl;

        Mat img2;
        cvtColor(img,img,COLOR_GRAY2BGR);
        img.copyTo(img2);

        for(int i=0; i<center_x.size(); i++){
            circle(img, Point(center_y[i],center_x[i]),0.5, Scalar(0,0,255),-1, 8,0);
        }
        //imshow("corner_img",img);
        //waitKey(0);
        String corner_file = filenames[i]+"_corner_Threshold_"+".png";
        imwrite(corner_file,img);


        center_x.clear();
        center_y.clear();

        non_max_suppress(R_w_T, filtersize, max_val);

        cout << "no_x_nms:" << center_x.size() << endl;
        cout << "no_y_nms:" << center_y.size() << endl;
        for(int i=0; i<center_x.size(); i++){
            circle(img2, Point(center_y[i],center_x[i]),0.5, Scalar(0,0,255),-1, 8,0);
        }
        //imshow("corner_img2",img2);
        //waitKey(0);
        String corner_file_2 = filenames[i]+"_corner_Threshold_nms_"+".png";
        imwrite(corner_file_2,img2);
    }
    return 0;
}
