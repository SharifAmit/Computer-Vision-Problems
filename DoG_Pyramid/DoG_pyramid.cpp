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

    for (int i=0 ; i<height ; i++) {
        for (int j=0 ; j<width ; j++) {
            filter[i][j] /= sum;
        }
    }
    return filter;
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

Array oneDconv(Array padded1d,Array gk){
    int inp_len = padded1d.size()-gk.size()+1;
    Array oneDconvOutput(inp_len);
    int k;
    double element_sum;
    for (int i=0; i<inp_len; i++)
	{
		k = i;
		element_sum = 0.0;
		for (int j=0; j<gk.size(); j++)
		{
            element_sum = element_sum + (padded1d[k]*gk[j]);
			k = k+1;
			oneDconvOutput[i] = element_sum;
		}
	}
    return oneDconvOutput;
}
Matrix twoDsepconv(Matrix padded2d,Array gk){
    int inp_row = padded2d.size()-gk.size()+1;
    int inp_col = padded2d[0].size()-gk.size()+1;
    Matrix twoDconvOutput(inp_row, Array(inp_col));
    Matrix twoDintermediate(padded2d.size(), Array(inp_col));
    Matrix twoDintermediatepadded(padded2d.size(),Array(padded2d[0].size()));
    for (int i=0; i<padded2d.size(); i++){
            Array padded1d(padded2d.size());
            for (int x=0; x<padded2d.size(); x++)
            {
                padded1d[x] = padded2d[i][x];
            }
            Array output = oneDconv( padded1d, gk);
            for (int y=0; y<inp_row; y++)
            {
                twoDintermediate[i][y] = output[y];
            }
    }
    for (int i=0; i<inp_col; i++){
            Array padded1d(padded2d[0].size());
            for (int x=0; x<padded2d[0].size(); x++)
            {
                padded1d[x] = twoDintermediate[x][i];
            }
            Array output = oneDconv( padded1d, gk);
            for (int y=0; y<inp_col; y++)
            {
                twoDconvOutput[y][i] = output[y];
            }
    }

    return twoDconvOutput;
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

Matrix downsampling2d(Matrix input,double factor){
    int out_height = input.size()/factor;
    int out_width = input[0].size()/factor;
    Matrix downsampled(out_height,Array(out_width));
    int k,m,max;
    
    for(int i=0; i <out_height; i++){
        k=i*factor;
        for(int j=0; j <out_width; j++){
            m=j*factor;
            max = -100;
            for(int x=0;x<factor;x++){
                for(int y=0;y<factor;y++){
                    if( max <input[k+x][m+y])
                        max =  input[k+x][m+y];
                }
            }
            downsampled[i][j]= max;
        }
    }

    return downsampled;
}

Matrix difference(Matrix prev, Matrix next){
    Matrix new_2d(prev.size(),Array(prev[0].size()));
    for(int i=0; i<prev.size();i++){
        for(int j=0; j<prev[0].size();j++){
            new_2d[i][j] = next[i][j] - prev[i][j];
        }
    }
    return new_2d;
}
Matrix replace(Matrix prev){
    Matrix new_2d(prev.size(),Array(prev[0].size()));
    for(int i=0; i<prev.size();i++){
        for(int j=0; j<prev[0].size();j++){
            new_2d[i][j] = prev[i][j];
        }
    }
    return new_2d;
}

int main()
{
    double sigma;
    int n;
    double s;
    cout << "insert sigma value" <<endl;
    cin >> sigma;
    cout << "insert number of levels(octaves)" <<endl;
    cin >> n;
    cout << "insert intermediate levels " <<endl;
    cin >> s;
    Mat img = imread("lenna.pgm", IMREAD_GRAYSCALE);
    Matrix img_mat = img2matrix(img);
    int filtersize = 3;
    Matrix padded2d = twoDpadding(img_mat,filtersize);
    int sigma_prev = sigma*(0.5);
    Matrix gk_2d_prev = Gaussianfilter2D(filtersize,filtersize,sigma_prev);
    Matrix twoDconvOutput_prev = twoDconv(padded2d,gk_2d_prev);

    Matrix normalized(img_mat.size(),Array(img_mat[0].size()));
    Matrix normalized_prev = twoDnormalize(twoDconvOutput_prev);
    Matrix padded2d_prev = twoDpadding(twoDconvOutput_prev,filtersize);

    Matrix downsampled(img_mat.size()/2,Array(img_mat[0].size()/2));
    double k = pow(2,1/s);
    for(int j=0; j<n;j++){
        if(j>0){
             downsampled = downsampling2d(normalized_prev,2);
             cout << downsampled.size() << " "<< downsampled[0].size() << endl;
        }
        int cnt = 0;
        for(int i=0; i<s;i++){
            if(i==0 and j>0){
                sigma = k*sigma;
                cout << sigma << endl;
                normalized_prev = replace(downsampled);
                padded2d_prev = twoDpadding(downsampled,filtersize);
                cout << padded2d_prev.size() << " "<< padded2d_prev[0].size() << endl;
            }
            else if(i>0){
                sigma = k*sigma;
                cout << sigma << endl;
                padded2d_prev = twoDpadding(normalized_prev,filtersize);
            }
            Matrix gk_2d = Gaussianfilter2D(filtersize,filtersize,sigma);
            Matrix twoDconvOutput = twoDconv(padded2d_prev,gk_2d);
            cout << twoDconvOutput.size() << " "<< twoDconvOutput[0].size() << endl;
            normalized = twoDnormalize(twoDconvOutput);

            Matrix diff =  difference(normalized_prev,normalized);
            normalized_prev = replace(normalized);
            Matrix normalized_final = twoDnormalize(diff);
            Mat convimg = matrix2img(normalized_final);
            String img_name = "image " + string(" level ")+to_string(j)+ string(" diff level ")+to_string(cnt);
            imshow(img_name,convimg);
            String im_save_name = img_name + string(".png");
            imwrite(im_save_name,convimg);
            waitKey(0);
            cnt += 1;
        }
    }
    
    //String filename_img = "lenna_" + string("gk_") + to_string(filtersize)+string(".png");
    //cout << filename_img << endl;
    //imwrite(filename_img,convimg);
}
