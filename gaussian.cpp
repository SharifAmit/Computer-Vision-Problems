#include <iostream>
#include <vector>
#include <assert.h>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

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


Array Gaussianfilter1D(int length, double sigma)
{
    Array filter(length);
    double sum=0.0;
    int start = -1*(length/2);
    int end = (length/2);
    double b;
    for (int i = start; i <= end; i++) { 
        b = (i*i);
        filter[i + end] = (exp(-(b / (2*sigma*sigma)))) / (sqrt(2*M_PI)*sigma); 
        sum += filter[i + end]; 
    }

    for (int i=0 ; i<length ; i++) {
            filter[i] /= sum;
    }

    return filter;
}

Array onedInput(String File,int max_line)
{   
    Array oneDarray (max_line);
    ifstream file(File);
    if (file.is_open()) {
        string line;
        int i = 0 ;
        while (getline(file, line)) {
            // using printf() in all tests for consistency
            //printf("%s", line.c_str());
            oneDarray[i++] = stof(line);
            //cout << line << endl;
            
            //for 
        }
        //cout << cnt << endl;
        file.close();
    }
    return oneDarray;
}

void oneDoutput(String Filename,Array arr){
    ofstream file (Filename);
    if (file.is_open())
    {
        for(int i = 0; i < arr.size(); i ++){
            file << arr[i] << "\n" ;
        }
        file.close();
    }

}

Array oneDpadding(Array inputArray,int filtersize){
    int n_length = inputArray.size()+filtersize-1;
    Array oneDarray(n_length);
    for (int i=0; i <n_length; i++){
        oneDarray[i] = 0.0;
    }
    for (int i = 0; i < inputArray.size(); i++) {
        oneDarray[i+(filtersize/2)] = inputArray[i];
    }
    return oneDarray;
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
Array oneDconv(Array padded1d,Array gk){
    int inp_len = padded1d.size()-gk.size()+1;
    Array oneDconvOutput(inp_len);
    int k;
    double element_sum;
    for (int i=0; i<padded1d.size(); i++)
	{
		k = i;
		element_sum = 0.0;
		for (int j=0; j<gk.size(); j++)
		{
			if(k>=0 && k<padded1d.size())
				element_sum = element_sum + (padded1d[k]*gk[j]);
 
			k = k+1;
			oneDconvOutput[i] = element_sum;
		}
	}
    return oneDconvOutput;
}


Matrix twoDconv(Matrix padded2d,Matrix gk){
    int inp_height = padded2d.size()-gk.size()+1;
    int inp_width = padded2d[0].size()-gk[0].size()+1;
    Matrix twoDconvOutput(inp_height, Array(inp_width));
    cout <<twoDconvOutput.size() << " " << twoDconvOutput[0].size()<<endl;
    int k,m;
    double element_sum;
    int cnt = 0;

    for (int i=0; i<inp_height; i++)
	{   

        for (int j=0; j<inp_width; j++)
        {
            
            k = i;
            m = j;
            element_sum = 0.0;
            for (int x=0; x<gk.size();x++)
            {
                for (int y=0; y<gk[0].size();y++)
                {
                    if(k>=0 && k<padded2d.size() && m>=0 && m<padded2d[0].size()){
                        element_sum = element_sum + (padded2d[k][m]*gk[x][y]);
                    }
                    m = m +1;
                    
                }

            k = k+1;
            twoDconvOutput[i][j] = element_sum;
            //cout << i << " " << j << endl;
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
            cout << normalized[i][j] << endl;
        }
    }
    return normalized;
}

int main()
{   

    // Image read and padding
    Mat img = imread("lenna.pgm", IMREAD_GRAYSCALE);
    Matrix img_mat = img2matrix(img);
    int filtersize = 55;

    Matrix padded2d = twoDpadding(img_mat,filtersize);
    cout << padded2d.size() << " " << padded2d[0].size() << endl;

    Matrix gk_2d = Gaussianfilter2D(filtersize,filtersize,11);

    Matrix twoDconvOutput = twoDconv(padded2d,gk_2d);

    Matrix normalized = twoDnormalize(twoDconvOutput);
    Mat convimg = matrix2img(normalized);
    cout << convimg.rows << " " << convimg.cols << endl;

    imshow("Image", convimg);
    waitKey(0);


    // 1D signal read and conversion
    String Filename = "Rect_128.txt";
    int oneD_size = 128;
    Array oneD = onedInput(Filename,oneD_size);
    for (int j = 0; j < oneD_size; ++j){
        //cout<<oneD[j]<<endl;
    }


    // 1D gaussian filter
    
    int a [3]= {1,5,11};
    for(int k =0 ; k <3; ++k){
        int filtersize = 5*a[k];
        Array padded1d = oneDpadding(oneD,filtersize);
        cout << padded1d.size() <<endl;
        Array gk_1d = Gaussianfilter1D(filtersize,a[k]); 
        
        Array oneDconvOutput = oneDconv(padded1d,gk_1d);
        String filename = "rect_" + to_string(oneDconvOutput.size())+"_gk_"+to_string(filtersize)+".txt";
        cout << filename << endl;
        oneDoutput(filename,oneDconvOutput);
    }
    // 2D gaussian filter
    for(int k =0 ; k <3; ++k){
        int filtersize = 5*a[k];
        Matrix gk2d = Gaussianfilter2D(filtersize,filtersize,a[k]); 
        for(int i = 0; i < filtersize; ++i) 
        {
            for (int j = 0; j < filtersize; ++j){
                //cout<<gk2d[i][j]<<"\t";
            }
            //cout<<endl;
        }
        //cout << gk2d.size()<< endl;
    }
}