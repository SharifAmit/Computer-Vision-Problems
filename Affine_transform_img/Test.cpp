#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <vector>
using namespace std;
extern "C" void solve_system(int, int, float**, float*, float*);

typedef vector<double> Array;
typedef vector<Array> Matrix;

Matrix init_zero_2d()

Array inverse_mapping(Array affine_arr){

    // a11 a12 b1 a21 a22 b2
    double div = 1/((affine_arr[0]*affine_arr[4])-(affine_arr[1]*affine_arr[3]));
    cout << "  " << endl;
    cout << "div" << end;
    cout << "---" << endl;
    cout << div << endl;
    Matrix x_prime(40, Array(48));
    Matrix affine_mat(3, Array(3));

    affine_mat[][]
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

void a_b(float **a, float *b,int person_1[8] ,int fixed_points[8],int m,int n){
    int k,w = 0;
    cout << "Value of a" << endl;
    cout << "----------" << endl;
    for(int i=1; i<m+1; i++){
            w=k;
            for(int j=1; j<n+1; j++){
                if(i%2!=0){
                    if(j<3){
                        a[i][j]= person_1[w];
                        w=k+1;
                        cout<< a[i][j] << " ";
                    }
                    else if(j==3){
                        a[i][j]= 1;
                        cout<< a[i][j] << " ";;
                    }
                    else if(j>3){
                        a[i][j]= 0;
                        cout<< a[i][j] << " ";
                    }
                 }
                 else if(i%2==0){
                    if(j<4){
                        a[i][j]= 0;
                        cout<< a[i][j] << " ";;
                    }
                    else if(j==4 || j==5){
                        a[i][j]= person_1[w];
                        w=k+1;
                        cout<< a[i][j] << " ";;
                    }
                    else if(j==6){
                        a[i][j]= 1;
                        cout<< a[i][j] << " ";;
                    }
                 }

                }
                cout << endl;
            if(i%2==0){
                k=k+2;
            }
    }
    k = 0;
    cout << "          " << endl;
    cout << "Value of b" << endl;
    cout << "----------" << endl;
    for(int i=1; i<m+1; i++){
         b[i] = fixed_points[k];
         k++;
         cout << b[i]<< " ";
    }
    cout << endl;


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

    int fixed_points[8]={10,8,   30,8,   20,26,   19,37};

     /* Fill matrix a and vector b with the desired values */
     /* The values should be placed at indices 1..m, 1..n */
    a_b(a,b,person_1[0],fixed_points,m,n);
    solve_system(m,n,a,x,b);

    Array x_arr = pointer_to_array(x,n);

    cout << "           " << endl;
    cout << "Value of X " << endl;
    cout << "-----------" << endl;
    for(int i=0; i<n; i++){
        cout << x_arr[i] << " ";
    }
    cout << endl;
     /* The solution is now in vector x, at indices 1..n */


 }

