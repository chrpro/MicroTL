#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>


// Make empty matrix filled with zeros
// int rows: number of rows in matrix
// int cols: number of columns in matrix
// returns: matrix of specified size, filled with zeros
matrix make_matrix(int rows, int cols)
{
    matrix m;
    m.rows = rows;
    m.cols = cols;
    m.shallow = 0;
    m.data = calloc(m.rows*m.cols, sizeof(float));
    return m;
}

// Make a matrix with uniformly random elements
// int rows, cols: size of matrix
// float s: range of randomness, [-s, s]
// returns: matrix of rows x cols with elements in range [-s,s]
matrix random_matrix(int rows, int cols, float s)
{
    matrix m = make_matrix(rows, cols);
    int i, j;
    for(i = 0; i < rows; ++i){
        for(j = 0; j < cols; ++j){
            m.data[i*cols + j] = 2*s*(rand()%1000/1000.0) - s;     
        }
    } 
    return m;
}

// Free memory associated with matrix
// matrix m: matrix to be freed
void free_matrix(matrix m)
{
    if (!m.shallow && m.data){
            free(m.data);
        }
}

// Copy a matrix
// matrix m: matrix to be copied
// returns: matrix that is a deep copy of m
matrix copy_matrix(matrix m)
{
    matrix c = make_matrix(m.rows, m.cols);
    for (int i = 0; i < m.rows * m.cols; ++i) {
        c.data[i] = m.data[i];
    }
    // memcpy(c.data, m.data, m.rows*m.cols *sizeof(*m.data));
    // for (int row = 0; row < m.rows; row++) {
    //     for (int col = 0; col < m.cols; col++) {
    //         int index = row * m.cols + col;
    //         c.data[index] = m.data[index];
    //     }
    // }

    return c;
}

// Transpose a matrix
// matrix m: matrix to be transposed
// returns: matrix, result of transposition
matrix transpose_matrix(matrix m)
{
    matrix t = make_matrix(m.cols, m.rows);
    for (int row = 0; row < m.rows; row++) {
        for (int col = 0; col < m.cols; col++) {
            int index = row * m.cols + col;
            int transposed = col * m.rows + row;
            t.data[transposed] = m.data[index];
        }
    }

    return t;
}

// Perform y = ax + y
// float a: scalar for matrix x
// matrix x: left operand to the scaled addition
// matrix y: unscaled right operand, also stores result
void axpy_matrix(float a, matrix x, matrix y)
{
    assert(x.cols == y.cols);
    assert(x.rows == y.rows);
    // TODO: 1.3 - Perform the weighted sum, store result back in y
    for (int row = 0; row < x.rows; row++) {
        for (int col = 0; col < x.cols; col++) {
            int index = row * x.cols + col;
            y.data[index] += x.data[index] * a;
        }
    }
}

// Perform matrix multiplication a*b, return result
// matrix a,b: operands
// returns: new matrix that is the result
matrix matmul(matrix a, matrix b)
{
    matrix c = make_matrix(a.rows, b.cols);
    // TODO: 1.4 - Implement matrix multiplication. Make sure it's fast!
    // printf("a.cols:%d\n",a.cols);
    // printf("b.rows:%d\n",b.rows);
    assert(a.cols == b.rows);

    
    // for (int row = 0; row < c.rows; row++) {
    //     for (int internal = 0; internal < a.cols; internal++) {
    //         for (int col = 0; col < c.cols; col++) {
    //             int index = row * c.cols + col;
    //             int aIndex = row * a.cols + internal; // a[row][internal]
    //             int bIndex = internal * b.cols + col; // b[internal][col]
    //             c.data[index] += (a.data[aIndex] * b.data[bIndex]);
    //         }
    //     }
    // }

    float* a_data = a.data;
    float* b_data = b.data;
    float* c_data = c.data;
    int i, j, k;
    float t00, t01, t10, t11;

    for (i = 0; i < a.rows - 1; i += 2) {
        for (j = 0; j < b.cols - 1; j += 2) {
            t00 = 0;
            t01 = 0;
            t10 = 0;
            t11 = 0;
            
            for (k = 0; k < a.cols; ++k) {
                t00 += a_data[i*a.cols + k] * b_data[k*b.cols + j];
                t01 += a_data[i*a.cols + k] * b_data[k*b.cols + j + 1];
                t10 += a_data[(i+1)*a.cols + k] * b_data[k*b.cols + j];
                t11 += a_data[(i+1)*a.cols + k] * b_data[k*b.cols + j + 1];
            }

            c_data[i*c.cols + j] = t00;
            c_data[i*c.cols + j + 1] = t01;
            c_data[(i+1)*c.cols + j] = t10;
            c_data[(i+1)*c.cols + j + 1] = t11;
        }

        if (j == b.cols - 1) {
            t00 = 0;
            t10 = 0;
            
            for (k = 0; k < a.cols; ++k) {
                t00 += a_data[i*a.cols + k] * b_data[k*b.cols + j];
                t10 += a_data[(i+1)*a.cols + k] * b_data[k*b.cols + j];
            }

            c_data[i*c.cols + j] = t00;
            c_data[(i+1)*c.cols + j] = t10;
        }
    }

    if (i == a.rows - 1) {
        for (j = 0; j < b.cols; ++j) {
            t00 = 0;
            
            for (k = 0; k < a.cols; ++k) {
                t00 += a_data[i*a.cols + k] * b_data[k*b.cols + j];
            }

            c_data[i*c.cols + j] = t00;
        }
    }


    return c;
}

// In-place, element-wise scaling of matrix
// float s: scaling factor
// matrix m: matrix to be scaled
void scal_matrix(float s, matrix m)
{
    int i, j;
    for(i = 0; i < m.rows; ++i){
        for(j =0 ; j < m.cols; ++j){
            m.data[i*m.cols + j] *= s;
        }
    }
}


// Used for matrix inversion
matrix augment_matrix(matrix m)
{
    int i,j;
    matrix c = make_matrix(m.rows, m.cols*2);
    for(i = 0; i < m.rows; ++i){
        for(j = 0; j < m.cols; ++j){
            c.data[i*c.cols + j] = m.data[i*m.cols + j];
        }
    }
    for(j = 0; j < m.rows; ++j){
        c.data[j*c.cols + j+m.cols] = 1;
    }
    return c;
}

// Invert matrix m
matrix matrix_invert(matrix m)
{
    int i, j, k;
    //print_matrix(m);
    matrix none = {0};
    if(m.rows != m.cols){
        fprintf(stderr, "Matrix not square\n");
        return none;
    }
    matrix c = augment_matrix(m);
    
    //cast to int16_t to fit mcu memory space
    float **cdata = calloc((uint16_t)c.rows, sizeof(float *));
    for(i = 0; i < c.rows; ++i){
        cdata[i] = c.data + i*c.cols;
    }


    for(k = 0; k < c.rows; ++k){
        float p = 0.;
        int index = -1;
        for(i = k; i < c.rows; ++i){
            float val = fabs(cdata[i][k]);
            if(val > p){
                p = val;
                index = i;
            }
        }
        if(index == -1){
            fprintf(stderr, "Can't do it, sorry!\n");
            free_matrix(c);
            return none;
        }

        float *swap = cdata[index];
        cdata[index] = cdata[k];
        cdata[k] = swap;

        float val = cdata[k][k];
        cdata[k][k] = 1;
        for(j = k+1; j < c.cols; ++j){
            cdata[k][j] /= val;
        }
        for(i = k+1; i < c.rows; ++i){
            float s = -cdata[i][k];
            cdata[i][k] = 0;
            for(j = k+1; j < c.cols; ++j){
                cdata[i][j] +=  s*cdata[k][j];
            }
        }
    }
    for(k = c.rows-1; k > 0; --k){
        for(i = 0; i < k; ++i){
            float s = -cdata[i][k];
            cdata[i][k] = 0;
            for(j = k+1; j < c.cols; ++j){
                cdata[i][j] += s*cdata[k][j];
            }
        }
    }
    //print_matrix(c);
    matrix inv = make_matrix(m.rows, m.cols);
    for(i = 0; i < m.rows; ++i){
        for(j = 0; j < m.cols; ++j){
            inv.data[i*m.cols + j] = cdata[i][j+m.cols];
        }
    }
    free_matrix(c);
    free(cdata);
    //print_matrix(inv);
    return inv;
}

matrix solve_system(matrix M, matrix b)
{
    matrix none = {0};
    matrix Mt = transpose_matrix(M);
    matrix MtM = matmul(Mt, M);
    matrix MtMinv = matrix_invert(MtM);
    if(!MtMinv.data) return none;
    matrix Mdag = matmul(MtMinv, Mt);
    matrix a = matmul(Mdag, b);
    free_matrix(Mt); free_matrix(MtM); free_matrix(MtMinv); free_matrix(Mdag);
    return a;
}

void test_matrix()
{
    int i;
    for(i = 0; i < 100; ++i){
        int s = rand()%4 + 3;
        matrix m = random_matrix(s, s, 10);
        matrix inv = matrix_invert(m);
        if(inv.data){
            matrix res = matmul(m, inv);
            print_matrix(res);
        }
    }
}


// Print a matrix
void print_matrix(matrix m)
{
    int i, j;
    printf(" __");
    for(j = 0; j < 16*m.cols-1; ++j) printf(" ");
    printf("__ \n");

    printf("|  ");
    for(j = 0; j < 16*m.cols-1; ++j) printf(" ");
    printf("  |\n");

    for(i = 0; i < m.rows; ++i){
        printf("|  ");
        for(j = 0; j < m.cols; ++j){
            printf("%3.8f ", m.data[i*m.cols + j]);
        }
        printf(" |\n");
    }
    printf("|__");
    for(j = 0; j < 16*m.cols-1; ++j) printf(" ");
    printf("__|\n");
}