//STD libs
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

//Training libs
#include "microTL.h"
#include "test.h"
#include "args.h"

// CMSIS Libs
#include "arm_math.h"
#include "arm_nnfunctions.h"

// Test images
#include "mnist-images.h"
// #include "image2.h"

// Neural Network
#include "parameters.h"
#include "weights.h"

// #include "network.h"

// image size  28 * 28 = 784
# define IMAGE_SIZE  784 
# define CLASSES 10

float cross_entropy_loss(matrix, layer);


uint32_t network(q7_t* input);

static q7_t conv1_out[CONV1_OUT_CH*CONV1_OUT_DIM*CONV1_OUT_DIM];
static q7_t pool1_out[CONV1_OUT_CH*POOL1_OUT_DIM*POOL1_OUT_DIM];
static q7_t conv2_out[CONV2_OUT_CH*CONV2_OUT_DIM*CONV2_OUT_DIM];
static q7_t pool2_out[CONV2_OUT_CH*POOL2_OUT_DIM*POOL2_OUT_DIM];
static q7_t interface_out[INTERFACE_OUT];
static q7_t linear_out[LINEAR_OUT];
static q7_t y_out[LINEAR_OUT];


// network weights
static q7_t conv1_w[CONV1_WT_SHAPE] = CONV1_WT;
static q7_t conv1_b[CONV1_BIAS_SHAPE] = CONV1_BIAS;
static q7_t conv2_w[CONV2_WT_SHAPE] =  CONV2_WT;
static q7_t conv2_b[CONV2_BIAS_SHAPE] = CONV2_BIAS;
static q7_t interface_w[INTERFACE_WT_SHAPE] = INTERFACE_WT;
static q7_t interface_b[INTERFACE_BIAS_SHAPE] = INTERFACE_BIAS;
static q7_t linear_w[LINEAR_WT_SHAPE] = LINEAR_WT;
static q7_t linear_b[LINEAR_BIAS_SHAPE] = LINEAR_BIAS;


static q15_t conv_buffer[MAX_CONV_BUFFER_SIZE];
static q15_t fc_buffer[MAX_FC_BUFFER];




int main(void)
{

    static const char chalmersBanner[] = {
		"============================================================\n"
		"Tranfers Learning for IoT                                   \n"
		"Copyright (C) Chalmers University. All rights reserved      \n"
		"============================================================\n"
	}; printf(chalmersBanner);
    uint32_t index = 0;

  


    // data train = load_image_classification_data("./mnist/mnist.train", "./mnist/mnist.labels");
    // data test  = load_image_classification_data("./mnist/mnist.test", "./mnist/mnist.labels");


    // for(int i = 0; i<4;i++){
    //     printf("%d\n",img_buffer1[i]);
    // }

    // pre_processing(&MNIST_IMG[0]);
    // run_nn(&MNIST_IMG[0]);
    // printf("max buffer");
    // for (int i = 0; i < 10816; i++)
    // {
    //     printf("%d",max_buffer2[i]);
    // }

    // uint8_t x = 0;
    // for (int i = 0; i < 10; i++)
    // {
       
    //     if (output_data[i] > output_data[x])
    //         x = i;
    // }

    // printf("output_pred: %d \n",x);



   int Pool2Out = CONV2_OUT_CH*POOL2_OUT_DIM*POOL2_OUT_DIM;


    net n = {0};
    n.n = 2;
    n.layers = calloc(2, sizeof(layer));
    n.layers[0] = make_connected_layer(Pool2Out, 32, RELU);
    n.layers[1] = make_connected_layer(32 , 10, SOFTMAX);


    // for (int i=0;i < 10 ;i ++){
    //     index = network(&MNIST_IMG[i]);
    //     printf("\n Real label: %d , predict: %u \n", MNIST_LABELS[i], index);
    // }


    int batch = 64;
    int iterations = 60;


    float rate = .0001;
    float momentum = .9;
    float decay = .0001;



    int iter;


    // for(iter = 0; iter < 10; ++iter){
    //     index = network(MNIST_IMG[iter]);
    //     printf("\n Real label: %d , predict: %lu \n", MNIST_LABELS[iter], index);
    // }


    // X data rows = total number of images, cols = size of input (Pool output of DNN)  
    
    // Y labels rows = total number of images, cols = Number of classes
 

    // X.rows = y.rows = 1;
    // X.cols = Pool2Out;
    // X.data = calloc(batch*X.cols, sizeof(float*));


    // y.cols = CLASSES;
    // y.data = calloc(1*y.cols, sizeof(float*));


    matrix X = make_matrix(TOTAL_IMAGE,Pool2Out);
    matrix y = make_matrix(TOTAL_IMAGE,CLASSES);

    int correct = 0;
    // for (i = 0; i < d.y.rows; ++i) {
    //     if (max_index(d.y.data + i*d.y.cols, d.y.cols) == max_index(p.data + i*p.cols, p.cols)) ++correct;
    // }
    // return (float)correct / d.y.rows;


    for(int imgi=0; imgi < TOTAL_IMAGE; imgi++)
    {

        index = network(MNIST_IMG[imgi]);
        if(MNIST_LABELS[imgi] == index){
            correct++;
        }
        // printf("\n Real label: %d , predict: %u \n", MNIST_LABELS[imgi], index);

        for (int i = 0; i < Pool2Out; i++){
            // imgi is the image index from which we get the DNN-pool-output
            // X.cols is the space for each inpute
            // i the index for the DNN-pool-output
            X.data[ imgi*X.cols + i] = pool2_out[i];
            // X.data[count*X.cols + i] = im.data[i];
        }
        
        // for (int i = 0; i < 10; i++)
        //             y.data[i] = 0;

        // the labels are one single float blob
        // labels represented by one-shot vector start from 0 thus the +1
        // imgi * y.cols find the start of each vector 
        y.data[ (imgi*y.cols)+ MNIST_LABELS[imgi] + 1] = 1;
        
    }
    
    printf("=> CMSIS accuracy: %f\n", (float)correct / TOTAL_IMAGE );



    data c;
    c.X = X;
    c.y = y;

    train_image_classifier(n, c, batch, iterations, rate, momentum, decay);
    printf("=> Training accuracy: %f\n", accuracy_net(n, c));
    // printf("=> Testing  accuracy: %f\n", accuracy_net(n, test));

    return 0;



    for(iter = 0; iter < iterations; ++iter){


        forward_net(n, c.X);
        float err = cross_entropy_loss(c.y, n.layers[n.n-1]);

        fprintf(stderr, " %d: Loss: %f\n", iter, err);
        backward_net(n);
        update_net(n, rate/1, momentum, decay);
        free_data(c);
    }

    

    // float ip1_wt[IP1_DIM * IP1_OUT] = IP1_WT;
    // // IP1_DIM * IP1_OUT 
    // for (int i=0; i < IP1_DIM * IP1_OUT; i++){
    //     n.layers[1].w.data [i]=  ip1_wt[i];
    //     // printf("%f\n",n.layers[1].w.data [i]);

    // }

    // n.layers[2] = make_connected_layer(32, 10, SOFTMAX);



    //TODO

    //pass forward to cmsis qnet

    // take the last out and use as data for fullyconnected of darknet

    // data b1 = random_batch(train, 1);
    // for (int i=0; i< b1.y.rows * b1.y.cols  ;i++){
    //     printf("%f",b1.y.data[i]);
    // }
    // printf("shallow %d\n", b1.y.shallow);
    // printf("X roes %d\n", b1.X.rows);
    // printf("X colds %d\n", b1.X.cols);
    // printf("y rows %d\n", b1.y.rows);
    // printf("y cols %d\n", b1.y.cols);
    // printf("%f\n", b1.X.data[787]);
    // printf("%f\n", b1.X.data[2]);
    // printf("%f\n", b1.X.data[3]);

    // train_image_classifier(n, train, batch, iters, rate, momentum, decay);

        // data b = random_batch(train, batch);
        // data b;


    // data random_batch(data d, int n)
    // {

        // int i, j;
        // for(i = 0; i < n; ++i){
        //     int ind = rand()%d.X.rows;
        //     for(j = 0; j < X.cols; ++j){
        //         X.data[i*X.cols + j] = d.X.data[ind*X.cols + j];
        //     }
        //     for(j = 0; j < y.cols; ++j){
        //         y.data[i*y.cols + j] = d.y.data[ind*y.cols + j];
        //     }
        // }




        //here b should be my data!
        // b.X.cols = DENSE_OUT;
        // for (int i = 0; i < DENSE_OUT; i++)
        //         b.X.data[i] = max_buffer1[i];



        // for (int i=0; i< b.y.rows * b.y.cols  ;i++){
        //     b.y.data[i] = 0;
        // }
        
        
        // for (int i=0; i< b.y.rows * b.y.cols  ;i++){
        //    printf("%f",b.y.data[i] );
        // }




    return 0;
}





uint32_t network(q7_t* input)
{
    
    // printf("run network");
	arm_convolve_HWC_q7_basic(input, CONV1_IM_DIM, CONV1_IM_CH, conv1_w, CONV1_OUT_CH, CONV1_KER_DIM, CONV1_PADDING,
						  CONV1_STRIDE, conv1_b, CONV1_BIAS_LSHIFT, CONV1_OUT_RSHIFT, conv1_out, CONV1_OUT_DIM,
						   conv_buffer, (q7_t *) fc_buffer);

    arm_maxpool_q7_HWC(conv1_out, POOL1_IM_DIM, POOL1_IM_CH, POOL1_KER_DIM, POOL1_PADDING, POOL1_STRIDE, POOL1_OUT_DIM, NULL, pool1_out);
    arm_relu_q7(pool1_out, POOL1_OUT_DIM * POOL1_OUT_DIM * CONV1_OUT_CH);
    arm_convolve_HWC_q7_basic(pool1_out, CONV2_IM_DIM, CONV2_IM_CH, conv2_w, CONV2_OUT_CH, CONV2_KER_DIM,
						  CONV2_PADDING, CONV2_STRIDE, conv2_b, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, conv2_out,
						  CONV2_OUT_DIM,  conv_buffer, NULL);
    arm_maxpool_q7_HWC(conv2_out, POOL2_IM_DIM, POOL2_IM_CH, POOL2_KER_DIM, POOL2_PADDING, POOL2_STRIDE, POOL2_OUT_DIM, NULL, pool2_out);
    arm_relu_q7(pool2_out, POOL2_OUT_DIM * POOL2_OUT_DIM * CONV2_OUT_CH);


    //clear buffer
    // memset(interface_out, 0, INTERFACE_OUT);
    
    // q15_t fc_buffer2[INTERFACE_DIM] = {0};

	arm_fully_connected_q7_opt(pool2_out, interface_w, INTERFACE_DIM, INTERFACE_OUT, INTERFACE_BIAS_LSHIFT, INTERFACE_OUT_RSHIFT, interface_b,
						  interface_out, fc_buffer);
	arm_relu_q7(interface_out, INTERFACE_OUT);
	arm_fully_connected_q7_opt(interface_out, linear_w, LINEAR_DIM, LINEAR_OUT, LINEAR_BIAS_LSHIFT, LINEAR_OUT_RSHIFT, linear_b,
						  linear_out, fc_buffer);
    arm_softmax_q7(linear_out, LINEAR_OUT, y_out);

	uint32_t index[1];
	q7_t result[1];
	uint32_t blockSize = sizeof(y_out);

	arm_max_q7(y_out, blockSize, result, index);

	return index[0];
}



