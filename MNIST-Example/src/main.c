//Standart libs
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>

//Training libs
#include "microTL.h"
#include "test.h"
#include "args.h"

// CMSIS Libs
#include "arm_math.h"
#include "arm_nnfunctions.h"

//Train images
#include "tain_data.h"
// Test images
#include "test_data.h"

// Neural Network
#include "parameters.h"
#include "weights.h"



# define CLASSES 10
# define NUM_OF_EXPRIEMENTS 1

uint32_t network(q7_t* input);




static q7_t conv1_out[CONV1_OUT_CH*CONV1_OUT_DIM*CONV1_OUT_DIM];
static q7_t pool1_out[CONV1_OUT_CH*POOL1_OUT_DIM*POOL1_OUT_DIM];
static q7_t conv2_out[CONV2_OUT_CH*CONV2_OUT_DIM*CONV2_OUT_DIM];
static q7_t pool2_out[CONV2_OUT_CH*POOL2_OUT_DIM*POOL2_OUT_DIM];
static q7_t conv3_out[CONV3_OUT_CH*CONV3_OUT_DIM*CONV3_OUT_DIM];
// static q7_t pool3_out[CONV3_OUT_CH*POOL3_OUT_DIM*POOL3_OUT_DIM];
static q7_t fc1_out[INTERFACE_OUT];
static q7_t y_out[INTERFACE_OUT];

static q7_t conv1_w[CONV1_WT_SHAPE] = CONV1_WT;
static q7_t conv1_b[CONV1_BIAS_SHAPE] = CONV1_BIAS;
static q7_t conv2_w[CONV2_WT_SHAPE] =  CONV2_WT;
static q7_t conv2_b[CONV2_BIAS_SHAPE] = CONV2_BIAS;
static q7_t conv3_w[CONV3_WT_SHAPE] =  CONV3_WT;
static q7_t conv3_b[CONV3_BIAS_SHAPE] = CONV3_BIAS;
static q7_t fc1_w[INTERFACE_WT_SHAPE] = INTERFACE_WT;
static q7_t fc1_b[INTERFACE_BIAS_SHAPE] = INTERFACE_BIAS;

static q15_t conv_buffer[MAX_CONV_BUFFER_SIZE];
static q15_t fc_buffer[MAX_FC_BUFFER];

void fc_2l_network_init(net * fully_con, int input_f_layer, int out_f_layer );

int main(void)
{

     FILE* fp ;
    //initialize the random function
    //seed with time in microseconds (works with Unix systems)
    struct timeval t1;
    gettimeofday(&t1, NULL);
    srand(t1.tv_usec * t1.tv_sec);

    const char chalmersBanner[] = {
		"============================================================\n"
		"Tranfers Learning on low-power IoT devices                  \n"
		"Chalmers University                                         \n"
		"============================================================\n"
	}; 

    printf(chalmersBanner);
    uint32_t index = 0;
    // int PoolOut = CONV3_OUT_CH*POOL3_OUT_DIM*POOL3_OUT_DIM;
    // int PoolOut = CONV2_OUT_CH*CONV2_OUT_DIM*CONV2_OUT_DIM;
    int PoolOut = CONV3_OUT_CH*CONV3_OUT_DIM*CONV3_OUT_DIM;


    int fc_layer_out = 128;
    // printf("conv out %d: \n",PoolOut);
    
    int batch = 4;

    float rate = .0001;
    float momentum = .9;
    float decay = .0002;

    // net transfer_learning = {0}; 
    // fc_2l_network_init(&transfer_learning, PoolOut, fc_layer_out);


    data training_data;


    matrix q_X0 = make_matrix(NODE_0_TOTAL_TRAIN_IMAGES,PoolOut);
    matrix q_Y0 = make_matrix(NODE_0_TOTAL_TRAIN_IMAGES,CLASSES);

 
    float scale = 1 << CONV3_OUT_Q;

    for(int imgi=0; imgi < NODE_0_TOTAL_TRAIN_IMAGES; imgi++)
    {
        index = network(NODE_0_TRAIN_IMAGES[imgi]);
        for (int i = 0; i < PoolOut; i++){
            // q_X0.data[ imgi*q_X0.cols + i] = (float) pool3_out[i] ;/// (float) pow(2,CONV3_OUT_Q) ;
            q_X0.data[ imgi*q_X0.cols + i] =  (float)conv3_out[i] / scale;
        
        }
        q_Y0.data[ (imgi*q_Y0.cols)+ NODE_0_TRAIN_LABELS[imgi] ] = 1;     
    }
    training_data.X = q_X0;
    training_data.y = q_Y0;

    matrix X_test = make_matrix(TOTAL_TEST_IMAGES,PoolOut);
    matrix Y_test = make_matrix(TOTAL_TEST_IMAGES,CLASSES);

    for(int imgi=0; imgi < TOTAL_TEST_IMAGES; imgi++)
    {
        index = network(TEST_IMAGES[imgi]);
        for (int i = 0; i < PoolOut; i++){
            X_test.data[ imgi*X_test.cols + i] = (float)conv3_out[i] / scale; 
            // X_test.data[ imgi*X_test.cols + i] = pool3_out[i] ;// / (float) pow(2,CONV3_OUT_Q) ;
        }
        Y_test.data[ (imgi*Y_test.cols)+ TEST_LABELS[imgi] ] = 1;      
    }

    data test;
    test.X=X_test;
    test.y=Y_test;

    int iterations = NODE_0_TOTAL_TRAIN_IMAGES / batch;
    printf("iterations = %d\n",iterations);
    

    char filename_format[] = "Accuracy_%d.csv";
    char filename[sizeof(filename_format) + 3];  // for up to 4 digit numbers

    for (int experiment = 1 ; experiment <= NUM_OF_EXPRIEMENTS; experiment ++){
        
        snprintf(filename, sizeof(filename), filename_format, experiment);
        printf("%s\n",filename);
        FILE *fp = fopen(filename, "w");  // open for reading
	
        fprintf(fp, "Train,Vallidation\n");	

        net transfer_learning = {0}; 
        fc_2l_network_init(&transfer_learning, PoolOut, fc_layer_out);
        for ( int epoch = 1; epoch < 15 ; epoch++){

            train_image_classifier(transfer_learning, training_data, batch, iterations,  rate, momentum, decay);
            float train_acc = accuracy_net(transfer_learning, training_data);
            printf("Epoch:%d=> TF Training accuracy: %f\n",epoch,train_acc);
            float test_acc = accuracy_net(transfer_learning, test);
            fprintf(fp, "%f,%f\n", train_acc, test_acc);

        }
        fclose(fp);
        float test_acc = accuracy_net(transfer_learning, test);
        printf("Test  accuracy: %f\n\n", test_acc);
        free(&transfer_learning.layers);

    }


    // for ( int iterations = 50; iterations < 600 ; iterations+=50){
    //     net transfer_learning = {0}; 
    //     fc_2l_network_init(&transfer_learning, PoolOut, fc_layer_out);
    //     train_image_classifier(transfer_learning, training_data, batch, iterations,  rate, momentum, decay);
    //     float train_acc = accuracy_net(transfer_learning, training_data);
    //     printf("Iters:%d=> TF Training accuracy: %f\n",iterations,train_acc);
    //     float test_acc = accuracy_net(transfer_learning, test);
    //     printf("Iters:%d=> TF Testing  accuracy: %f\n\n",iterations, test_acc);
    //     			// save results
    //     fprintf(fp, "%d,%f\n", iterations, test_acc);
    //     // free_connected_layer(transfer_learning.layers[0]);
    //     // free(&transfer_learning.layers[0]);
    //     // free(&transfer_learning.layers[1]);
    // }

    // fclose(fp);


    return 0;

}



uint32_t network(q7_t* input)
{


    	arm_convolve_HWC_q7_basic(input, CONV1_IM_DIM, CONV1_IM_CH, conv1_w, 
                        CONV1_OUT_CH, CONV1_KER_DIM, CONV1_PADDING, CONV1_STRIDE, 
                        conv1_b, CONV1_BIAS_LSHIFT, 
                          CONV1_OUT_RSHIFT, conv1_out, CONV1_OUT_DIM,
						  conv_buffer, NULL);

        arm_maxpool_q7_HWC(conv1_out, POOL1_IM_DIM, POOL1_IM_CH, POOL1_KER_DIM, POOL1_PADDING,
         POOL1_STRIDE, POOL1_OUT_DIM, (q7_t *) conv_buffer, pool1_out);
        arm_relu_q7(pool1_out, POOL1_OUT_DIM * POOL1_OUT_DIM * CONV1_OUT_CH);

        arm_convolve_HWC_q7_basic(pool1_out, CONV2_IM_DIM, CONV2_IM_CH, conv2_w, CONV2_OUT_CH, CONV2_KER_DIM,
						  CONV2_PADDING, CONV2_STRIDE, conv2_b, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, conv2_out,
						  CONV2_OUT_DIM, conv_buffer, NULL);

        arm_maxpool_q7_HWC(conv2_out, POOL2_IM_DIM, POOL2_IM_CH, POOL2_KER_DIM, POOL2_PADDING, POOL2_STRIDE, POOL2_OUT_DIM, (q7_t *) conv_buffer, pool2_out);
        arm_relu_q7(pool2_out, POOL2_OUT_DIM * POOL2_OUT_DIM * CONV2_OUT_CH);

        arm_convolve_HWC_q7_basic(pool2_out, CONV3_IM_DIM, CONV3_IM_CH, conv3_w, CONV3_OUT_CH, CONV3_KER_DIM,
						  CONV3_PADDING, CONV3_STRIDE, conv3_b, CONV3_BIAS_LSHIFT, CONV3_OUT_RSHIFT, conv3_out,
						  CONV3_OUT_DIM, conv_buffer, NULL);

   return 0 ;

        // arm_maxpool_q7_HWC(conv3_out, POOL3_IM_DIM, POOL3_IM_CH, POOL3_KER_DIM, POOL3_PADDING, POOL3_STRIDE, POOL3_OUT_DIM,  (q7_t *) conv_buffer, pool3_out);
        
     
        // arm_relu_q7(pool3_out, POOL3_OUT_DIM * POOL3_OUT_DIM * CONV3_OUT_CH);

        // arm_fully_connected_q7_opt(pool3_out, fc1_w, INTERFACE_DIM, INTERFACE_OUT, INTERFACE_BIAS_LSHIFT, INTERFACE_OUT_RSHIFT, fc1_b,
		// 				  fc1_out,  fc_buffer);
    

        arm_softmax_q7(fc1_out, INTERFACE_OUT, y_out);


        uint32_t index[1];
        q7_t result[1];
        uint32_t blockSize = sizeof(y_out);

        arm_max_q7(y_out, blockSize, result, index);
        //printf("Classified class %i\n", index[0]);

        return index[0];
}



void fc_2l_network_init(net * fully_con, int input_f_layer, int out_f_layer ){
    fully_con->n = 2;
    fully_con->layers = calloc(2, sizeof(layer));
    fully_con->layers[0] = make_connected_layer(input_f_layer, out_f_layer, RELU);
    fully_con->layers[1] = make_connected_layer(out_f_layer, CLASSES, SOFTMAX);
}
