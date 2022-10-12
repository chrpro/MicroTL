#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "uwnet.h"


// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_maxpool_layer(layer l, matrix in)
{
    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    matrix out = make_matrix(in.rows, outw*outh*l.channels);
    printf("size of max outt:%d \n",  outw*outh*l.channels);
    // iterate over the input and fill in the output with max values
    int n; // image idx in batch
    int ch; // channel idx in image
    int i, j; // pooling window idx (upleft) in image
    int r, c; // idx in out
    int x, y; // idx in pooling window (consistent with i, j)
    float max; // hold the current max in window
    int pix; // value of current pixel

    for (n = 0; n < in.rows; ++n) {
        for (ch = 0; ch < l.channels; ++ch) {
            for (r = 0, i = 0; r < outh; ++r, i += l.stride) {
                for (c = 0, j = 0; c < outw; ++c, j += l.stride) {
                    max = in.data[in.cols*n
                                  + l.width*l.height*ch
                                  + l.width*i + j];
                    for (x = 0; x < l.size; ++x) {
                        if (i+x < l.height) {
                            for (y = 0; y < l.size; ++y) {
                                if (j+y < l.width) {
                                    pix = in.data[in.cols*n
                                                  + l.width*l.height*ch
                                                  + l.width*(i+x) + (j+y)];
                                    if (pix > max) {
                                        max = pix;
                                    }
                                }
                            }
                        }
                    }
                    out.data[out.cols*n
                             + outw*outh*ch
                             + outw*r + c] = max;
                }
            }
        }
    }

    l.in[0] = in;
    free_matrix(l.out[0]);
    l.out[0] = out;
    free_matrix(l.delta[0]);
    l.delta[0] = make_matrix(out.rows, out.cols);
    activate_matrix(out, RELU);
    return out;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix prev_delta: error term for the previous layer
void backward_maxpool_layer(layer l, matrix prev_delta)
{
    matrix in    = l.in[0];
    matrix out   = l.out[0];
    matrix delta = l.delta[0];

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    // find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.
    int n; // image idx in batch
    int ch; // channel idx in image
    int i, j; // pooling window idx (upleft) in image
    int r, c; // idx in out
    int x, y; // idx in pooling window (consistent with i, j)
    int pix_idx; // direct idx of current pixel in in(input)
    int max_idx; // direct idx of max pixel in in(input)

    for (n = 0; n < in.rows; ++n) {
        for (ch = 0; ch < l.channels; ++ch) {
            for (r = 0, i = 0; r < outh; ++r, i += l.stride) {
                for (c = 0, j = 0; c < outw; ++c, j += l.stride) {
                    max_idx = in.cols*n
                              + l.width*l.height*ch
                              + l.width*i + j;
                    for (x = 0; x < l.size; ++x) {
                        if (i+x < l.height) {
                            for (y = 0; y < l.size; ++y) {
                                if (j+y < l.width) {
                                    pix_idx = in.cols*n
                                              + l.width*l.height*ch
                                              + l.width*(i+x) + (j+y);
                                    if (in.data[pix_idx] > in.data[max_idx]) {
                                        max_idx = pix_idx;
                                    }
                                }
                            }
                        }
                    }
                    prev_delta.data[max_idx] += delta.data[out.cols*n
                                                          + outw*outh*ch
                                                          + outw*r + c];
                }
            }
        }
    }

}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer l, float rate, float momentum, float decay)
{
    printf("update maxpool\n");
}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(int w, int h, int c, int size, int stride)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.size = size;
    l.stride = stride;
    l.in = calloc(1, sizeof(matrix));
    l.out = calloc(1, sizeof(matrix));
    l.delta = calloc(1, sizeof(matrix));
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update   = update_maxpool_layer;
    return l;
}
