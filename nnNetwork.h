// include guard
#ifndef NNNETWORK_H
#define NNNETWORK_H

#include "nnLayer.h"

// TODO: consider dynamic allocation for layers (avoided for simplicity)
#define MAX_LAYERS 100

typedef struct nnNetwork
{
    int layer_count;
    nnLayer *layers[MAX_LAYERS]; // a list of pointers to layers
} nnNetwork;

void train(nnNetwork *network, double **target_input, double **target_output, int target_count, double learning_rate, int epochs);
void predict(nnNetwork *network, double *input, double *output);
nnNetwork *nnCreateNetwork();
int addLayerToNetwork(nnNetwork *network, nnLayer *layer);
int nnDumpNetwork(nnNetwork *network, const char *filename);
nnNetwork *nnLoadNetwork(const char *filename);
void nnFreeNetwork(nnNetwork *network);
#endif // NNNETWORK_H