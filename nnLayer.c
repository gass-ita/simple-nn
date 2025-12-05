#include "nnLayer.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

nnLayer *nnCreateLayer(int neuron_count, int input_count, nnActivationFunction activationFunction)
{
    if (neuron_count <= 0 || neuron_count > MAX_NEURONS || input_count <= 0 || input_count > MAX_NEURONS)
    {
        fprintf(stderr, "Invalid neuron or input count\n");
        return NULL;
    }
    nnLayer *layer = (nnLayer *)malloc(sizeof(nnLayer));
    if (layer == NULL)
    {
        fprintf(stderr, "Memory allocation failed for nnLayer\n");
        return NULL;
    }

    layer->neuron_count = neuron_count;
    layer->input_count = input_count;
    layer->activationFunction = activationFunction;

    // Initialize the inputs and outputs arrays with malloc (they will be of the same size during the entire lifecycle of the layer)
    layer->inputs = (double *)malloc(input_count * sizeof(double));
    layer->outputs = (double *)malloc(neuron_count * sizeof(double));

    return layer;
}

// Helper per inizializzare i pesi in modo casuale (Essenziale!)
// Inizializza pesi tra -1.0 e 1.0
void init_layer_random(nnLayer *layer)
{
    for (int i = 0; i < layer->neuron_count; i++)
    {
        // Init Bias
        layer->bias[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;

        // Init Pesi
        for (int j = 0; j < layer->input_count; j++)
        {
            layer->weights[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        }
    }
}

// forward takes in input an array of input of size input_count
// the output is pointed to the output of the network (it will be available until forward is called again)
void forward(nnLayer *layer, double *input, double **output)
{
    for (int i = 0; i < layer->neuron_count; i++)
    {
        double sum = layer->bias[i];
        for (int j = 0; j < layer->input_count; j++)
        {
            layer->inputs[j] = input[j];
            sum += layer->weights[i][j] * input[j];
        }
        layer->outputs[i] = activate(layer->activationFunction, sum);
    }
    *output = layer->outputs;
}

/**
 * layer: puntatore al layer corrente
 * outputGradient: gradienti ricevuti dal layer successivo (size: neuron_count)
 * inputGradient: array DOVE SCRIVERE i gradienti per il layer precedente (size: input_count)
 * learningRate: fattore di apprendimento (es. 0.01)
 */
void backward(nnLayer *layer, double *outputGradient, double *inputGradient, double learningRate)
{
    // 1. Puliamo l'inputGradient (sarà un accumulatore)
    for (int i = 0; i < layer->input_count; i++)
    {
        inputGradient[i] = 0.0;
    }

    // 2. Iteriamo su ogni neurone del layer corrente
    for (int j = 0; j < layer->neuron_count; j++)
    {
        // Calcolo del Delta del neurone (Errore Locale)
        // Delta = Gradiente * Derivata(Output)
        double derivative = activateDerivative(layer->activationFunction, layer->outputs[j]);
        double delta = outputGradient[j] * derivative;

        // 3. Aggiornamento del Bias
        // bias_new = bias_old - (learning_rate * delta)
        layer->bias[j] -= delta * learningRate;

        // 4. Calcolo gradienti per i pesi e propagazione indietro
        for (int i = 0; i < layer->input_count; i++)
        {
            // A. Calcolo del gradiente da passare indietro (Backpropagation)
            // dE/dX_i += Delta_j * W_ji
            // Nota: usiamo il peso attuale prima di aggiornarlo (o una copia),
            // ma per SGD semplice usare il peso corrente è un'approssimazione accettabile.
            inputGradient[i] += delta * layer->weights[j][i];

            // B. Aggiornamento del Peso
            // weight_new = weight_old - (learning_rate * input * delta)
            double weightGradient = layer->inputs[i] * delta;
            layer->weights[j][i] -= weightGradient * learningRate;
        }
    }
}

void nnFreeLayer(nnLayer *layer)
{
    if (!layer)
    {
        return;
    }

    free(layer->inputs);
    free(layer->outputs);
    free(layer);
    return;
}

void nnPrintLayerInfo(const nnLayer *layer)
{
    if (layer == NULL)
    {
        printf("Layer is NULL\n");
        return;
    }

    printf("Layer Info:\n");
    printf("Neurons: %d\n", layer->neuron_count);
    printf("Inputs per Neuron: %d\n", layer->input_count);
    printf("Activation Function: %d\n", layer->activationFunction);
    // print weights and biases
    for (int i = 0; i < layer->neuron_count; i++)
    {
        printf(" Neuron %d: Bias = %f | Weights = [", i, layer->bias[i]);
        for (int j = 0; j < layer->input_count; j++)
        {
            printf("%f", layer->weights[i][j]);
            if (j < layer->input_count - 1)
                printf(", ");
        }
        printf("]\n");
    }
}

// ACTIVATION FUNCTION IMPLEMENTATIONS

double activate(nnActivationFunction func, double x)
{
    switch (func)
    {
    case ACTIVATION_RELU:
        return x > 0 ? x : 0;
    case ACTIVATION_SIGMOID:
        return 1.0 / (1.0 + exp(-x));
    case ACTIVATION_TANH:
        return tanh(x);
    case ACTIVATION_LEAKYRELU:
        return x > 0 ? x : 0.01 * x;
    default:
        return x; // Identity as default
    }
}

// Calcola la derivata basata sul valore di OUTPUT del neurone
double activateDerivative(nnActivationFunction func, double outputVal)
{
    switch (func)
    {
    case ACTIVATION_RELU:
        return outputVal > 0 ? 1 : 0;
    case ACTIVATION_SIGMOID:
        // La derivata della sigmoide è f(x) * (1 - f(x))
        return outputVal * (1.0 - outputVal);
    case ACTIVATION_TANH:
        // La derivata di tanh è 1 - tanh^2(x)
        return 1.0 - (outputVal * outputVal);
    case ACTIVATION_LEAKYRELU:
        return outputVal > 0 ? 1 : 0.01;
    default:
        return 1; // Derivata dell'identità
    }
}
