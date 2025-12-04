#include "nnLayer.h"
#include "nnNetwork.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define MODEL_BAK "trained_network.bin"
#define TRAIN_SET "mnist_train.csv"
#define TEST_SET "mnist_test.csv"
#define TRAIN_SAMPLES 60000
#define TEST_SAMPLES 10000

#define MNIST_IMG_SIZE 784
#define MNIST_LABELS 10
#define MAX_LINE_LEN 8192 // Aumentato per sicurezza

// Funzione per liberare la memoria di un dataset
void free_data(double **inputs, double **targets, int samples)
{
    for (int i = 0; i < samples; i++)
    {
        free(inputs[i]);
        free(targets[i]);
    }
    free(inputs);
    free(targets);
    printf("Dataset memory freed.\n");
}

void load_mnist_data(const char *filename, int samples, double ***inputs, double ***targets)
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        fprintf(stderr, "Error: Unable to open file %s\n", filename);
        exit(1);
    }

    *inputs = (double **)malloc(samples * sizeof(double *));
    *targets = (double **)malloc(samples * sizeof(double *));

    char buffer[MAX_LINE_LEN];
    int count = 0;

    printf("Loading %d samples from %s...\n", samples, filename);

    while (fgets(buffer, MAX_LINE_LEN, file) && count < samples)
    {
        (*inputs)[count] = (double *)malloc(MNIST_IMG_SIZE * sizeof(double));
        (*targets)[count] = (double *)malloc(MNIST_LABELS * sizeof(double));

        for (int k = 0; k < MNIST_LABELS; k++)
            (*targets)[count][k] = 0.0;

        char *token = strtok(buffer, ",");
        if (token != NULL)
        {
            int label = atoi(token);
            if (label >= 0 && label < MNIST_LABELS)
            {
                (*targets)[count][label] = 1.0;
            }
        }

        for (int i = 0; i < MNIST_IMG_SIZE; i++)
        {
            token = strtok(NULL, ",");
            if (token != NULL)
            {
                (*inputs)[count][i] = (double)atoi(token) / 255.0;
            }
            else
            {
                (*inputs)[count][i] = 0.0;
            }
        }
        count++;
    }
    fclose(file);
    printf("Loading completed.\n");
}

int get_predicted_digit(double *output_array, int size)
{
    int max_index = 0;
    double max_val = output_array[0];
    for (int i = 1; i < size; i++)
    {
        if (output_array[i] > max_val)
        {
            max_val = output_array[i];
            max_index = i;
        }
    }
    return max_index;
}

void evaluate_accuracy(nnNetwork *network, double **inputs, double **targets, int samples, const char *name)
{
    printf("\n--- ACCURACY ON %s DATASET ---\n", name);
    int correct = 0;

    for (int i = 0; i < samples; i++)
    {
        double p_out[10];
        predict(network, inputs[i], p_out);

        int p = get_predicted_digit(p_out, 10);
        int t = get_predicted_digit(targets[i], 10);

        if (p == t)
            correct++;

        // Mostra solo i primi 5 risultati per verifica visiva
        if (i < 5)
        {
            printf("Sample %d: Pred: %d | Real: %d %s\n", i, p, t, (p == t) ? "(OK)" : "(FAIL)");
        }
    }
    double acc = (double)correct / samples * 100.0;
    printf(">>> Risultato: %.2f%% (%d/%d corretti)\n", acc, correct, samples);
}

int main()
{
    srand(time(NULL));
    nnNetwork *network = importNetwork(MODEL_BAK);

    if (network)
    {
        printf("found model backup at %s", MODEL_BAK);
        goto test;
    }

    network = nnCreateNetwork();

    int EPOCHS = 100;
    double LR = 0.2;

    // --- FASE 1: CARICAMENTO TRAINING SET ---
    double **train_inputs = NULL;
    double **train_targets = NULL;
    load_mnist_data(TRAIN_SET, TRAIN_SAMPLES, &train_inputs, &train_targets);

    // --- FASE 2: CREAZIONE RETE ---
    printf("Topology creation...\n");
    // 784 -> 64 -> 10 è leggero e veloce.
    // Se vuoi più precisione prova 784 -> 128 -> 10
    nnLayer *hidden = nnCreateLayer(64, MNIST_IMG_SIZE, ACTIVATION_SIGMOID);
    nnLayer *hidden_2 = nnCreateLayer(32, 64, ACTIVATION_SIGMOID);
    nnLayer *output = nnCreateLayer(MNIST_LABELS, 32, ACTIVATION_SIGMOID);

    init_layer_random(hidden);
    init_layer_random(hidden_2);
    init_layer_random(output);
    addLayerToNetwork(network, hidden);
    addLayerToNetwork(network, hidden_2);
    addLayerToNetwork(network, output);

    // --- FASE 3: TRAINING ---
    printf("Starting training (%d epochs, LR %.2f)...\n", EPOCHS, LR);
    train(network, train_inputs, train_targets, TRAIN_SAMPLES, LR, EPOCHS);
    exportNetwork(network, MODEL_BAK);

test:
    evaluate_accuracy(network, train_inputs, train_targets, TRAIN_SAMPLES, "TRAIN");

    free_data(train_inputs, train_targets, TRAIN_SAMPLES);

    double **test_inputs = NULL;
    double **test_targets = NULL;

    load_mnist_data(TEST_SET, TEST_SAMPLES, &test_inputs, &test_targets);

    evaluate_accuracy(network, test_inputs, test_targets, TEST_SAMPLES, "TEST");

    // --- FINAL CLEANUP ---
    free_data(test_inputs, test_targets, TEST_SAMPLES);
    nnFreeNetwork(network);

    return 0;
}