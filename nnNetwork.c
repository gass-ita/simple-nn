#include "nnNetwork.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

nnNetwork *nnCreateNetwork()
{
    nnNetwork *network = (nnNetwork *)malloc(sizeof(nnNetwork));
    if (network == NULL)
    {
        fprintf(stderr, "Memory allocation failed for nnNetwork\n");
        return NULL;
    }
    network->layer_count = 0;
    return network;
}

int addLayerToNetwork(nnNetwork *network, nnLayer *layer)
{
    if (network->layer_count >= MAX_LAYERS)
    {
        fprintf(stderr, "Cannot add more layers, maximum reached\n");
        return -1;
    }
    network->layers[network->layer_count] = layer;
    network->layer_count++;
    return 0;
}

void train(nnNetwork *network, double **target_input, double **target_output, int target_count, double learning_rate, int epochs)
{
    int layer_count = network->layer_count;
    nnLayer **layers = network->layers;
    int output_count = layers[layer_count - 1]->neuron_count;
    // Allocazione di buffer temporanei per i gradienti.
    // Usiamo MAX_NEURONS per essere sicuri che siano abbastanza grandi per qualsiasi layer.
    // 'next_layer_grads' contiene i gradienti in arrivo dal layer successivo (o dalla loss function).
    // 'prev_layer_grads' conterrà i gradienti calcolati da passare al layer precedente.
    double next_layer_grads[MAX_NEURONS];
    double prev_layer_grads[MAX_NEURONS];

    printf("Inizio training per %d epoche su %d esempi...\n", epochs, target_count);

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        double total_loss = 0.0;

        // Loop su ogni esempio del dataset (SGD)
        for (int i = 0; i < target_count; i++)
        {
            double *current_input = target_input[i];
            double *current_target = target_output[i];

            // --- FASE 1: Forward Propagation ---
            // Passiamo l'input attraverso tutti i layer
            for (int l = 0; l < layer_count; l++)
            {
                // forward scrive i risultati in layers[l]->outputs
                forward(layers[l], current_input, NULL);
                // L'output di questo layer diventa l'input del prossimo
                current_input = layers[l]->outputs;
            }

            // A questo punto, 'current_input' punta all'output dell'ultimo layer
            double *final_output = current_input;
            nnLayer *last_layer = layers[layer_count - 1];

            // --- Calcolo Loss e Gradiente Iniziale ---
            // Usiamo MSE (Mean Squared Error).
            // Gradiente rispetto all'output = 2 * (output - target)
            for (int j = 0; j < output_count; j++)
            {
                double error = final_output[j] - current_target[j];
                total_loss += error * error; // Accumulo per statistiche (Loss = sum((y-t)^2))

                // Scriviamo il gradiente iniziale nel buffer
                next_layer_grads[j] = 2.0 * error;
            }

            // --- FASE 2: Backward Propagation ---
            // Iteriamo dai layer finali verso i primi
            for (int l = layer_count - 1; l >= 0; l--)
            {
                nnLayer *curr_layer = layers[l];

                // Aggiorna i pesi e calcola i gradienti per il layer precedente
                backward(curr_layer, next_layer_grads, prev_layer_grads, learning_rate);

                // scambio i buffer per la prossima iterazione (indietro)
                memcpy(next_layer_grads, prev_layer_grads, curr_layer->input_count * sizeof(double));
            }
        }

        // Calcolo loss media per l'epoca (MSE)
        double average_loss = total_loss / target_count;
        if ((epoch + 1) % 100 == 0 || epoch == 0)
        { // Stampa ogni 100 epoche
            printf("Epoch %d/%d - Loss (MSE): %f\n", epoch + 1, epochs, average_loss);
        }
    }

    // Pulizia memoria temporanea

    printf("Training completato.\n");
}

void predict(nnNetwork *network, double *input, double *output)
{
    double *current_input = input;

    // Passiamo l'input attraverso tutti i layer
    for (int l = 0; l < network->layer_count; l++)
    {
        forward(network->layers[l], current_input, NULL);
        current_input = network->layers[l]->outputs;
    }

    // Copiamo l'output finale
    memcpy(output, current_input, network->layers[network->layer_count - 1]->neuron_count * sizeof(double));
}
