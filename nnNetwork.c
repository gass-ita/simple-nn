#include "nnNetwork.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

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

// Binary Export (Deep Copy to File)
int nnDumpNetwork(nnNetwork *network, const char *filename)
{
    FILE *f = fopen(filename, "wb");
    if (f == NULL)
    {
        fprintf(stderr, "Error opening file %s for writing\n", filename);
        return 1;
    }

    // 1. Write Network Metadata
    fwrite(&network->layer_count, sizeof(int), 1, f);

    // 2. Loop through layers and write Deep Data
    for (int i = 0; i < network->layer_count; i++)
    {
        nnLayer *layer = network->layers[i];

        // A. Write Layer Metadata
        fwrite(&layer->neuron_count, sizeof(int), 1, f);
        fwrite(&layer->input_count, sizeof(int), 1, f);
        int activation = (int)layer->activationFunction;
        fwrite(&activation, sizeof(int), 1, f);

        // B. Write Biases (Contiguous memory, single write)
        fwrite(layer->bias, sizeof(double), layer->neuron_count, f);

        // C. Write Weights (2D array, write row by row)
        for (int n = 0; n < layer->neuron_count; n++)
        {
            fwrite(layer->weights[n], sizeof(double), layer->input_count, f);
        }
    }

    fclose(f);
    printf("Network exported to %s (Binary)\n", filename);
    return 0;
}

// Binary Import (Rebuild from File)
nnNetwork *nnLoadNetwork(const char *filename)
{
    FILE *f = fopen(filename, "rb");
    if (f == NULL)
    {
        fprintf(stderr, "Error opening file %s for reading\n", filename);
        return NULL;
    }

    nnNetwork *network = nnCreateNetwork();
    if (!network)
    {
        fclose(f);
        return NULL;
    }

    // 1. Read Network Metadata
    int layer_count = 0;
    if (fread(&layer_count, sizeof(int), 1, f) != 1)
    {
        fprintf(stderr, "Failed to read layer count\n");
        fclose(f);
        return NULL; // Should probably free network here
    }

    // 2. Rebuild Layers
    for (int i = 0; i < layer_count; i++)
    {
        int neuron_count, input_count, activation_val;

        // A. Read Layer Metadata
        fread(&neuron_count, sizeof(int), 1, f);
        fread(&input_count, sizeof(int), 1, f);
        fread(&activation_val, sizeof(int), 1, f);

        // B. Create the layer structure in memory
        nnLayer *layer = nnCreateLayer(neuron_count, input_count, (nnActivationFunction)activation_val);
        if (!layer)
        {
            fclose(f);
            return NULL; // Should handle cleanup
        }

        // C. Read Biases
        fread(layer->bias, sizeof(double), neuron_count, f);

        // D. Read Weights
        for (int n = 0; n < neuron_count; n++)
        {
            fread(layer->weights[n], sizeof(double), input_count, f);
        }

        // Add reconstructed layer to network
        addLayerToNetwork(network, layer);
    }

    fclose(f);
    printf("Network imported from %s successfully (Binary).\n", filename);
    return network;
}

void train(nnNetwork *network, double **target_input, double **target_output, int target_count, double learning_rate, int epochs)
{
    int layer_count = network->layer_count;
    nnLayer **layers = network->layers;
    int output_count = layers[layer_count - 1]->neuron_count;
    // 'next_layer_grads' has the gradients from the next layer produced by the back propagation.
    // 'prev_layer_grads' will contain the gradients calculated to pass to the previous layer.
    double next_layer_grads[MAX_NEURONS];
    double prev_layer_grads[MAX_NEURONS];

    printf("Starting training %d epochs on %d samples...\n", epochs, target_count);
    clock_t total_start_time = clock();
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        clock_t epoch_start_time = clock();

        double total_loss = 0.0;

        // loop for each example given
        for (int i = 0; i < target_count; i++)
        {
            double *current_input = target_input[i];
            double *current_target = target_output[i];

            // Forward propagation, getting the prediction from the network
            for (int l = 0; l < layer_count; l++)
            {
                forward(layers[l], current_input, &current_input);
            }

            // the last value is stored in current_input
            double *final_output = current_input;
            nnLayer *last_layer = layers[layer_count - 1];

            // Initial gradient using MSE derivative
            for (int j = 0; j < output_count; j++)
            {
                double error = final_output[j] - current_target[j];
                total_loss += error * error; // Accumulate for statistics (Loss = sum((y-t)^2))

                // Write the initial gradient to the buffer
                next_layer_grads[j] = 2.0 * error;
            }

            // backward propagation through layers
            for (int l = layer_count - 1; l >= 0; l--)
            {
                nnLayer *curr_layer = layers[l];

                // Update weights and calculate gradients for the previous layer
                backward(curr_layer, next_layer_grads, prev_layer_grads, learning_rate);

                // swap buffers for the next iteration (backward)
                memcpy(next_layer_grads, prev_layer_grads, curr_layer->input_count * sizeof(double));
            }
        }
        // --- Calcoli Statistiche Epoca ---

        // 1. Loss Media
        double average_loss = total_loss / target_count;

        // 2. Tempo trascorso in questa epoca
        clock_t now = clock();
        double epoch_duration = (double)(now - epoch_start_time) / CLOCKS_PER_SEC;

        // 3. Tempo totale trascorso dall'inizio
        double total_elapsed = (double)(now - total_start_time) / CLOCKS_PER_SEC;

        // 4. Stima ETA (basata sulla media del tempo per epoca finora)
        double avg_time_per_epoch = total_elapsed / (epoch + 1);
        int remaining_epochs = epochs - (epoch + 1);
        double eta_seconds = avg_time_per_epoch * remaining_epochs;

        // Formattazione ETA in ore/min/sec per leggibilità
        int eta_h = (int)eta_seconds / 3600;
        int eta_m = ((int)eta_seconds % 3600) / 60;
        int eta_s = (int)eta_seconds % 60;

        // 5. Percentuale completamento
        double progress = ((double)(epoch + 1) / epochs) * 100.0;

        // Stampa (puoi cambiare la condizione % 1 per stampare meno frequentemente)
        if ((epoch + 1) % 1 == 0 || epoch == 0)
        {
            printf("Epoch %d/%d [%.1f%%] | Loss: %.6f | Time: %.2fs | ETA: %02d:%02d:%02d\n",
                   epoch + 1,
                   epochs,
                   progress,
                   average_loss,
                   epoch_duration,
                   eta_h, eta_m, eta_s);
        }
    }
    printf("Training completed\n");
}

// forwards the whole network and copy the output to the specified output array that must be allocated from the caller
void predict(nnNetwork *network, double *input, double *output)
{
    double *current_input = input;

    for (int l = 0; l < network->layer_count; l++)
    {
        forward(network->layers[l], current_input, &current_input);
    }

    // copy the final output to the output given by the user
    memcpy(output, current_input, network->layers[network->layer_count - 1]->neuron_count * sizeof(double));
}

void nnFreeNetwork(nnNetwork *network)
{
    if (!network)
    {
        return;
    }

    for (int i = 0; i < network->layer_count; i++)
    {
        nnFreeLayer(network->layers[i]);
    }

    free(network);
}
