#include "layer.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#define PI 3.14159265358979323846

void train(nnLayer **layers, int layer_count, double **target_input, double **target_output, int target_count, int input_count, int output_count, double learning_rate, int epochs)
{
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

void predict(nnLayer **layers, int layer_count, double *input, double *output)
{
    double *current_input = input;

    // Passiamo l'input attraverso tutti i layer
    for (int l = 0; l < layer_count; l++)
    {
        forward(layers[l], current_input, NULL);
        current_input = layers[l]->outputs;
    }

    // Copiamo l'output finale
    for (int j = 0; j < layers[layer_count - 1]->neuron_count; j++)
    {
        output[j] = current_input[j];
    }
}

int main()
{
    srand(time(NULL)); // Seed per numeri casuali
    // --- 1. PREPARAZIONE DATI DI TRAINING ---
    int samples = 50;   // Numero di esempi
    int input_cnt = 1;  // 1 solo input (x)
    int output_cnt = 1; // 1 solo output (sin(x))

    // Allocazione array di puntatori come richiesto dalla firma di `train`
    double **inputs = (double **)malloc(samples * sizeof(double *));
    double **targets = (double **)malloc(samples * sizeof(double *));

    printf("Generazione dati (0 a 2*PI)...\n");
    for (int i = 0; i < samples; i++)
    {
        inputs[i] = (double *)malloc(input_cnt * sizeof(double));
        targets[i] = (double *)malloc(output_cnt * sizeof(double));

        // Input: Valore tra 0 e 2 PI
        double x = ((double)i / samples) * (2 * PI);

        // Esempio: Normalizziamo l'input tra 0 e 1 per aiutare la rete a convergere meglio
        // (Le reti neurali preferiscono input piccoli).
        // Ma per questo test usiamo x grezzo per vedere se impara.
        inputs[i][0] = x;

        // Target: sin(x)
        targets[i][0] = sin(x);
    }

    // --- 2. CREAZIONE DELLA RETE ---
    // Architettura: 1 Input -> 10 Hidden (Tanh) -> 1 Output (Tanh)
    // Usiamo Tanh in output perché il seno va da -1 a 1.

    printf("Creazione Topologia Rete...\n");
    nnLayer *hidden = nnCreateLayer(10, input_cnt, ACTIVATION_TANH);
    nnLayer *output = nnCreateLayer(output_cnt, 10, ACTIVATION_TANH);

    // Inizializzazione casuale dei pesi
    init_layer_random(hidden);
    init_layer_random(output);

    nnLayer *network[] = {hidden, output};

    // --- 3. TRAINING ---
    // Parametri: learning_rate basso (0.01) e molte epoche (20000) per funzioni curve
    double learning_rate = 0.05; // 0.01 - 0.1 è un buon range
    int epochs = 10000;

    train(network, 2, inputs, targets, samples, input_cnt, output_cnt, learning_rate, epochs);

    // --- 4. TEST E VERIFICA ---
    printf("\n--- TEST RISULTATI ---\n");
    printf("   X    |   Reale   | Predetto | Errore \n");
    printf("--------|-----------|----------|--------\n");

    double total_error = 0;

    // Testiamo su alcuni punti sparsi
    for (int i = 0; i < samples; i += 5)
    { // Stampa ogni 5 campioni
        double *in = inputs[i];

        double predicted_output[1];
        predict(network, 2, in, predicted_output);
        double real_output = targets[i][0];
        double error = fabs(predicted_output[0] - real_output);
        total_error += error;
        printf("%6.2f | %9.6f | %8.6f | %6.6f\n", in[0], real_output, predicted_output[0], error);
    }

    // Test su un valore mai visto (PI greco)
    double test_val[] = {PI};
    double predicted_pi[1];
    predict(network, 2, test_val, predicted_pi);
    double real_pi = sin(PI);
    double error_pi = fabs(predicted_pi[0] - real_pi);
    printf("\nTest su x = PI:\n");
    printf("Predetto: %f | Reale: %f | Errore: %f\n", predicted_pi[0], real_pi, error_pi);
    total_error += error_pi;
    double average_error = total_error / (samples / 5 + 1); // +1 per il test su PI
    printf("\nErrore Medio sui test: %f\n", average_error);

    // --- 5. PULIZIA MEMORIA ---
    for (int i = 0; i < samples; i++)
    {
        free(inputs[i]);
        free(targets[i]);
    }
    free(inputs);
    free(targets);

    // print predicted values from 0 to 2PI with a step of PI/64
    printf("\n--- PREDIZIONI DA 0 a 2PI ---\n");
    for (double x = 0; x <= 2 * PI; x += PI / 64)
    {
        double input_val[] = {x};
        double predicted_output[1];
        predict(network, 2, input_val, predicted_output);
        printf("x: %6.2f | Predetto sin(x): %f | Reale sin(x): %f\n", x, predicted_output[0], sin(x));
    }

    nnFreeLayer(hidden);
    nnFreeLayer(output);

    return 0;
}
