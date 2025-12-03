#include "nnLayer.h"
#include "nnNetwork.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#define PI 3.14159265358979323846

int main()
{
    srand(time(NULL)); // Seed per numeri casuali
    nnNetwork *network = nnCreateNetwork();

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

    printf("Creazione Topologia Rete...\n");
    nnLayer *hidden = nnCreateLayer(20, input_cnt, ACTIVATION_LINEAR);
    nnLayer *hidden_2 = nnCreateLayer(30, 20, ACTIVATION_TANH);
    nnLayer *hidden_3 = nnCreateLayer(15, 30, ACTIVATION_TANH);
    nnLayer *hidden_4 = nnCreateLayer(15, 15, ACTIVATION_TANH);
    nnLayer *output = nnCreateLayer(output_cnt, 15, ACTIVATION_LINEAR);

    // Inizializzazione casuale dei pesi
    init_layer_random(hidden);
    init_layer_random(hidden_2);
    init_layer_random(hidden_3);
    init_layer_random(hidden_4);
    init_layer_random(output);

    addLayerToNetwork(network, hidden);
    addLayerToNetwork(network, hidden_2);
    addLayerToNetwork(network, hidden_3);
    addLayerToNetwork(network, hidden_4);
    addLayerToNetwork(network, output);

    // --- 3. TRAINING ---
    double learning_rate = 0.01; // 0.01 - 0.1 è un buon range
    int epochs = 10000;

    train(network, inputs, targets, samples, learning_rate, epochs);
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
        predict(network, in, predicted_output);
        double real_output = targets[i][0];
        double error = fabs(predicted_output[0] - real_output);
        total_error += error;
        printf("%6.2f | %9.6f | %8.6f | %6.6f\n", in[0], real_output, predicted_output[0], error);
    }
    // Test su un valore mai visto (PI greco) e (3*PI/2)
    double test_val[] = {PI};
    double predicted_pi[1];
    predict(network, test_val, predicted_pi);
    double real_pi = sin(PI);
    double error_pi = fabs(predicted_pi[0] - real_pi);
    printf("\nTest su x = PI:\n");
    printf("Predetto: %f | Reale: %f | Errore: %f\n", predicted_pi[0], real_pi, error_pi);
    total_error += error_pi;
    double test_val_2[] = {3 * PI / 2};
    double predicted_3pi2[1];
    predict(network, test_val_2, predicted_3pi2);
    double real_3pi2 = sin(3 * PI / 2);
    double error_3pi2 = fabs(predicted_3pi2[0] - real_3pi2);
    printf("\nTest su x = 3PI/2:\n");
    printf("Predetto: %f | Reale: %f | Errore: %f\n", predicted_3pi2[0], real_3pi2, error_3pi2);
    total_error += error_3pi2;
    double average_error = total_error / (samples / 5 + 2); // +2 per i test su PI e 3PI/2
    printf("\nErrore Medio sui test: %f\n", average_error);
    printf("\n--- PREDIZIONI DA 0 a 2PI ---\n");
    for (double x = 0; x <= 2 * PI; x += PI / 64)
    {
        double input_val[] = {x};
        double predicted_output[1];
        predict(network, input_val, predicted_output);
        printf("x: %6.2f | Predetto sin(x): %f | Reale sin(x): %f\n", x, predicted_output[0], sin(x));
    }
    printf("\n--- FINE PREDIZIONI ---\n");
    // --- 5. PULIZIA MEMORIA ---
    for (int i = 0; i < samples; i++)
    {
        free(inputs[i]);
        free(targets[i]);
    }
    free(inputs);
    free(targets);

    // TODO: free network layers and network itself
    return 0;
}