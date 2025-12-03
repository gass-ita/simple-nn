#define MAX_NEURONS 1024

typedef enum ActivationFunction
{
    ACTIVATION_RELU,
    ACTIVATION_SIGMOID,
    ACTIVATION_TANH,
    ACTIVATION_LEAKYRELU
} nnActivationFunction;

typedef struct nnLayer
{
    int neuron_count;
    int input_count;
    double bias[MAX_NEURONS];
    double weights[MAX_NEURONS][MAX_NEURONS];

    // backward propagation arrays
    double *inputs;
    double *outputs;

    nnActivationFunction activationFunction;
} nnLayer;

nnLayer *nnCreateLayer(int neuron_count, int input_count, nnActivationFunction activationFunction);
void nnFreeLayer(nnLayer *layer);
void nnPrintLayerInfo(const nnLayer *layer);
void forward(nnLayer *layer, double *input, double *output);
void backward(nnLayer *layer, double *outputGradient, double *inputGradient, double learningRate);
double activate(nnActivationFunction func, double x);
double activateDerivative(nnActivationFunction func, double outputVal);
void init_layer_random(nnLayer *layer);