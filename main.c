#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


// Function to perform tanh activation
double tanh_activation(double x) {
    return tanh(x);
}

// Function to calculate the derivative of tanh
double dTanh(double x) {
    return 1.0 - tanh(x) * tanh(x);
}

// Function to initialize weights randomly
double init_weights() {
    return 2.0 * (((double)rand()) / ((double)RAND_MAX)) - 1.0;
}

// Function to normalize data
void normalize(double *data, size_t size) {
    double min = data[0], max = data[0];
    for (size_t i = 1; i < size; i++) {
        if (data[i] < min)
            min = data[i];
        if (data[i] > max)
            max = data[i];
    }

    for (size_t i = 0; i < size; i++) {
        data[i] = (data[i] - min) / (max - min);
    }
}

// Function to shuffle array elements
void shuffle(int *array, size_t n) {
    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; i++) {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

#define numInputs 3
#define numHiddenNodes 8
#define numOutputs 1
#define numTrainingSets 8

// Function to generate random inputs and save them to a file
void generateRandomInputs(const char *fileName) {
    FILE *file = fopen(fileName, "w");
    if (file == NULL) {
        printf("Error opening file.\n");
        return;
    }

    srand(time(NULL)); // Seed for random number generation

    for (int i = 0; i < numTrainingSets; i++) {
        fprintf(file, "%d %d %d\n", rand() % 3000 + 1000, rand() % 6 + 1, rand() % 4 + 1);
    }

    fclose(file);
}

// Function to read inputs from a file
void readInputsFromFile(const char *fileName, double training_inputs[numTrainingSets][numInputs]) {
    FILE *file = fopen(fileName, "r");
    if (file == NULL) {
        printf("Error opening file.\n");
        return;
    }

    for (int i = 0; i < numTrainingSets; i++) {
        fscanf(file, "%lf %lf %lf", &training_inputs[i][0], &training_inputs[i][1], &training_inputs[i][2]);
    }

    fclose(file);
}

int main(void) {
    FILE *output_file = fopen("tta_stream_v1.out", "w");
    if (output_file == NULL) {
        printf("Error opening output file.\n");
        return -1;
    }

    generateRandomInputs("tta_stream_v1.in");

    const double lr = 0.1;

    double hiddenLayer[numHiddenNodes];
    double outputLayer[numOutputs];

    double hiddenLayerBias[numHiddenNodes];
    double outputLayerBias[numOutputs];

    double hiddenWeights[numInputs][numHiddenNodes];
    double outputWeights[numHiddenNodes][numOutputs];

    double training_inputs[numTrainingSets][numInputs];
    double training_outputs[numTrainingSets][numOutputs] = {
        {200000}, {300000}, {150000}, {400000},
        {250000}, {180000}, {320000}, {500000}
    };

    readInputsFromFile("tta_stream_v1.in", training_inputs);

    for (size_t i = 0; i < numInputs; i++) {
        normalize(training_inputs[i], numTrainingSets);
    }

    for (int i = 0; i < numInputs; i++) {
        for (int j = 0; j < numHiddenNodes; j++) {
            hiddenWeights[i][j] = init_weights();
        }
    }

    for (int i = 0; i < numHiddenNodes; i++) {
        for (int j = 0; j < numOutputs; j++) {
            outputWeights[i][j] = init_weights();
        }
    }

    for (int i = 0; i < numOutputs; i++) {
        outputLayerBias[i] = init_weights();
    }

    int trainingSetOrder[] = {0, 1, 2, 3, 4, 5, 6, 7};
    int numberofEpochs = 1000;

    for (int epoch = 0; epoch < numberofEpochs; epoch++) {
        shuffle(trainingSetOrder, numTrainingSets);

        for (int x = 0; x < numTrainingSets; x++) {
            int i = trainingSetOrder[x];

            // Forward pass
            for (int j = 0; j < numHiddenNodes; j++) {
                double activation = hiddenLayerBias[j];
                for (int k = 0; k < numInputs; k++) {
                    activation += training_inputs[i][k] * hiddenWeights[k][j];
                }
                hiddenLayer[j] = tanh_activation(activation);
            }

            for (int j = 0; j < numOutputs; j++) {
                double activation = outputLayerBias[j];
                for (int k = 0; k < numHiddenNodes; k++) {
                    activation += hiddenLayer[k] * outputWeights[k][j];
                }
                outputLayer[j] = activation; // Linear activation for output layer
            }

            // Print to file instead of console
            fprintf(output_file, "Area: %g sq.ft, Bedrooms: %g, Parking: %g    Price: %g    Predicted Price: %g \n",
                   training_inputs[i][0], training_inputs[i][1], training_inputs[i][2],
                   training_outputs[i][0], outputLayer[0]);

            // Backpropagation
            double deltaOutput[numOutputs];
            for (int j = 0; j < numOutputs; j++) {
                double error = (training_outputs[i][j] - outputLayer[j]);
                deltaOutput[j] = error;
            }

            double deltaHidden[numHiddenNodes];
            for (int j = 0; j < numHiddenNodes; j++) {
                double error = 0.0;
                for (int k = 0; k < numOutputs; k++) {
                    error += deltaOutput[k] * outputWeights[j][k];
                }
                deltaHidden[j] = error * dTanh(hiddenLayer[j]);
            }

            // Update weights and biases
            for (int j = 0; j < numOutputs; j++) {
                outputLayerBias[j] += deltaOutput[j] * lr;
                for (int k = 0; k < numHiddenNodes; k++) {
                    outputWeights[k][j] += hiddenLayer[k] * deltaOutput[j] * lr;
                }
            }

            for (int j = 0; j < numHiddenNodes; j++) {
                hiddenLayerBias[j] += deltaHidden[j] * lr;
                for (int k = 0; k < numInputs; k++) {
                    hiddenWeights[k][j] += training_inputs[i][k] * deltaHidden[j] * lr;
                }
            }
        }
    }

    // Print final weights and biases to the output file
    fprintf(output_file, "Final Hidden Weights\n");
    for (int j = 0; j < numHiddenNodes; j++) {
        fprintf(output_file, "[ ");
        for (int k = 0; k < numInputs; k++) {
            fprintf(output_file, "%f ", hiddenWeights[k][j]);
        }
        fprintf(output_file, "]\n");
    }

    fprintf(output_file, "Final Hidden Biases\n[ ");
    for (int j = 0; j < numHiddenNodes; j++) {
        fprintf(output_file, "%f ", hiddenLayerBias[j]);
    }
    fprintf(output_file, "]\n");

    fprintf(output_file, "Final Output Weights\n");
    for (int j = 0; j < numOutputs; j++) {
        fprintf(output_file, "[ ");
        for (int k = 0; k < numHiddenNodes; k++) {
            fprintf(output_file, "%f ", outputWeights[k][j]);
        }
        fprintf(output_file, "]\n");
    }

    fprintf(output_file, "Final Output Biases\n[ ");
    for (int j = 0; j < numOutputs; j++) {
        fprintf(output_file, "%f ", outputLayerBias[j]);
    }
    fprintf(output_file, "]\n");

    fclose(output_file); // Close the output file

    return 0;
}

