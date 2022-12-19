
#include <vector>
#include <random>

using namespace std;

class NeuralNetwork {
  public:
    NeuralNetwork(int num_inputs, int num_outputs, int num_hidden_layers, 
      int* num_nodes_per_hidden_layer): 
      num_inputs_(num_inputs),
      num_outputs_(num_outputs),
      num_hidden_layers_(num_hidden_layers),
      num_nodes_per_hidden_layer_(num_nodes_per_hidden_layer) {

      num_layers_ = num_hidden_layers_ + 1;

      weights_ = new double**[num_layers_];
      biases_ = new double*[num_layers_];

      activations_ = new double*[num_layers_];
      errors_ = new double*[num_layers_];

      for (int i = 0; i < num_layers_; i++) {
        int num_inputs;
        int num_outputs;
        if (i == 0) {
          // First layer has numInputs inputs
          num_inputs = num_inputs_;
        } else {
          // Other layers have numNodesPerHiddenLayer[i-1] inputs
          num_inputs = num_nodes_per_hidden_layer_[i - 1];
        }
        if (i == num_layers_ - 1) {
          // Output layer has numOutputs outputs
          num_outputs = num_outputs_;
        } else {
          // Other layers have numNodesPerHiddenLayer[i] outputs
          num_outputs = num_nodes_per_hidden_layer_[i];
        }
        // Initialize the weights and biases for this layer
        weights_[i] = new double*[num_inputs];
        biases_[i] = new double[num_inputs];
        for (int j = 0; j < num_inputs; j++) {
          weights_[i][j] = new double[num_outputs];
          for (int k = 0; k < num_outputs; k++){
            weights_[i][j][k] = randomValue(-0.5, 0.5);
          }
          biases_[i][j] = randomValue(0.5, 0.5);
        }
        // Initialize the activation and errors for this layer
        activations_[i] = new double[num_inputs];
        errors_[i] = new double[num_inputs];
      }
    }
    // Destructor
    ~NeuralNetwork(){
      for (int i = 0; i < num_layers_; i++){
        int num_inputs;
        int num_outputs;
        if (i == 0) {
          // First layer has numInputs inputs
          num_inputs = num_inputs_;
        } else {
          // Other layers have numNodesPerHiddenLayer[i-1] inputs
          num_inputs = num_nodes_per_hidden_layer_[i - 1];
        }
        if (i == num_layers_ - 1) {
          // Output layer has numOutputs outputs
          num_outputs = num_outputs_;
        } else {
          // Other layers have numNodesPerHiddenLayer[i] outputs
          num_outputs = num_nodes_per_hidden_layer_[i];
        }
        for (int j = 0; j < num_inputs; j++) {
          delete[] weights_[i][j];
        }
        delete[] biases_[i];
        delete[] weights_[i];
        delete[] activations_[i];
        delete[] errors_[i];
      }
      delete[] biases_;
      delete[] weights_;
      delete[] activations_;
      delete[] errors_;
    }

    // Feedforward method
    double* feedforward(double* inputs){

    }

    // Backpropagation method
    void backpropagate(double** targets, double learningRate);


    double randomValue(double min, double max) {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<> dist(min, max);
      return dist(gen);
    }
  private:

    int num_layers_;
    int num_inputs_;
    int num_outputs_;
    int num_hidden_layers_;

    int* num_nodes_per_hidden_layer_;
    double*** weights_;
    double** biases_;
    double** activations_;
    double** errors_;
};