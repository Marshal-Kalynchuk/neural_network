#include <vector>
#include <random>
#include <math.h>
#include <iostream>
#include <istream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <string.h>

using namespace std;

class NeuralNetwork {
  public:
    NeuralNetwork(int num_inputs, int num_outputs, int num_hidden_layers, 
      int* num_nodes_per_hidden_layer): 
      num_inputs_(num_inputs),
      num_outputs_(num_outputs),
      num_hidden_layers_(num_hidden_layers),
      num_nodes_per_hidden_layer_(num_nodes_per_hidden_layer){

      if (num_nodes_per_hidden_layer_ == nullptr) {
        std::cerr << "Error: num_nodes_per_hidden_ is nullptr" << std::endl;
        exit(1);
      }

      num_layers_ = num_hidden_layers_ + 1;

      weights_ = new double**[num_layers_];
      biases_ = new double*[num_layers_];

      for (int i = 0; i < num_layers_; i++) {

        int num_inputs = get_num_inputs(i);
        int num_outputs = get_num_outputs(i);

        // Initialize the weights and biases for this layer
        weights_[i] = new double*[num_outputs];
        biases_[i] = new double[num_outputs];

        for (int j = 0; j < num_outputs; j++) {
          weights_[i][j] = new double[num_inputs];
          for (int k = 0; k < num_inputs; k++){
            weights_[i][j][k] = randomValue(-0.5, 0.5);
          }
          biases_[i][j] = randomValue(-0.5, 0.5);
        }
        
      }
      activations_ = new double*[num_layers_ + 1];
      errors_ = new double*[num_layers_ + 1];
      for (int i = 0; i < num_layers_ + 1; i++){
        int num_nodes;
        if (i == 0) {
          // First layer has num_inputs nodes
          num_nodes = num_inputs_;
        } else if (i == num_layers_) {
          // Last layer has num_outputs nodes
          num_nodes = num_outputs_;
        } else {
          // Other layers have num_nodes_per_hidden_layer[i-1] nodes
          num_nodes = num_nodes_per_hidden_layer_[i - 1];
        }
        // Initialize the activation and errors for this layer
        activations_[i] = new double[num_nodes];
        errors_[i] = new double[num_nodes];
      }
    }

    // Destructor
    ~NeuralNetwork(){
      for (int i = 0; i < num_layers_; i++){
        int num_outputs = get_num_outputs(i);
        for (int j = 0; j < num_outputs; j++) 
          delete[] weights_[i][j];
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
      if (inputs == nullptr){
        std::cerr << "Error: Input is null" << std::endl;
        exit(1);
      }
      // Set the activations of the input layer to the input values
      for (int i = 0; i < num_inputs_; i++) {
        activations_[0][i] = inputs[i];
      }
      // Loop over the layers of the network
      for (int i = 0; i < num_layers_; i++) {

        int num_inputs = get_num_inputs(i);
        int num_outputs = get_num_outputs(i);
       
        // Calculate the weighted sum of the inputs for each node in the next layer
        for (int j = 0; j < num_outputs; j++) {
          double weighted_sum = 0.0;
          for (int k = 0; k < num_inputs; k++) {
            weighted_sum += activations_[i][k] * weights_[i][j][k];
          }
          // Add the bias of the node
          weighted_sum += biases_[i][j];
          // Calculate the activation of the node using the sigmoid function
          activations_[i + 1][j] = 1.0 / (1.0 + exp(-weighted_sum));
        }
      }
      // Return the activations of the output layer
      return activations_[num_layers_ - 1];
    }

    // Backpropagation method
    void backpropagate(double* targets, double learningRate){
      int num_layers = num_hidden_layers_ + 1; // +1 for output layer

      // Calculate the error in the output layer
      for (int i = 0; i < num_outputs_; i++){
        double output = activations_[num_layers][i];
        errors_[num_layers][i] = (targets[i] - output) * output * (1.0 - output);
      }

      // Calculate the errors in the hidden layers
      for (int i = num_layers - 1; i > 0; i--){
        // Calculate the number of nodes
        int num_inputs = 0;
        int num_outputs = 0;

        for (int j = 0; j < num_inputs; j++){

        }
      }
    }

    int get_num_inputs(int layer_index){
      return (layer_index == 0) ? num_inputs_ : num_nodes_per_hidden_layer_[layer_index - 1];
    }

    int get_num_outputs(int layer_index){
      return (layer_index == num_layers_ - 1) ? num_outputs_ : num_nodes_per_hidden_layer_[layer_index];
    }

    double randomValue(double min, double max) {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<> dist(min, max);
      return dist(gen);
    }
  private:
    int num_inputs_;
    int num_outputs_;
    int num_hidden_layers_;
    int num_layers_;

    int* num_nodes_per_hidden_layer_;
    double*** weights_;
    double** biases_;
    double** activations_;
    double** errors_;
};