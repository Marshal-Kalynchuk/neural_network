
#include <vector>
#include <random>

using namespace std;

class NeuralNetwork {
  public:
    NeuralNetwork(int num_inputs, int num_outputs, int num_hidden_layers, 
      std::vector<int> num_nodes_per_hidden_layer): 
      num_inputs_(num_inputs),
      num_outputs_(num_outputs),
      num_hidden_layers_(num_hidden_layers),
      num_nodes_per_hidden_layer_(num_nodes_per_hidden_layer) {

      int num_layers = num_hidden_layers_ + 1; // +1 for the output layer

      weights_.resize(num_layers);
      biases_.resize(num_layers);
      for (int i = 0; i < num_layers; i++) {
        int num_inputs;
        int num_outputs;
        if (i == 0) {
          // First layer has numInputs inputs
          num_inputs = num_inputs_;
        } else {
          // Other layers have numNodesPerHiddenLayer[i-1] inputs
          num_inputs = num_nodes_per_hidden_layer_[i - 1];
        }
        if (i == num_layers - 1) {
          // Output layer has numOutputs outputs
          num_outputs = num_outputs_;
        } else {
          // Other layers have numNodesPerHiddenLayer[i] outputs
          num_outputs = num_nodes_per_hidden_layer_[i];
        }
        // Initialize the weights and biases for this layer
        weights_[i].resize(num_outputs, std::vector<double>(num_inputs));
        biases_[i].resize(num_outputs);
        for (int j = 0; j < num_outputs; j++) {
          for (int k = 0; k < num_inputs; k++) {
            weights_[i][j][k] = randomValue(-0.5, 0.5);
          }
          biases_[i][j] = randomValue(-0.5, 0.5);
        }
      }
    }
    // Destructor
    ~NeuralNetwork();

    // Feedforward method
    std::vector<double> feedforward(std::vector<double> inputs);

    // Backpropagation method
    void backpropagate(std::vector<double> targets, double learningRate);

    // Other methods and member variables

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
    std::vector<int> num_nodes_per_hidden_layer_;
    std::vector<std::vector<vector<double>>> weights_;
    std::vector<std::vector<double>> biases_;
    std::vector<std::vector<double>> activations_;
    std::vector<std::vector<double>> errors_;
};