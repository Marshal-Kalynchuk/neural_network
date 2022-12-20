#include "neural_network.cpp"
#include <iostream>
#include <istream>
#include <vector>
using namespace std;

int main(){

  int num_inputs = 3;
  int num_outputs = 2;
  int num_hidden_layers = 2;
  int num_nodes_per_hidden_layer[2] = {2, 3};

  NeuralNetwork nn(num_inputs, num_outputs, num_hidden_layers, num_nodes_per_hidden_layer);
  double input[3] = {1.0, 2.0, 3.0};
  double* result = nn.feedforward(input);
  for (int i = 0; i < num_outputs; i++) {
    std::cout << result[i] << std::endl;
  }

  return 0;
}