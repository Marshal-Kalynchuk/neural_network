#include "neural_network.cpp"
#include <vector>
using namespace std;

int main(){
  std::vector<int> num_nodes_per_hidden_layer = {3, 4};
  NeuralNetwork nn(3, 3, 2, num_nodes_per_hidden_layer);
  return 0;
}