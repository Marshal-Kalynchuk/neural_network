#include "neural_network.cpp"
#include <iostream>
#include <istream>
#include <vector>
#include <sstream>
using namespace std;

template<typename T> void print_element(T t, const int& width){
  cout << left << setw(width) << setfill(' ') << t;
}

int main(){

  // Display parameters
  const int str_width = 10;
  const int num_width = 10;
  std::cout << std::fixed;
  std::cout << std::setprecision(2);

  // Dataset parameters
  int data_size = 28*28;
  int target_size = 10;

  // Initialize arrays to hold training data
  int train_size = 60000;
  double** train_data = new double*[train_size];
  double** train_targets = new double*[train_size];
  for (int i = 0; i < train_size; i++){
    train_data[i] = new double[data_size];
    train_targets[i] = new double[target_size];
  }

  // Initialize arrays to hold testing data
  int test_size = 10000;
  double** test_data = new double*[test_size];
  double** test_targets = new double*[test_size];
  for (int i = 0; i < test_size; i++){
    test_data[i] = new double[data_size];
    test_targets[i] = new double[target_size];
  }

  // dataset files
  char* train_file = "mnist_train.csv";
  char* test_file = "mnist_test.csv";

  // Load the training data
  std::ifstream train_file_stream(train_file);
  if (train_file_stream.is_open()){
    std::string line;
    int count = 0;
    int label = 0;
    while (std::getline(train_file_stream, line))
    {
        std::istringstream iss(line);
        iss >> label;
        train_targets[count][label] = 1; 
        for (int i = 0; i < data_size; i++)
        {
            iss >> train_data[count][i];
        }
        count++;
    }
    train_file_stream.close();
  } else {
    cerr << "Error, could not open file: " << train_file << endl;
    exit(1);
  }

  // Neural network parameters
  int learning_rate = 2;
  int num_inputs = data_size;
  int num_outputs = target_size;
  int num_hidden_layers = 2;
  int num_nodes_per_hidden_layer[3] = {100, 10};

  NeuralNetwork nn(num_inputs, num_outputs, num_hidden_layers, num_nodes_per_hidden_layer);
 
  for (int i = 0; i < 30000; i++) {
    double* output = nn.feedforward(train_data[i]);

    print_element("\nOutput: ", str_width);
    for (int j = 0; j < 10; j++)
      print_element(output[j], num_width);
    print_element("\nTarget: ", str_width);
    for (int j = 0; j < 10; j++)
      print_element(train_targets[i][j], num_width);
    cout << "\n================================================================" << endl;

    nn.backpropagate(train_targets[i], 0.2);
  } 

  // Clean up allocated memory
  for (int i = 0; i < train_size; i++){
    delete [] train_data[i];
    delete [] train_targets[i];
  }
  delete [] train_data;
  delete [] train_targets;

  for (int i = 0; i < test_size; i++){
    delete [] test_data[i];
    delete [] test_targets[i];
  }
  delete [] test_data;
  delete [] test_targets;

  return 0;
}