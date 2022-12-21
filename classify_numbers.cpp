#include "neural_network.cpp"
#include <iostream>
#include <istream>
#include <vector>
#include <sstream>
using namespace std;

// Display parameters
const int str_width = 10;
const int num_width = 10;

void display_data(double* output, double* target);

void shuffle_data(double **&data, double **&targets, int size);

template<typename T> void print_element(T t, const int& width){
  cout << left << setw(width) << setfill(' ') << t;
}

int main(){

  // Display parameters
  std::cout << std::fixed;
  std::cout << std::setprecision(2);

  // Dataset parameters
  int data_size = 28*28;
  int target_size = 10;
  int max_pixel_value = 255;

  // dataset files
  string train_file = "mnist_train.csv";
  string test_file = "mnist_test.csv";

  // Initialize arrays to hold training data
  cout<< "Allocating memory for trainig data..."<< std::endl;
  int train_size = 60000;
  double** train_data = new double*[train_size];
  double** train_targets = new double*[train_size];
  for (int i = 0; i < train_size; i++){
    train_data[i] = new double[data_size];
    train_targets[i] = new double[target_size];
  }
  
  // Initialize arrays to hold testing data
  cout<< "Allocating memory for testing data..."<< std::endl;
  int test_size = 10000;
  double** test_data = new double*[test_size];
  double** test_targets = new double*[test_size];
  for (int i = 0; i < test_size; i++){
    test_data[i] = new double[data_size];
    test_targets[i] = new double[target_size];
  }

  // Load the training data
  cout << "Loading training data..." << endl;
  std::ifstream train_file_stream(train_file);
  if (train_file_stream.is_open()){
    std::string line, value;
    int i = 0, j = 0;
    while (std::getline(train_file_stream, line))
    {
        j = 0;
        stringstream str(line);
        while (std::getline(str, value, ',')){
          if (j == 0){
            train_targets[i][stoi(value)] = 1.0;
          } else {
            train_data[i][j - 1] = stod(value) / max_pixel_value;
          }
          j++;
        }
        i++;
    }
    train_file_stream.close();
  } else {
    cerr << "Error, could not open file: " << train_file << endl;
    exit(1);
  }

  // Training parameters
  int epochs = 100;
  int batch_size = 100;

  // Neural network parameters
  double learning_rate = 0.001;
  int num_inputs = data_size;
  int num_outputs = target_size;
  int num_hidden_layers = 3;
  int num_nodes_per_hidden_layer[num_hidden_layers] = {600, 600, 600};

  cout << "Allocating memory for neural network..."<<endl;
  NeuralNetwork nn(num_inputs, num_outputs, num_hidden_layers, num_nodes_per_hidden_layer);

  cout << "Starting training..."<<endl;
  // Train the neural network
  for (int i = 0; i < epochs; i++){
    shuffle_data(train_data, train_targets, batch_size);
    double *output, sum_error = 0, correct = 0;
    int highest = 0;
    for (int j = 0; j < batch_size; j++) {
      output = nn.feedforward(train_data[j]);
      for (int k = 0; k < 10; k++) {
        sum_error += pow((train_targets[j][k] - output[k]), 2);
        if (output[k] > output[highest]) highest = k;
      }
      correct += (train_targets[j][highest] == 1) ? 1 : 0;
      nn.backpropagate(train_targets[j], learning_rate);
    } 
    //nn.backpropagate(train_targets, learning_rate, batch_size);
    std::cout << "Epoch="<<i<<" Error="<<sum_error/batch_size<<" Score="<<correct/batch_size<<std::endl;
  }

  
  for (int i = 0; i < 20; i++)
    display_data(nn.feedforward(train_data[i]), train_targets[i]);
  
  // Test the neural network
  //for (int i = 0; i < test_size; i++) {
  //  nn.feedforward(test_data[i]);
  // }
  
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

void display_data(double* output, double* target){
  std::cout << std::fixed;
  std::cout << std::setprecision(2);
  print_element("\nOutput: ", str_width);
  for (int i = 0; i < 10; i++)
    print_element(output[i], num_width);
  print_element("\nTarget:", str_width);
  for (int i = 0; i < 10; i++)
    print_element(target[i], num_width);
  std::cout << std::endl;
}

// Struct to hold both train_data and train_targets
struct data_target_pair {
  double* train_data;
  double* train_targets;
};

void shuffle_data(double **&data, double **&targets, int size){
    // Create a vector of data_target_pairs
  std::vector<data_target_pair> data_target_pairs;
  for (int i = 0; i < size; i++) {
    data_target_pair pair;
    pair.train_data = data[i];
    pair.train_targets = targets[i];
    data_target_pairs.push_back(pair);
  }
   std::mt19937 rng(std::random_device{}());

  // Shuffle the data_target_pairs vector
  std::shuffle(data_target_pairs.begin(), data_target_pairs.end(), rng);

  // Copy the shuffled data back into train_data and train_targets
  for (int i = 0; i < size; i++) {
    data[i] = data_target_pairs[i].train_data;
    targets[i] = data_target_pairs[i].train_targets;
  }
}