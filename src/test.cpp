#include <iostream>
#include <neural.hpp>
#include <string>


int main(int argc, char ** argv) {
    Neural::NeuralNetwork neuralNet;
    std::string trainedNN, testingSet, output;

    std::cout << "Please specify a trained ANN file: ";
    std::cin >> trainedNN;
    std::cout << "Please specify a testing set file: ";
    std::cin >> testingSet;
    std::cout << "Please specify a metric output file: ";
    std::cin >> output;

    neuralNet.loadFromFile(trainedNN);                     // load in the trained network parameters
    neuralNet.testUsingFile_andOutput(testingSet, output); // test the dataset using the network

    return 0;
}
