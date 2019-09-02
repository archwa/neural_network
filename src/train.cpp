#include <iostream>
#include <neural.hpp>
#include <string>


int main(int argc, char ** argv) {
    Neural::NeuralNetwork neuralNet;
    std::string untrainedNN, trainingSet, output;
    double alpha;
    unsigned int epochs;

    std::cout << "Please specify an untrained ANN file: ";
    std::cin >> untrainedNN;
    std::cout << "Please specify a training set file: ";
    std::cin >> trainingSet;
    std::cout << "Please specify a trained ANN output file: ";
    std::cin >> output;
    std::cout << "Please specify a learning rate (alpha): ";
    std::cin >> alpha;
    std::cout << "Please specify # of iterations (epochs): ";
    std::cin >> epochs;

    neuralNet.loadFromFile(untrainedNN);                                            // load in the untrained network parameters
    neuralNet.trainUsingFile_andLearningRate_andEpochs(trainingSet, alpha, epochs); // train using training set file
    neuralNet.saveToFile(output);                                                   // save the trained network weights

    return 0;
}
