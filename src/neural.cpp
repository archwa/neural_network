#include <cstdlib>
#include <fstream>
#include <neural.hpp>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// implement the neural network functionality here
//  (e.g. back-propagation algorithm)


// for updating weights
double Neural::sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double Neural::sigmoidPrime(double x) {
    return sigmoid(x) * (1.0 - sigmoid(x));
}

Neural::NeuralNetwork::NeuralNetwork() {
    // idk... don't do anything?  w/e
}

Neural::NeuralNetwork::NeuralNetwork(std::vector<std::vector<std::vector<double>>> weights) {
    this->configureUsingWeights(weights);
}

// initialize the neural network with the provided weights
void Neural::NeuralNetwork::configureUsingWeights(std::vector<std::vector<std::vector<double>>> weights) {
    // initializing the neural network with the weights
    unsigned int i, j, k;
    Neural::Node * node;
    Neural::Link * link;
    std::vector<Node *> layerNodes;

    // index 0 at a layer always has activation at a constant -1 (except at the output layer)

    // for each node in the input layer...
    for(i = 0; i < weights[0].size(); i++) {

        // initialize node parameters to null
        node = new Neural::Node;

        // handle the bias weight/node
        if(i) {
            node->activation = 0.0;
            node->isBias = false;
        } else {
            node->activation = -1.0;
            node->isBias = true;
        }

        node->inLinks.push_back(nullptr); // there are no in links for inputs
        
        // push the node into the node list of the input layer
        layerNodes.push_back(node);
    }
    // push the input layer
    this->nodes.push_back(layerNodes);
    layerNodes.clear();

    
    // for each hidden layer...initialize all the nodes and corresponding links
    for(i = 1; i < weights.size() - 1; i++) {

        // for each node in the hidden layer...
        for(j = 0; j < weights[i].size(); j++) {

            // create a new node
            node = new Neural::Node;

            // handle the bias weight/node
            if(j) {
                node->activation = 0.0;
                node->isBias = false;

                // for each weight coming into a node...
                for(k = 0; k < weights[i][j].size(); k++) {
                    // question: is the link pointing back to the correct node?

                    // create a new link
                    link = new Neural::Link;
                    link->weight = weights[i][j][k];    // store the actual weight in the link
                    link->from = this->nodes[i - 1][k]; // link "from" is previous node
                    link->to = node;                    // link "to" is current node
                    link->fco[0] = i - 1;
                    link->fco[1] = k;
                    link->tco[0] = i;
                    link->tco[1] = j;

                    this->linkList.push_back(link);

                    // push the link into the link lists (OUT and IN)
                    this->nodes[i - 1][k]->outLinks.push_back(link); // add link to previous node OUT link list
                    node->inLinks.push_back(link);                   // add link to current node IN link list
                }
            } else {
                node->activation = -1.0;
                node->isBias = true;
                node->inLinks.push_back(nullptr); // there are no in links for any bias node
            }

            // push the node into the node list of the corresponding hidden layer
            layerNodes.push_back(node);
        }
        // push the given hidden layer
        this->nodes.push_back(layerNodes);
        layerNodes.clear();
    }


    // for each node in the output layer...
    for(i = 0; i < weights[weights.size() - 1].size(); i++) {

        // create a new node
        node = new Neural::Node;
        node->activation = 0.0;
        node->isBias = false;

        // for each weight coming into a node...
        for(j = 0; j < weights[weights.size() - 1][i].size(); j++) {
            
            // create a new link
            link = new Neural::Link;
            link->weight = weights[weights.size() - 1][i][j]; // store the actual weight in the link
            link->from = this->nodes[weights.size() - 2][j];  // link "from" is previous node
            link->to = node;                                  // link "to" is current node
            link->fco[0] = weights.size() - 2;
            link->fco[1] = j;
            link->tco[0] = weights.size() - 1;
            link->tco[1] = i;
            
            this->linkList.push_back(link);

            // push the link into the link lists (OUT and IN)
            this->nodes[weights.size() - 2][j]->outLinks.push_back(link); // add link to previous node OUT link list
            node->inLinks.push_back(link);                                // add link to current node IN link list
        }

        // push the node into the node list of the output layer
        layerNodes.push_back(node);
    }
    // push the list of output layer nodes to the list of layers
    this->nodes.push_back(layerNodes);
    layerNodes.clear();

    /*  for debugging whether the weights were correctly generated / associated
    std::cout << this->linkList.size() << std::endl;
    for(i = 0; i < this->linkList.size(); i++) {
        std::cout << "From: " << this->linkList[i]->fco[0] << "," << this->linkList[i]->fco[1] << " ~ ";
        std::cout << "To: " << this->linkList[i]->tco[0] << "," << this->linkList[i]->tco[1] << " ~ ";
        std::cout << "Weight: " << this->linkList[i]->weight << std::endl;
    }
    */
}


Neural::NeuralNetwork::~NeuralNetwork() {
    // free up all the memory :)
}


// train the neural network using a training set matching the dimensions of the neural network
void Neural::NeuralNetwork::train(double alpha, unsigned int epochs, std::vector<Neural::DataPoint> trainingSet) {
    // implement back propagation here
    unsigned int i, j, k, l, epox;
    double sum;
    Node * node;
    Link * link;
    
    for(epox = 0; epox < epochs; epox++) {
        for(i = 0; i < trainingSet.size(); i++) {

            // copy input vector values into the input nodes' activations
            for(j = 1; j < this->nodes[0].size(); j++) {
                this->nodes[0][j]->activation = trainingSet[i].input[j - 1];
            }
            
            // for every hidden/output layer, propagate forward
            for(j = 1; j < this->nodes.size(); j++) {

                // for every node in the hidden/output layer...
                for(k = 0; k < this->nodes[j].size(); k++) {

                    // let us calculate the activation for each node that is not a bias node
                    if(this->nodes[j][k]->isBias) {
                        this->nodes[j][k]->in = 0.0; // only set to 0 if it's a bias node
                    } else { // otherwise, compute the activation for each non-bias node
                        sum = 0.0;

                        // for every input link, multiply the weight by the activation of the node the current node is linked to (backward)
                        for(l = 0; l < this->nodes[j][k]->inLinks.size(); l++) {
                            sum += this->nodes[j][k]->inLinks[l]->weight * this->nodes[j][k]->inLinks[l]->from->activation;
                        }

                        this->nodes[j][k]->in = sum;
                        this->nodes[j][k]->activation = Neural::sigmoid(sum);
                    }

                }
            }

            // for every output node, calculate the delta
            for(j = 0; j < this->nodes[this->nodes.size() - 1].size(); j++) {
                node = this->nodes[this->nodes.size() - 1][j]; // this gives us an output node

                // calculate the output node's delta
                node->delta = sigmoidPrime(node->in) * (trainingSet[i].output[j] - node->activation);
            }

            // for every hidden layer, prop the delts brah
            for(j = this->nodes.size() - 2; j > 0; j--) { // begin at the last hidden layer, end at the first hidden layer
                for(k = 0; k < this->nodes[j].size(); k++) { // for each node in a given hidden layer
                    node = this->nodes[j][k];

                    sum = 0.0;
                    for(l = 0; l < node->outLinks.size(); l++) {
                        sum += node->outLinks[l]->weight * node->outLinks[l]->to->delta;
                    }
                    node->delta = sigmoidPrime(node->in) * sum;
                }
            }

            // update all the weights for each link
            for(j = 0; j < this->linkList.size(); j++) { // for each link, update the weight
                link = this->linkList[j]; // get the link
                link->weight = link->weight + (alpha * link->from->activation * link->to->delta); // update the weight
            }
        }
    }
}

void Neural::NeuralNetwork::loadFromFile(std::string const& filePath) {
    unsigned int i, j;
    std::string line, token;
    std::vector<unsigned int> nodeCounts;

    // read in from the file
    std::ifstream inputFile(filePath);


    if(!inputFile.is_open()) {
        std::cout << "Warning: Could not load from file!" << std::endl;
        std::cout << "         Failed to open '" << filePath << "' for reading." << std::endl;
    } else {

        std::getline(inputFile, line);
        std::istringstream iss(line);
        
        // get layer + node information
        while(iss >> token) { 
            nodeCounts.push_back(static_cast<unsigned int>(std::strtoll(token.c_str(), nullptr, 10)));
        }

        std::vector<double> weightList;
        std::vector<std::vector<double>> nodeList;
        std::vector<std::vector<std::vector<double>>> layerList;

        // bias node at input layer
        weightList.push_back(1);
        nodeList.push_back(weightList);
        weightList.clear();

        // for each node in the input layer...
        for(i = 0; i < nodeCounts[0]; i++) {
            weightList.push_back(1);
            nodeList.push_back(weightList);
            weightList.clear();
        }
        layerList.push_back(nodeList);
        nodeList.clear();


        // for each hidden/output layer...
        for(i = 1; i < nodeCounts.size(); i++) {
            // bias node at a hidden layer
            if(i < nodeCounts.size() - 1) {
                weightList.push_back(1);
                nodeList.push_back(weightList);
                weightList.clear();
            }

            // for each node in the hidden/output layer...
            for(j = 0; j < nodeCounts[i]; j++) {
                // get the line and interpret each token as a weight to be added to the input weight node stuff
                std::getline(inputFile, line);
                iss = std::istringstream(line);
                while(iss >> token) { 
                    weightList.push_back(std::strtod(token.c_str(), nullptr));
                }

                // add the weight list to the node list
                nodeList.push_back(weightList);
                weightList.clear();
            }

            // add the node list to the layer list
            layerList.push_back(nodeList);
            nodeList.clear();
        }
        
        this->configureUsingWeights(layerList);
    }
}

void Neural::NeuralNetwork::saveToFile(std::string const& filePath) {
    unsigned int i, j, k;
    std::ofstream outputFile(filePath);
    if(!outputFile.is_open()) {
        //uh oh...
        std::cout << "Can't save :(" << std::endl;
    } else {
        // write the # of nodes at each layer to file
        for(i = 0; i < this->nodes.size(); i++) {
            outputFile << ((i == this->nodes.size() - 1) ? this->nodes[i].size() : this->nodes[i].size() - 1);
            if(i != this->nodes.size() - 1) {
                outputFile << " ";
            }
        }
        outputFile << std::endl;

        // write the weights for each node 
        for(i = 1; i < this->nodes.size(); i++) {
            for(j = 0; j < this->nodes[i].size(); j++) {
                if(!this->nodes[i][j]->isBias) {
                    for(k = 0; k < this->nodes[i][j]->inLinks.size(); k++) {
                        outputFile << std::setprecision(3) << std::fixed << this->nodes[i][j]->inLinks[k]->weight;
                        if(k != this->nodes[i][j]->inLinks.size() - 1) {
                            outputFile << " ";
                        }
                    }
                    outputFile << std::endl;
                }
            }
        }
    }
}

void Neural::NeuralNetwork::trainUsingFile_andLearningRate_andEpochs(std::string const& filePath, double alpha, unsigned int epochs) {
    std::string token;
    std::ifstream inputFile(filePath);
    double el;
    unsigned long long dataPointCount, inputDim, outputDim;
    std::vector<Neural::DataPoint> trainingSet;
    unsigned int i, j;
    Neural::DataPoint trainPoint;

    if(!inputFile.is_open()) {
        std::cout << "Warning: Could not train using file!" << std::endl;
        std::cout << "         Failed to open '" << filePath << "' for reading." << std::endl;
    } else {
        inputFile >> dataPointCount >> inputDim >> outputDim;

        for(i = 0; i < dataPointCount; i++) {
            for(j = 0; j < inputDim; j++) {
                inputFile >> el;
                trainPoint.input.push_back(el);
            }
            for(j = 0; j < outputDim; j++) {
                inputFile >> el;
                trainPoint.output.push_back(el);
            }
                
            trainingSet.push_back(trainPoint); // push the datapoint into the set
            trainPoint.input.clear();
            trainPoint.output.clear();
        }

        this->train(alpha, epochs, trainingSet);
    }
}

void Neural::NeuralNetwork::testUsingFile_andOutput(std::string const& dataFilePath, std::string const& outputFilePath) {
    std::ifstream inputFile(dataFilePath);
    std::ofstream outputFile(outputFilePath);
    double el, a, b, c, d, acc, pre, re, sum;
    unsigned long long dataPointCount, inputDim, outputDim;
    unsigned int i, j, k, l, outputClass;
    std::vector<Neural::DataPoint> testingSet;
    Neural::DataPoint testPoint;

    std::vector<Neural::MetricPoint> testingMetrics;
    Neural::MetricPoint metricPoint;

    if(!inputFile.is_open()) {
        std::cout << "Warning: Could not test using file!" << std::endl;
        std::cout << "         Failed to open '" << dataFilePath << "' for reading." << std::endl;
    } else if(!outputFile.is_open()) {
        std::cout << "Warning: Could not test using file!" << std::endl;
        std::cout << "         Failed to open '" << outputFilePath << "' for writing." << std::endl;
    } else {
        inputFile >> dataPointCount >> inputDim >> outputDim;

        for(i = 0; i < dataPointCount; i++) {
            for(j = 0; j < inputDim; j++) {
                inputFile >> el;
                testPoint.input.push_back(el);
            }
            for(j = 0; j < outputDim; j++) {
                inputFile >> el;
                testPoint.output.push_back(el);
            }
                
            testingSet.push_back(testPoint); // push the datapoint into the set
            testPoint.input.clear();
            testPoint.output.clear();
        }

        inputFile.close();

        metricPoint.a = metricPoint.b = metricPoint.c = metricPoint.d = 0;
        for(i = 0; i < this->nodes[this->nodes.size() - 1].size(); i++) {
           testingMetrics.push_back(metricPoint);
        }
        
        for(i = 0; i < testingSet.size(); i++) {
            // copy input vector values into the input nodes' activations
            for(j = 1; j < this->nodes[0].size(); j++) {
                this->nodes[0][j]->activation = testingSet[i].input[j - 1];
            }

            // for every hidden/output layer, propagate forward
            for(j = 1; j < this->nodes.size(); j++) {

                // for every node in the hidden/output layer...
                for(k = 0; k < this->nodes[j].size(); k++) {

                    // let us calculate the activation for each node that is not a bias node
                    if(this->nodes[j][k]->isBias) {
                        this->nodes[j][k]->in = 0.0; // only set to 0 if it's a bias node
                    } else { // otherwise, compute the activation for each non-bias node
                        sum = 0.0;

                        // for every input link, multiply the weight by the activation of the node the current node is linked to (backward)
                        for(l = 0; l < this->nodes[j][k]->inLinks.size(); l++) {
                            sum += this->nodes[j][k]->inLinks[l]->weight * this->nodes[j][k]->inLinks[l]->from->activation;
                        }

                        this->nodes[j][k]->in = sum;
                        this->nodes[j][k]->activation = Neural::sigmoid(sum);
                    }
                }
            }

            // calculate the metrics for each boolean class at the output, remembering to round
            for(j = 0; j < testingMetrics.size(); j++) {
                outputClass = round(this->nodes[this->nodes.size() - 1][j]->activation);

                if(outputClass == 1 && testingSet[i].output[j] == 1) {
                    testingMetrics[j].a++;
                }
                if(outputClass == 1 && testingSet[i].output[j] == 0) {
                    testingMetrics[j].b++;
                }
                if(outputClass == 0 && testingSet[i].output[j] == 1) {
                    testingMetrics[j].c++;
                }
                if(outputClass == 0 && testingSet[i].output[j] == 0) {
                    testingMetrics[j].d++;
                }
            }
        }

        outputFile << std::setprecision(3) << std::fixed;

        for(i = 0; i < testingMetrics.size(); i++) {
            a = testingMetrics[i].a;
            b = testingMetrics[i].b;
            c = testingMetrics[i].c;
            d = testingMetrics[i].d;

            testingMetrics[i].overallAccuracy = (a + d) / (a + b + c + d);
            testingMetrics[i].precision = a / (a + b);
            testingMetrics[i].recall = a / (a + c);
            testingMetrics[i].f1 = (2 * testingMetrics[i].precision * testingMetrics[i].recall) / (testingMetrics[i].precision + testingMetrics[i].recall);
            
            outputFile << static_cast<unsigned int>(a) << " ";
            outputFile << static_cast<unsigned int>(b) << " ";
            outputFile << static_cast<unsigned int>(c) << " ";
            outputFile << static_cast<unsigned int>(d) << " ";
            outputFile << testingMetrics[i].overallAccuracy << " ";
            outputFile << testingMetrics[i].precision << " ";
            outputFile << testingMetrics[i].recall << " ";
            outputFile << testingMetrics[i].f1 << std::endl;
        }

        a = b = c = d = acc = pre = re = 0.0;

        for(i = 0; i < testingMetrics.size(); i++) {
            a += testingMetrics[i].a;
            b += testingMetrics[i].b;
            c += testingMetrics[i].c;
            d += testingMetrics[i].d;
        }
        
        outputFile << (a + d) / (a + b + c + d) << " ";
        outputFile << a / (a + b) << " ";
        outputFile << a / (a + c) << " ";
        outputFile << ((2 * (a / (a + b)) * (a / (a + c))) / ((a / (a + b)) + (a / (a + c)))) << std::endl;

        for(i = 0; i < testingMetrics.size(); i++) {
            acc += testingMetrics[i].overallAccuracy;
            pre += testingMetrics[i].precision;
            re += testingMetrics[i].recall;
        }
        
        acc /= testingMetrics.size();
        pre /= testingMetrics.size();
        re /= testingMetrics.size();

        outputFile << acc << " ";
        outputFile << pre << " ";
        outputFile << re << " ";
        outputFile << ((2 * pre * re) / (pre + re)) << std::endl;
    }

    outputFile.close();
}
