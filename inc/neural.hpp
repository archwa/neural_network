#ifndef __NEURAL_HPP__
#define __NEURAL_HPP__

#include <cmath>
#include <string>
#include <vector>

// not sure what to include quite yet...

namespace Neural {

    double sigmoid(double x);
    double sigmoidPrime(double x);

    struct Node;
    
    typedef struct {
        std::vector<double> input;
        std::vector<double> output;
    } DataPoint;

    typedef struct {
        unsigned int a;
        unsigned int b;
        unsigned int c;
        unsigned int d;

        double overallAccuracy;
        double precision;
        double recall;
        double f1;
    } MetricPoint;

    typedef struct {
        double weight;

        struct Node * from;
        struct Node * to;

        unsigned int fco[2];
        unsigned int tco[2];
    } Link;

    typedef struct Node {
        double in;    // input sum (double)
        double activation; // obtained from sigmoid of in (activation)
        double delta; // obtained by back-prop

        bool isBias;

        std::vector<Link *> inLinks;
        std::vector<Link *> outLinks;
    } Node;

    class NeuralNetwork {
        public:
            NeuralNetwork();
            NeuralNetwork(std::vector<std::vector<std::vector<double>>>);
            ~NeuralNetwork();
            // list of layers
            // => each layer has a list of nodes
            // => each node has a list of double values for link weights
            
            void configureUsingWeights(std::vector<std::vector<std::vector<double>>>);

            void train(double, unsigned int, std::vector<DataPoint>); // 0:input ; 1:output
            void trainUsingFile_andLearningRate_andEpochs(std::string const&, double, unsigned int);

            void testUsingFile_andOutput(std::string const&, std::string const&);

            void loadFromFile(std::string const&);
            void saveToFile(std::string const&);
        private:
            std::vector<std::vector<Node *>> nodes;
            std::vector<Link *> linkList;
            void backPropagate();
    };

    // the first weight is the bias weight which is always attached to a node outputting -1
    
}

#endif
