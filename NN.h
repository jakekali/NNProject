//
// Created by Jacob on 11/18/2021.
//
#include <vector>
#include "json.hpp"
using json = nlohmann::json;

#ifndef NN_NN_H
#define NN_NN_H


class NN {
public:
    NN(std::vector<int> layersSizesIn);
    int loadWeightsToNode (int layer, int index, std::vector<double> weightsPrevious);
    json exportNetwork(const std::string& filename, bool sable);
    int loadWeightsFromFile(const std::string& filename);
    std::vector<double> eval(std::vector<double> input);
    int deltas(std::vector<double> input);
    int updateWeights(double lr);
    int cleanUpBackProb();
    int train(std::string filename, int epoch, double lr);
    std::vector<double> runningAcc;
    std::vector<double> runningLoss;
private:
    class node{
    private:
        double currValue;
    public:
        void reset();
        int index;
        int layer;
        int setCurrentValue(double newValue);
        std::vector<double> weightsPrevious;
        bool fixed = false;
        double getCurrVal() const{return currValue;};
        double delta;
        double weightedSum;
    };
    std::vector<int> layersSizes;
    std::vector<std::vector<node>> neuralNetwork;
};


#endif //NN_NN_H
