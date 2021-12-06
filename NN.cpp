//
// Created by Jacob on 11/18/2021.
//

#include "NN.h"
#include <vector>
#include <iomanip>
#include <fstream>
#include <cctype>
#include <string>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <ctime>    
// #include "tensorboard_logger.h"

using json = nlohmann::json;

/*!
    @brief construct a new NN

    Returns a new NN, whose layers are of the sizes specified in the array.
    Inits Weights & Biases Randomly, First Node in everylayer (at index 0 is a fixed -1 input)
    Weights are stored going into any node

    @param std::vector<int> layersSizes {SizeOfInputLayer ...,Size of Hidden Layers, ... Size of Output Layer}

    @return a new NN, whose layers are of the sizes specified in the array.

    */
NN::NN(std::vector<int> layersSizesIn) {
    neuralNetwork.resize(layersSizesIn.size());
    for(int i = 0; i < layersSizesIn.size(); i++){
        layersSizes.push_back(layersSizesIn[i]);
        neuralNetwork[i].resize(layersSizesIn[i]+1);
        neuralNetwork[i][0].setCurrentValue(-1);
        neuralNetwork[i][0].fixed = true;
        neuralNetwork[i][0].weightsPrevious.resize(0);
        neuralNetwork[i][0].layer = i;
        neuralNetwork[i][0].index = 0;
        neuralNetwork[i][0].weightedSum = -1;
        for(int j = 1; j < neuralNetwork[i].size();j++){
            neuralNetwork[i][j].layer = i;
            neuralNetwork[i][j].index = j;
            neuralNetwork[i][j].fixed = false;
            neuralNetwork[i][j].setCurrentValue(0);
            if(i == 0){
                neuralNetwork[i][j].weightsPrevious.resize(0);
            }else{
                neuralNetwork[i][j].weightsPrevious.resize(neuralNetwork[i-1].size());
                for(double & weightsPreviou : neuralNetwork[i][j].weightsPrevious){
                    //random weights between -1 and 1
                    double f = (double)rand() / RAND_MAX;
                    weightsPreviou = (-1) + f * (2);
                }
            }
        }
    }
}
/*!
    @brief updated all the incoming weights for a given node

    Returns -1 if the node does not exist else 0;

    @param int layer, int index, std::vector<double> weightsPrevious

    @return -1 if the node does not exist else 0;

    */
int NN::loadWeightsToNode(int layer, int index, std::vector<double> weightsPrevious) {
    if(neuralNetwork[layer-1].size() != weightsPrevious.size()){
        return -1;
    }
    for(int i = 0; i < neuralNetwork[layer][index].weightsPrevious.size(); i++){
        neuralNetwork[layer][index].weightsPrevious[i] = weightsPrevious[i];
    }
    return 0;
}

/*!
    @brief exports the NN into json or text file

    Returns if sable == true -> outputs a file with the fiven file name
            if sable != true -> returns a json object and writes file

    @param std::string filename, bool sable

    @return -if sable == true -> outputs a file with the fiven file name
            if sable != true -> returns a json object

    */

json NN::exportNetwork(const std::string& filename, bool sable) {
    //Sable - Standard Assigment format (.txt)
    if(sable){
        std::ofstream o(filename);
        int i = 0;
        for (int num : layersSizes){
            if(i == layersSizes.size() -1){
                o << num;
            }else {
                o << num << ' ';
            }
            i++;
        }
        o << '\n';

        for(int j = 1; j < layersSizes.size(); j ++){
            //this counter starts from one because the first fixed input node at index zero does not have any incoming weights
            for(int n = 1; n < neuralNetwork[j].size(); n++){
                int w = 0;
                for (double & weight : neuralNetwork[j][n].weightsPrevious){
                    if(w == neuralNetwork[j][n].weightsPrevious.size() - 1){
                        o << std::setprecision(3) << std::fixed << weight;

                    }else{
                        o << std::setprecision(3) << std::fixed << weight << ' ';
                    }
                    w++;
                }
                o << '\n';
            }
        }
        o.close();
        return 0;
    }else{
    //!Sable - Json Format (.json)(!sable){
        json exportJ;
        exportJ["numLayers"] = layersSizes.size();
        exportJ["layersSize"] = layersSizes;
        for(int layerID = 0; layerID < layersSizes.size(); layerID++){
            for(int nodeID = 0; nodeID < neuralNetwork[layerID].size(); nodeID++){
                exportJ[std::to_string(layerID)][std::to_string(nodeID)]["layer"] = neuralNetwork[layerID][nodeID].layer;
                exportJ[std::to_string(layerID)][std::to_string(nodeID)]["index"] = neuralNetwork[layerID][nodeID].index;
                exportJ[std::to_string(layerID)][std::to_string(nodeID)]["fixed"] = neuralNetwork[layerID][nodeID].fixed;
                exportJ[std::to_string(layerID)][std::to_string(nodeID)]["currValue"] = neuralNetwork[layerID][nodeID].getCurrVal();
                for(int k = 0; k < neuralNetwork[layerID][nodeID].weightsPrevious.size(); k++) {
                    exportJ[std::to_string(layerID)][std::to_string(nodeID)]["weightsPrevious"][k] = neuralNetwork[layerID][nodeID].weightsPrevious[k];
                }
            }
        }

        std::ofstream o(filename+".json");
        o << std::setw(4) << exportJ << std::endl;
        return exportJ;
    }
}
// true if the argument is whitespace, false otherwise
bool space(char c)
{
    return isspace(c);
}

// false if the argument is whitespace, true otherwise
bool not_space(char c)
{
    return !isspace(c);
}
std::vector<double> split(const std::string& str)
{
    typedef std::string::const_iterator iter;
    std::vector<double> ret;
    iter i = str.begin();
    while (i != str.end())
    {
        // ignore leading blanks
        i = find_if(i, str.end(), not_space);
        // find end of next word
        iter j = find_if(i, str.end(), space);
        // copy the characters in [i, j)
        if (i != str.end())
            ret.push_back(std::stod(std::string(i, j)));
        i = j;
    }
    return ret;
}
int NN::loadWeightsFromFile(const std::string& filename) {
    std::ifstream in(filename);
    std::string currLine;
    int layer = 1;
    int index = 1;
    //just to skip first line
    std::getline(in, currLine);
    while(std::getline(in, currLine)){
        if(this->loadWeightsToNode(layer,index,split(currLine)) == -1){
            std::cerr << "Bruh \n";
        }
        index++;
        if(index == neuralNetwork[layer].size()){
            index = 1;
            layer++;
        }
    }

    return 0;
}

double sigmoid(double x) {
    double result;
    result = 1 / (1 + exp(-x));
    return result;
}

double sigmoidPrime(double x) {
    double result;
    result = sigmoid(x) * (1 - sigmoid(x));
    return result;
}

std::vector<double> NN::eval(std::vector<double> input) {
    if(input.size() != (neuralNetwork[0].size()-1)){
        std::cerr << "input size mismatch input size: " << input.size() << "nn size: " << neuralNetwork[0].size() << '\n';
    }
    //Copies the input vector of a single training example to the input nodes of the NN
    //Correct here 0 was putting shit in wrong places, skipping first input
    for (int i = 1; i < neuralNetwork[0].size(); i++){
        neuralNetwork[0][i].setCurrentValue(input[i-1]);
    }

    for(int l = 1; l < neuralNetwork.size(); l++){
        for(int j = 1; j < neuralNetwork[l].size(); j++){
            double newVal = 0;
            if(neuralNetwork[l][j].weightsPrevious.size() != neuralNetwork[l-1].size()){
                std::cerr << "major size mismatch \n";
            }
            for(int w = 0; w < neuralNetwork[l][j].weightsPrevious.size();w++){
                newVal += ((1.0) * neuralNetwork[l][j].weightsPrevious[w] * neuralNetwork[l-1][w].getCurrVal());
            }
            neuralNetwork[l][j].weightedSum = newVal;
            neuralNetwork[l][j].setCurrentValue(sigmoid(newVal));
        }
    }
    std::vector<double> ret;
    ret.resize(neuralNetwork[neuralNetwork.size()-1].size()-1);
    for(int i = 1; i < neuralNetwork[neuralNetwork.size()-1].size(); i++){
        ret[i-1] = neuralNetwork[neuralNetwork.size()-1][i].getCurrVal();
    }
    return ret;
}
/*!
 d   @brief caclulates deltas as part of the backprogration learning
            changes deltas within the nodes class.

    Returns -1 if the output vector size does not match the input.
            0 on success.

    @param std::vector<double> correctValues

    @return  -1 if the output vector size does not match the input.
            0 on success.

    */
int NN::deltas(std::vector<double> correctValues) {
    //check that the correctValues and outputs are the same size
    if(correctValues.size() != neuralNetwork[neuralNetwork.size()-1].size() -1){
        std::cerr << "Bruh, input mismatch with correct values. correctValues.size() = " << correctValues.size() << "but, neuralNetwork[neuralNetwork.size()-1].size() -1 = " << neuralNetwork[neuralNetwork.size()-1].size() -1 << '\n';
        return -1;
    }

    // double loss;
    // int max = INT_MIN;
    // int max_index = -1;
    // //tensorboard output
    // for(int i = 1; i < neuralNetwork[neuralNetwork.size()-1].size(); i++){
    //     loss += neuralNetwork[neuralNetwork.size()-1][i].getCurrVal() - correctValues[i-1];
    //     if(max > neuralNetwork[neuralNetwork.size()-1][i].getCurrVal()){
    //         max = neuralNetwork[neuralNetwork.size()-1][i].getCurrVal();
    //         max_index = i;
    //     }
    // }
    // runningLoss.push_back(loss);
    // if(correctValues[max_index-1] == 1){
    //     runningAcc.push_back(1);
    // }else{
    //     runningAcc.push_back(0);
    // }



    //for each node in the last row -- calculate deltas
    for(int index = 1; index < neuralNetwork[neuralNetwork.size()-1].size(); index++){
        neuralNetwork[neuralNetwork.size()-1][index].delta = sigmoidPrime(neuralNetwork[neuralNetwork.size()-1][index].weightedSum) * (correctValues[index -1] - neuralNetwork[neuralNetwork.size()-1][index].getCurrVal());
    }

    //for the rest of the nodes
    //loop through the remaining layers in reverse order
    for(int layer = neuralNetwork.size() - 2; layer > 0; layer--){
        //for each node:
        for(int node = 1; node < neuralNetwork[layer].size(); node++){
            double sum = 0.0;
            for(int nodeAfter = 1; nodeAfter < neuralNetwork[layer+1].size(); nodeAfter++){
                sum += neuralNetwork[layer+1][nodeAfter].weightsPrevious[node] * neuralNetwork[layer+1][nodeAfter].delta;
            }
            // calculate the delta for each of the
            neuralNetwork[layer][node].delta = sigmoidPrime(neuralNetwork[layer][node].weightedSum) * sum ;
        }
    }
    return 0;
}

int NN::updateWeights(double lr) {
    //for each node in the network:
    //for each later
    int layer = neuralNetwork.size();
    for(layer = layer -1; layer > 0; layer--){
        for (int nodeIn = 0; nodeIn < neuralNetwork[layer].size(); nodeIn++){
            for(int weight = 0; weight < neuralNetwork[layer][nodeIn].weightsPrevious.size();weight++){
                double change = (lr * neuralNetwork[layer-1][weight].getCurrVal() * neuralNetwork[layer][nodeIn].delta);
                double og = neuralNetwork[layer][nodeIn].weightsPrevious[weight];
                neuralNetwork[layer][nodeIn].weightsPrevious[weight] = change + og;
                //std::cerr << neuralNetwork[layer][nodeIn].weightsPrevious[weight] << "\n";
            }
        }
    }
    return 0;
}

int NN::cleanUpBackProb() {
    //for each node in the network:
    //for each later
    int layer = neuralNetwork.size();
    for(layer = layer -1; layer >= 0; layer--){
        for (int nodeIn = 0; nodeIn < neuralNetwork[layer].size(); nodeIn++){
            neuralNetwork[layer][nodeIn].reset();
        }
    }

    return 0;
}
/*!
    @brief sets the current value of a given node if the node is not fixed input

    Returns -1 if the node is a fixed input node.
            0 on success.

    @param int newValue

    @return -1 if the node is a fixed input node.
            0 on success.

    */
int NN::train(std::string filename, int epochs, double lr) {
    // const char*  name = "logger.pb";
    // TensorBoardLogger logger(name);
    // int lossC = 0;
    // int accC = 0;
    for(int epoch = 0; epoch < epochs; epoch++){
        //load in training examples:
        std::ifstream in(filename);
        std::string currLine;
        std::getline(in, currLine);

        std::vector<double> params = split(currLine);
        for(int example = 0; example < params[0]; example++){
            std::getline(in, currLine);
            if(currLine[0] == ' '){
                break;
            }
            std::vector<double> ex = split(currLine);
            auto first = ex.cbegin();
            auto last = ex.cbegin() + params[1];
            auto first2 = ex.cbegin() + params[1];
            auto last2 = ex.cend();
            std::vector<double> inputVec(first, last);
            std::vector<double> outVec(first2,last2);

            eval(inputVec);
            deltas(outVec);
            updateWeights(lr);
            cleanUpBackProb();

        }
        in.close();
        std::cout << "completed epoch #" << epoch << std::endl;


        // if(epoch % 20 == 0){
        //     double lossTot = 0;
        //     for(int i = 0; i < runningLoss.size(); i++){
        //         lossTot += runningLoss[i];
        //     }
        //     lossTot /= runningLoss.size();
        //     logger.add_scalar("loss", lossC, lossTot);
        //     lossC++;
        //     runningLoss.clear();
        // }

        // if(epoch % 50 == 0){
        //     double accTot = 0;
        //     for(int i = 0; i < runningAcc.size(); i++){
        //         accTot += runningAcc[i];
        //     }
        //     accTot /= runningAcc.size();
        //     logger.add_scalar("acc", accC, accTot);
        //     accC++;
        //     runningAcc.clear();
        // }
    }


    return 0;
}




//Node Class
/*!
    @brief sets the current value of a given node if the node is not fixed input

    Returns -1 if the node is a fixed input node.
            0 on success.

    @param int newValue

    @return -1 if the node is a fixed input node.
            0 on success.

    */
int NN::node::setCurrentValue(double newValue) {
    if(!fixed) {
        currValue = newValue;
    }else {
        std::cerr << "Bruh, you can't change fixed input";
        return -1;
    }
    return 0;
}

void NN::node::reset() {
    if(!fixed){
        currValue = 0;
        weightedSum = 0;
    }else{
        currValue = -1;
        weightedSum = -1;
    }
    delta = 0;
}