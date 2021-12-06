#include <iostream>
#include <vector>
#include "NN.cpp"
#include "stat.cpp"
using json = nlohmann::json;



int fileEval(const std::string& filein, const std::string& fileout, NN &j, std::vector<int> sizer){
    //open file
    std::ifstream in(filein);
    std::string currLine;

    std::filebuf fb;
    fb.open (fileout,std::ios::out);
    std::ostream os(&fb);

    //just to skip first line
    std::getline(in, currLine);
    std::vector<stat> stats;
    stats.resize(sizer[sizer.size()-1]);
    int k = 0;
    while(std::getline(in, currLine)){
        std::vector<double> ex = split(currLine);
        auto first = ex.cbegin();
        auto last = ex.cbegin() + sizer[0];
        auto first2 = ex.cbegin() + sizer[0];
        auto last2 = ex.cend();
        std::vector<double> inputVec(first, last);
        std::vector<double> outVec(first2,last2);

        auto ret = j.eval(inputVec);

        for(int i = 0; i < stats.size(); i++){

            if(ret[i] >= 0.5 && outVec[i] == 1){
                stats[i].A++;
            }else if(ret[i] >= 0.5 && outVec[i] == 0){
                stats[i].B++;
            }else if(ret[i] < 0.5 && outVec[i] == 1){
                stats[i].C++;
            }else if(ret[i] < 0.5 && outVec[i] == 0){
                stats[i].D++;
            }
        }
        k++;
    }
    stat totStats;
    double acc = 0;
    double prec = 0;
    double recall = 0;


    for(int i = 0; i < stats.size(); i++) {
        //Micro
        totStats.A += stats[i].A;
        totStats.B += stats[i].B;
        totStats.C += stats[i].C;
        totStats.D += stats[i].D;
        //Macro
        acc += stats[i].getAccuracy();
        prec += stats[i].getPrecision();
        recall += stats[i].getRecall();

        os << stats[i].A << " " << stats[i].B << " " << stats[i].C << " " << stats[i].D << " ";
        os <<  std::setprecision(3) << std::fixed << stats[i].getAccuracy() << " ";
        os <<  std::setprecision(3) << std::fixed << stats[i].getPrecision() << " ";
        os <<  std::setprecision(3) << std::fixed << stats[i].getRecall() << " ";
        os <<  std::setprecision(3) << std::fixed << stats[i].getF1() << "\n";
    }

    //Get Micro Average

    os <<  std::setprecision(3) << std::fixed << totStats.getAccuracy() << " ";
    os <<  std::setprecision(3) << std::fixed << totStats.getPrecision() << " ";
    os <<  std::setprecision(3) << std::fixed << totStats.getRecall() << " ";
    os <<  std::setprecision(3) << std::fixed << totStats.getF1() << "\n";

    //Get Macro Averages
    os <<  std::setprecision(3) << std::fixed << acc/stats.size() << " ";
    os <<  std::setprecision(3) << std::fixed << prec/stats.size() << " ";
    os <<  std::setprecision(3) << std::fixed << recall/stats.size() << " ";
    os <<  std::setprecision(3) << std::fixed << (2 * prec/stats.size() * recall/stats.size())/(prec/stats.size() + recall/stats.size()) << "\n";


    fb.close();

    return 0;

}
int main() {
    std::cout << "Select an Option: \n1. Test \n2. Train\n";
    std::string opt;
    std::cin >> opt;
    if(opt == "1"){
        //User Input
        std::cout << "Enter the name of a trained network file: \n";
        std::string weightsIn;
        std::cin >> weightsIn;

        std::cout << "Enter the name of a testing file: \n";
        std::string testFile;
        std::cin >> testFile;

        std::cout << "Enter the name of an output file: \n";
        std::string outfile;
        std::cin >> outfile;

       //get the first line of the weights file
        std::ifstream in(weightsIn);
        //check if the file exists
        if(!in){
            std::cout << "File does not exist\n";
            return 0;
        }
        std::string currLine;
        std::getline(in, currLine);
        //close file
        in.close();
    


        auto sizes = split(currLine);
        std::vector<int> nnSize(sizes.begin(),sizes.end());

        NN j = NN(nnSize);
        j.loadWeightsFromFile(weightsIn);
        fileEval(testFile, outfile, j,nnSize);


    }else if (opt == "2"){
        std::cout << "Enter the name of inital weights file: \n";
        std::string weightsIn;
        std::cin >> weightsIn;
        std::cout << "Enter the name of a training file: \n";
        std::string testFile;
        std::cin >> testFile;
        std::cout << "Enter the name of an output file: \n";
        std::string outfile;
        std::cin >> outfile;
        std::cout << "Enter a positive integer of epochs: \n";
        std::string epochs;
        std::cin >> epochs;
        std::cout << "Enter a floating point learning rate: \n";
        std::string lr;
        std::cin >> lr;

        int numEpochs = std::stoi(epochs);
        double numLr = std::stod(lr);

        //get the first line of the weights file
        std::ifstream in(weightsIn);
        //check if the file exists
        if(!in){
            std::cout << "File does not exist\n";
            return 0;
        }
        std::string currLine;
        std::getline(in, currLine);
        //close file
        in.close();

        //check if the file exists
        std::ifstream ll(testFile);
        if(!ll){
            std::cout << "File does not exist\n";
            return 0; 
        }
        ll.close();
        
        auto sizes = split(currLine);
        std::vector<int> nnSize(sizes.begin(),sizes.end());

        NN j = NN(nnSize);
        j.loadWeightsFromFile(weightsIn);
        j.train(testFile, numEpochs, numLr);
        j.exportNetwork(outfile, true);
    }else{
        std::cout << "Enter size of input layer: \n";
        std::string inLayerString;
        std::cin >> inLayerString;
        std::cout << "Enter size of hidden layer: \n";
        std::string hiddenLayerString;
        std::cin >> hiddenLayerString;
        std::cout << "Enter size of output layer: \n";
        std::string outputLayerString;
        std::cin >> outputLayerString;

        //convert to string to ints
        int inLayer = std::stoi(inLayerString);
        int hiddenLayer = std::stoi(hiddenLayerString);
        int outputLayer = std::stoi(outputLayerString);
        NN j = NN({inLayer, hiddenLayer, outputLayer}); 
        std::cout << "Enter output file name: \n";
        std::string outFile;
        std::cin >> outFile;
        j.exportNetwork(outFile, true);
    }
    return 0;
}