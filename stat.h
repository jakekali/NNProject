//
// Created by Jacob on 11/30/2021.
//

#ifndef NN_STAT_H
#define NN_STAT_H


class stat {
public:
//                 Expected = 1	   Expected = 0
//    Predicted = 1	    A	            B
//    Predicted = 0	    C	            D
    int A =0;
    int B =0;
    int C =0;
    int D =0;
    double getAccuracy() {
        return (1.0) * ((double) (A + D) / (double) (A + B + C + D));
    }
    double getPrecision() {
        return (double) (1.0) *  ((double) A/ (double) (A+B));
    }

    double getRecall(){
        return (double) (1.0) *  ((double) A/ (double) (A+C));
    };
    double getF1(){
        return (2 * (double) getPrecision() * (double) getRecall())/ (double) (getPrecision() + (double) getRecall());
    };
};


#endif //NN_STAT_H
