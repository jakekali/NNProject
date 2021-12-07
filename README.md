# NNProject
## How to compile
Download and run make
## Datasets Tested
### Provided WDBC Breast Cancer Dataset
1. Trial #1: [Trained for 100 epochs, hidden layer of size 5, with a learning rate of 0.1](bb/readMe.md)
### Provided Grades Dataset
1. Trial #2: [Trained for 100 epochs, hidden layer of size 10, with a learning rate of 0.05](grades/README.md)
### MNIST Dataset
The MNIST dataset is an acronym that stands for the Modified National Institute of Standards and Technology dataset. It is a dataset of 60,000 small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively. <br />
The data was obtained from one of the sample datasets avaible through [Google Colab](https://colab.research.google.com/drive/1s3PtLj_46hLp9kKiYipDcBYfkNRwmZFH?usp=sharing). I performed some manipulations using pandas, in colab, to normalize the data by dividing by 255, and to hot encode an output vector the output.  <br />
**Input:** 784 doubles between 0 and 1 presenting the flattened pixel values of a 28x28 pixel grayscale image. The pixel values were divided by 255 to normalize the data between 0 and 1. Available  @ [fmnist/fmnist.training.txt](fmnist/fmnist.training.txt) <br />
**Output:** 10 binary classes presenting the numbers 0 through 1. Available  @ [fmnist/fmnist.test.txt](fmnist/fmnist.test.txt) <br />
**Initial Weights:** The Inital Weights (for each network) were generated using the c++ standard rand() function using the following code:
```cpp
        double f = (double)rand() / RAND_MAX;
        weightsPreviou = (-1) + f * (2);
```
**Trials Performed: (Fair Warning - Some these took several hours to completly train. I ran them on kahan.ee.cooper.edu over the course of several days.**
1. Trial #1: [Trained for 500 epochs, hidden layer of size 25, with a learning rate of 0.01]()
2. Trial #2: [Trained for 500 epochs, hidden layer of size 25, with a learning rate of 0.05]()
3. Trial #3: [Trained for 2000 epochs, hidden layer of size 64, with a learning rate of 0.05]()
4. Trial #4: [Trained for 2000 epochs, hidden layer of size 64, with a learning rate of 0.75]()
5. Trial #5: [Trained for 2000 epochs, hidden layer of size 64, with a learning rate of 0.2]()
6. Trial #6: [Trained for 3000 epochs, hidden layer of size 64, with a learning rate of 0.005]()
