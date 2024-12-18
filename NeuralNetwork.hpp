#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <Eigen/Dense>

class NeuralNetwork {
public:
    NeuralNetwork(int input_size, int hidden_size, int output_size, double alpha);
    void init_params();
    void forward_prop(const Eigen::MatrixXd& X);
    void backward_prop(const Eigen::MatrixXd& X, const Eigen::VectorXi& Y);
    void update_params();
    void train(const Eigen::MatrixXd& X, const Eigen::VectorXi& Y, int iterations);

    Eigen::VectorXi get_predictions();
    double get_accuracy(const Eigen::VectorXi& predictions, const Eigen::VectorXi& Y);

    Eigen::MatrixXd Z1, A1, Z2, A2;

private:
    int input_size;
    int hidden_size;
    int output_size;
    double alpha; 
    int m;         

    Eigen::MatrixXd W1;
    Eigen::VectorXd b1;
    Eigen::MatrixXd W2;
    Eigen::VectorXd b2;

    Eigen::MatrixXd dW1;
    double db1; 
    Eigen::MatrixXd dW2;
    double db2;
};

#endif
