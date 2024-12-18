#include "NeuralNetwork.hpp"
#include "utils.hpp"
#include <iostream>
#include <random>

using namespace std;

/*
    Parameters are initialized later on.
*/

NeuralNetwork::NeuralNetwork(int input_size, int hidden_size, int output_size, double alpha)
: input_size(input_size), hidden_size(hidden_size), output_size(output_size), alpha(alpha)
{
}

//Random filler values

void NeuralNetwork::init_params() {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(-0.5, 0.5);

    W1 = Eigen::MatrixXd(hidden_size, input_size);
    b1 = Eigen::VectorXd(hidden_size);
    W2 = Eigen::MatrixXd(output_size, hidden_size);
    b2 = Eigen::VectorXd(output_size);

    for (int i = 0; i < W1.rows(); ++i) {
        for (int j = 0; j < W1.cols(); ++j) {
            W1(i,j) = dist(gen);
        }
        b1(i) = dist(gen);
    }

    for (int i = 0; i < W2.rows(); ++i) {
        for (int j = 0; j < W2.cols(); ++j) {
            W2(i,j) = dist(gen);
        }
        b2(i) = dist(gen);
    }
}

void NeuralNetwork::forward_prop(const Eigen::MatrixXd& X) {
    Z1 = (W1 * X).colwise() + b1;

    A1 = ReLU(Z1);

    Z2 = (W2 * A1).colwise() + b2;
    A2 = softmax(Z2);
}

void NeuralNetwork::backward_prop(const Eigen::MatrixXd& X, const Eigen::VectorXi& Y) {
    Eigen::MatrixXd one_hot_Y = one_hot(Y, output_size);
    Eigen::MatrixXd dZ2 = A2 - one_hot_Y;
    m = X.cols(); 

    dW2 = (1.0/(double)m) * (dZ2 * A1.transpose());
    db2 = (1.0/(double)m) * dZ2.sum(); 

    Eigen::MatrixXd dZ1 = (W2.transpose() * dZ2).array() * ReLU_deriv(Z1).array();
    dW1 = (1.0/(double)m) * (dZ1 * X.transpose());
    db1 = (1.0/(double)m) * dZ1.sum();
}

void NeuralNetwork::update_params() {
    W1 = W1 - alpha * dW1;
    b1 = b1.array() - alpha * db1;
    W2 = W2 - alpha * dW2;
    b2 = b2.array() - alpha * db2;
}

Eigen::VectorXi NeuralNetwork::get_predictions() {
    Eigen::VectorXi predictions(A2.cols());
    for (int i = 0; i < A2.cols(); ++i) {
        A2.col(i).maxCoeff(&predictions(i));
    }
    return predictions;
}

double NeuralNetwork::get_accuracy(const Eigen::VectorXi& predictions, const Eigen::VectorXi& Y) {
    int correct = 0;
    for (int i = 0; i < Y.size(); ++i) {
        if (predictions(i) == Y(i)) {
            correct++;
        }
    }
    return (double)correct / (double)Y.size();
}

void NeuralNetwork::train(const Eigen::MatrixXd& X, const Eigen::VectorXi& Y, int iterations) {
    this->m = X.cols();
    init_params();

    for (int i = 0; i < iterations; ++i) {
        forward_prop(X);
        backward_prop(X, Y);
        update_params();

        if (i % 20 == 0) {
            cout << "Iteration: " << i << endl;
            Eigen::VectorXi preds = get_predictions();
            double acc = get_accuracy(preds, Y);
            cout << "Accuracy: " << acc << endl;
        }
    }
}
