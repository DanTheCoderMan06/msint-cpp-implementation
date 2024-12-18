#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "NeuralNetwork.hpp"
#include "utils.hpp"
#include <Eigen/Dense>

using namespace std;

int main() {
    string filename;

    cout << "Enter filename!" << endl;
    cin >> filename;

    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening " << filename << endl;
        return 1;
    }

    string line;
    if (!getline(file, line)) {
        cerr << "Error reading header from " << filename << endl;
        return 1;
    }

    int input_size = 784; 
    int m_train = 0;

    while (getline(file, line)) {
        if (!line.empty())
            m_train++;
    }
    file.close();

    Eigen::MatrixXd X_train(input_size, m_train);
    Eigen::VectorXi Y_train(m_train);

    file.open(filename);
    getline(file, line);

    int col = 0;
    while (getline(file, line)) {
        if (line.empty()) continue;

        stringstream ss(line);
        string val;

        if (!getline(ss, val, ',')) {
            cerr << "Error: Missing label in line" << endl;
            return 1;
        }
        int label = stoi(val);
        Y_train(col) = label;

        for (int i = 0; i < input_size; i++) {
            if (!getline(ss, val, ',')) {
                cerr << "Error: Missing pixel value in line" << endl;
                return 1;
            }
            double pixel = stod(val);
            X_train(i, col) = pixel;
        }

        col++;
    }
    file.close();

    X_train.array() /= 255.0;

    int hidden_size = 10; 
    int output_size = 10;  
    double alpha = 0.10;
    int iterations = 0;

    cout << "How many iterations shall happen? (Must be multiple of 25, if not is rounded)" << endl;
    cin >> iterations;

    iterations = (iterations/25) * 25;
    iterations = (iterations > 0) ? iterations : 25;

    NeuralNetwork nn(input_size, hidden_size, output_size, alpha);
    nn.train(X_train, Y_train, iterations);

    Eigen::VectorXi preds = nn.get_predictions();
    double acc = nn.get_accuracy(preds, Y_train);
    cout << "Final training accuracy: " << acc << endl;

    return 0;
}
