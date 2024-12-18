#include "utils.hpp"
#include <cmath>

Eigen::MatrixXd one_hot(const Eigen::VectorXi& Y, int output_size) {
    Eigen::MatrixXd one_hot_mat = Eigen::MatrixXd::Zero(output_size, Y.size());
    
    for (int i = 0; i < Y.size(); ++i) {
        one_hot_mat(Y(i), i) = 1.0;
    }

    return one_hot_mat;
}

Eigen::MatrixXd ReLU(const Eigen::MatrixXd& Z) {
    return Z.cwiseMax(0.0);
}

Eigen::MatrixXd ReLU_deriv(const Eigen::MatrixXd& Z) {
    Eigen::MatrixXd dZ = Z;
    for (int i = 0; i < dZ.rows(); ++i) {
        for (int j = 0; j < dZ.cols(); ++j) {
            dZ(i,j) = (Z(i,j) > 0) ? 1.0 : 0.0;
        }
    }
    return dZ;
}

Eigen::MatrixXd softmax(const Eigen::MatrixXd& Z) {
    Eigen::MatrixXd Z_shifted = Z;
    for (int i = 0; i < Z.cols(); ++i) {
        double max_val = Z.col(i).maxCoeff();
        Z_shifted.col(i) = Z.col(i).array() - max_val;
    }

    Eigen::MatrixXd expZ = Z_shifted.array().exp();
    Eigen::MatrixXd sumExp = expZ.colwise().sum();
    for (int i = 0; i < Z.cols(); ++i) {
        expZ.col(i) = expZ.col(i) / sumExp(i);
    }
    return expZ;
}
