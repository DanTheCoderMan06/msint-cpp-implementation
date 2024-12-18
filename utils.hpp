#ifndef UTILS_HPP
#define UTILS_HPP

#include <Eigen/Dense>

Eigen::MatrixXd one_hot(const Eigen::VectorXi& Y, int output_size);

Eigen::MatrixXd ReLU(const Eigen::MatrixXd& Z);
Eigen::MatrixXd ReLU_deriv(const Eigen::MatrixXd& Z);
Eigen::MatrixXd softmax(const Eigen::MatrixXd& Z);

#endif
