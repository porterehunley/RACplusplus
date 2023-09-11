#include "Eigen/Dense"

#include "_distances.h"


Eigen::MatrixXd pairwise_cosine(const Eigen::MatrixXd& array_a, const Eigen::MatrixXd& array_b) {
    return Eigen::MatrixXd::Ones(array_a.cols(), array_b.cols()) - (array_a.transpose() * array_b);
}

Eigen::MatrixXd pairwise_euclidean(const Eigen::MatrixXd& array_a, const Eigen::MatrixXd& array_b) {
    Eigen::MatrixXd D = -2.0 * array_a.transpose() * array_b;
    D.colwise() += array_a.colwise().squaredNorm().transpose();
    D.rowwise() += array_b.colwise().squaredNorm();

    return D.array().sqrt();
}

// TODO fix the manhatten distance function

// Eigen::MatrixXd pairwise_manhattan(const Eigen::MatrixXd& array_a, const Eigen::MatrixXd& array_b) {
//     Eigen::MatrixXd abs_diff = (array_a.colwise() - array_b.colwise().transpose()).array().abs();
//     Eigen::VectorXd manhattan_distances = abs_diff.rowwise().sum();
//     Eigen::MatrixXd D = manhattan_distances.replicate(array_a.rows(), 1);

//     return D;
// }
