#ifndef DISTANCES_H 
#define DISTANCES_H 

#include "Eigen/Dense"


Eigen::MatrixXd pairwise_cosine(const Eigen::MatrixXd& array_a, const Eigen::MatrixXd& array_b);

Eigen::MatrixXd pairwise_euclidean(const Eigen::MatrixXd& array_a, const Eigen::MatrixXd& array_b);

Eigen::MatrixXd pairwise_manhattan(const Eigen::MatrixXd& array_a, const Eigen::MatrixXd& array_b);

#endif
