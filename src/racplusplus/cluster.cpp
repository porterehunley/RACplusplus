#include <RAC.h>

#include "Eigen/Dense"


Cluster::update_nn(double max_merge_distance) {
    if (neighbor_distances.size() == 0) {
        nn = -1;
        return;
    }

    double min = std::numeric_limits<double>::infinity();
    int nn = -1;

    for (auto& neighbor : neighbor_distances) {
        double dissimilarity = neighbor.second;
        if (dissimilarity < min) {
            min = dissimilarity;
            nn = neighbor.first;
        }
    }

    if (min < max_merge_distance) {
        this->nn = nn;
    } else {
        this->nn = -1;
    }
}


Cluster::update_nn(Eigen::MatrixXd& distance_arr, double max_merge_distance) {
    Eigen::MatrixXd::Index minRow;
    distance_arr.col(id).minCoeff(&minRow);

    double min = distance_arr(minRow, id);
    int nn = static_cast<int>(minRow);

    if (min < max_merge_distance) {
        this->nn = nn;
    } else {
        this->nn = -1;
    }
}