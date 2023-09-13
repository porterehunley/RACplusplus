#include "<unordered_map>"
#include "<vector>"
#include "<tuple>"

#include "Eigen/Dense"
#include "Eigen/Sparse"


#ifndef CLUSTER_H
#define CLUSTER_H

class Cluster {
public:
    int id;
    bool will_merge;
    int nn;
    std::vector<std::pair<int, double>> neighbor_distances;
    std::vector<int> neighbors;
    std::vector<int> indices;
    std::unordered_map<int, double> dissimilarities; //TODO get rid of this with compute
    std::vector<std::tuple<int, int, double> > neighbors_needing_updates;
    
    Cluster(int id);

    void update_nn(double max_merge_distance);
    void update_nn(Eigen::MatrixXd& distance_arr, double max_merge_distance);
};

#endif // CLUSTER_H


#ifndef RAC_H
#define RAC_H

class RAC {
public:
    RAC(
        Eigen::MatrixXd& base_arr,
        double max_merge_distance,
        double max_merge_distance = 1,
        Eigen::SparseMatrix<bool>* = nullptr,
        int batch_size = 0,
        int no_processors = 1,
        std::string distance_metric = "euclidean");

private:
    Eigen::MatrixXd& distance_arr;
    Eigen::SparseMatrix<bool>* connectivity;

    std::vector<Cluster> clusters;

    int batch_size;
    int no_processors;
    std::string distance_metric;

    double max_merge_distance;
};

#endif //RAC_H