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
    RAC(double max_merge_distance = 1,
        Eigen::SparseMatrix<bool>* connectivity = nullptr,
        int batch_size = 0,
        int no_processors = 1,
        std::string distance_metric = "euclidean");

    void fit(Eigen::MatrixXd& base_arr);
    std::vector<int> predict();
    std::vector<int> fit_predict(Eigen::MatrixXd& base_arr);


private:
    // Instance vars
    Eigen::MatrixXd& distance_arr;
    Eigen::SparseMatrix<bool>* connectivity;
    std::vector<Cluster> clusters;
    int batch_size;
    int no_processors;
    std::string distance_metric;
    double max_merge_distance;

    // Methods
    Eigen::MatrixXd calculate_initial_dissimilarities(Eigen::MatrixXd& base_arr);
    void merge_clusters_full(
        std::vector<std::pair<int, int> >& merges,
        std::vector<std::pair<int, int> >& all_merges);
    void merge_cluster_full(
        std::pair<int, int>& merge,
        std::vector<std::pair<int, int>>& merges);
    void parallel_merge_clusters(std::vector<std::pair<int,int>> merges);
    void update_cluster_neighbors(std::vector<std::pair<int, int>> merges);
    void update_cluster_nn_dist();
    void update_cluster_dissimilarities(std::vector<std::pair<int, int> >& merges);
    std::vector<std::pair<int, int>> find_reciprocal_nn();
    std::vector<Cluster*> make_clusters(int no_clusters);
    std::vector<int> get_cluster_indices();
};

#endif //RAC_H