#include <tuple>
#include <vector>
#include <thread>
#include <algorithm>

#include "RAC.h"
#include "distances/_distances.h"
#include "utils.h"


void RAC::RAC(
    double max_merge_distance = 1,
    Eigen::SparseMatrix<bool>* connectivity = nullptr,
    int batch_size = 0,
    int no_processors = 1,
    std::string distance_metric = "euclidean") {

    this->max_merge_distance = max_merge_distance;

    // CONNECTIVITY must be symmetric
    this->connectivity = connectivity;

    this->batch_size = batch_size;
    this->no_processors = no_processors;
    this->distance_metric = distance_metric;
}


Eigen::MatrixXd RAC::calculate_initial_dissimilarities(Eigen::MatrixXd& base_arr) {
    Eigen::MatrixXd distance_mat;
    switch (distance_metric) {
    case "cosine":
        distance_mat = pairwise_cosine(base_arr, base_arr).array();
        break;

    case "manhatten":
        distance_mat = pairwise_manhattan(base_arr, base_arr).array();
        break;

    default:
        distance_mat = pairwise_euclidean(base_arr, base_arr).array();
        break;
    }

    size_t clusterSize = clusters.size();
    for (size_t i=0; i<clusterSize; i++) {
        double min = std::numeric_limits<double>::infinity();
        int nn = -1;

        auto currentCluster = clusters[i];

        for (size_t j=0; j<clusterSize; j++) {
            if (i == j) {
                distance_mat(i, j) = std::numeric_limits<double>::infinity();
                continue;
            }

            double distance = distance_mat(i, j);
            if (distance < max_merge_distance) {
                currentCluster->neighbors.push_back(j);

                if (distance < min) {
                    min = distance;
                    nn = j;
                }
            }
        }

        currentCluster -> nn = nn;
    }

    return distance_mat;
}


void RAC::merge_clusters_full(
    std::vector<std::pair<int, int> >& merges,
    std::vector<std::pair<int, int> >& all_merges) {

    for (std::pair<int, int> merge : merges) {
        merge_cluster_full(merge, all_merges);
    }
}


void RAC::merge_cluster_full(
    std::pair<int, int>& merge,
    std::vector<std::pair<int, int>>& merges) {

    Cluster* main_cluster = clusters[merge.first];
    Cluster* secondary_cluster = clusters[merge.second];

    // Get main and secondary columns from distance array
    Eigen::VectorXd main_col = distance_arr.col(main_cluster->id);
    Eigen::VectorXd secondary_col = distance_arr.col(secondary_cluster->id);


    // Loop through merges and change main_col vals
    for (auto& merge : merges) {
        if (merge.first == main_cluster->id || merge.second == main_cluster->id) {
            continue;
        }

        int merge_main = merge.first;
        int merge_secondary = merge.second;

        int merge_main_size = clusters[merge_main]->indices.size();
        int merge_secondary_size = clusters[merge_secondary]->indices.size();

        main_col[merge_main] = (merge_main_size * main_col[merge_main] + merge_secondary_size * main_col[merge_secondary]) / (merge_main_size + merge_secondary_size);
        main_col[merge_secondary] = main_col[merge_main];

        secondary_col[merge_main] = (merge_main_size * secondary_col[merge_main] + merge_secondary_size * secondary_col[merge_secondary]) / (merge_main_size + merge_secondary_size);
        secondary_col[merge_secondary] = secondary_col[merge_main];
    }

    int main_size = main_cluster->indices.size();
    int secondary_size = secondary_cluster->indices.size();

    // average main_col and secndary_col
    Eigen::VectorXd avg_col = (main_size * main_col + secondary_size * secondary_col) / (main_size + secondary_size);
    avg_col[secondary_cluster->id] = std::numeric_limits<double>::infinity();

    // Swap main and secondary columns for avg col
    distance_arr.col(main_cluster->id) = avg_col;

    // Set secondary column to infinity
    main_cluster->indices.insert(
        main_cluster->indices.end(),
        secondary_cluster->indices.begin(),
        secondary_cluster->indices.end());
}


void RAC::parallel_merge_clusters(std::vector<std::pair<int,int>> merges) {
    std::vector<std::thread> threads;

    std::vector<std::vector<std::pair<int, int>>> merge_chunks;
    merge_chunks = utils::chunk_vector(merges, no_threads);

    for (size_t i=0; i<no_threads; i++) {
        std::thread merge_thread = std::thread(
            merge_clusters_full,
            std::ref(merge_chunks[i]),
            std::ref(merges));

        threads.push_back(std::move(merge_thread));
    }

    for (size_t i=0; i<no_threads; i++) {
        threads[i].join();
    }
}


void RAC::update_cluster_neighbors(std::vector<std::pair<int, int>> merges) {

    for (size_t i=0; i<merges.size(); i++) {
        int cluster_id = merges[i].first;

        distance_arr.row(cluster_id) = distance_arr.col(cluster_id);
    }

    for (size_t i=0; i<merges.size(); i++) {
        int cluster_id = merges[i].second;

        distance_arr.col(cluster_id) = Eigen::VectorXd::Constant(
            distance_arr.cols(), std::numeric_limits<double>::infinity());

        distance_arr.row(cluster_id) = distance_arr.col(cluster_id);
    }
}


void RAC::update_cluster_nn_dist() {
    for (Cluster* cluster : clusters) {
        if (cluster == nullptr) {
            continue;
        }

        if (cluster->will_merge || (cluster->nn != -1 && clusters[cluster->nn] != nullptr && clusters[cluster->nn]->will_merge)) {
            cluster->update_nn(distance_arr, max_merge_distance);
        }
    }
}


void RAC::update_cluster_dissimilarities(std::vector<std::pair<int, int> >& merges) {
    if (merges.size() / no_processors > 10) {
        // TODO refactor this
        parallel_merge_clusters(merges, distance_arr, clusters, 1);
    } else {

        for (std::pair<int, int> merge : merges) {
            merge_cluster_full(merge, merges, clusters, distance_arr);
        }
    }

    update_cluster_neighbors(distance_arr, merges);
}


std::vector<std::pair<int, int>> RAC::find_reciprocal_nn() {
    std::vector<std::pair<int, int>> reciprocal_nn;
    for (Cluster* cluster : clusters) {
        if (cluster == nullptr) {
            continue;
        }

        cluster -> will_merge = false;
        if (cluster->nn != -1 && clusters[cluster->nn] != nullptr) {
            cluster->will_merge = (clusters[cluster->nn]->nn == cluster->id);
        }

        if (cluster->will_merge && cluster->id < cluster->nn) {
            reciprocal_nn.push_back(std::make_pair(cluster->id, cluster->nn));
        }
    }

    return reciprocal_nn;
}


std::vector<Cluster*> RAC::make_clusters(int no_clusters) {
    std::vector<Cluster*> new_clusters;
    for (long i = 0; i < base_arr.cols(); ++i) {
        Cluster* cluster = new Cluster(i);
        new_clusters.push_back(cluster);
    }

    return new_clusters
}


std::vector<int> RAC::get_cluster_indices() {
    std::vector<std::pair<int, int> > cluster_idx;
    for (Cluster* cluster : clusters) {
        if (cluster == nullptr) {
            continue;
        }

        for (int index : cluster->indices)  {
            cluster_idx.push_back(std::make_pair(index, cluster->id));
        }
    }

    std::sort(cluster_idx.begin(), cluster_idx.end());

    std::vector<int> cluster_labels;
    for (const auto& [index, cluster_id] : cluster_idx) {
        cluster_labels.push_back(cluster_id);
    }

    return cluster_labels;
}


std::vector<int> RAC::predict() {
    std::vector<std::pair<int, int>> merges = find_reciprocal_nn()
    while (merges.size() != 0) {
        update_cluster_dissimilarities(merges)

        update_cluster_nn_dist()

        remove_secondary_clusters(merges)

        merges = find_reciprocal_nn()
    }

    return get_cluster_indices()
}


void RAC::fit(Eigen::MatrixXd& base_arr) {
    if (distance_metric == "cosine") {
        base_arr = base_arr.transpose().colwise().normalized().eval();
    } else {
        base_arr = base_arr.transpose().eval();
    }

    // Initialize internal points
    clusters = make_clusters(base_arr.cols())
    distance_arr = calculate_initial_dissimilarities(base_arr)
}

std::vector<int> fit_predict(Eigen::MatrixXd& base_arr) {
    fit(base_arr)
    return predict()
}

