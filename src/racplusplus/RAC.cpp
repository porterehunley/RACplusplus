#include <tuple>
#include <vector>
#include <thread>

#include "RAC.h"
#include "distances/_distances.h"
#include "utils.h"


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


void RAC::update_cluster_neighbors(
    Eigen::MatrixXd& distance_arr,
    std::vector<std::pair<int, int>> merges) {

    // copy merges columns to rows in distance_arr
    for (size_t i=0; i<merges.size(); i++) {
        int cluster_id = merges[i].first;

        distance_arr.row(cluster_id) = distance_arr.col(cluster_id);
    }

    for (size_t i=0; i<merges.size(); i++) {
        int cluster_id = merges[i].second;

        distance_arr.col(cluster_id) = Eigen::VectorXd::Constant(distance_arr.cols(), std::numeric_limits<double>::infinity());
        distance_arr.row(cluster_id) = distance_arr.col(cluster_id);
    }
}
