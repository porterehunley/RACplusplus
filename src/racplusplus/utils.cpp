#include <vector>


template <typename T>
std::vector<std::vector<T> > chunk_vector(
    std::vector<T>& vector,
    size_t no_chunks) {

    std::vector<std::vector<T> > vector_chunks(no_chunks);

    size_t chunk_size = vector.size() / no_chunks;
    size_t remainder = vector.size() % no_chunks; 

    size_t start = 0, end = 0;
    for (size_t i = 0; i < no_chunks; i++) {
        end = start + chunk_size;
        if (i < remainder) { // distribute the remainder among the first "remainder" chunks
            end++;
        }

        if (end <= vector.size()) {
            vector_chunks[i] = std::vector<T>(vector.begin() + start, vector.begin() + end);
        } 
        start = end;
    }

    return vector_chunks;
}
