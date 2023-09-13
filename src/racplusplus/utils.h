#include <vector>


#ifndef UTILS_H
#define UTILS_H

namespace utils {
    template <typename T>
    std::vector<std::vector<T>> chunk_vector(
        std::vector<T>& vector, size_t no_chunks);
}

#endif //UTILS_H
