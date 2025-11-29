/**
 * @file morris.hpp
 * @brief Header-only C++14 implementation of Morris Screening Method with OpenMP parallelization
 *
 * This library implements the Method of Morris for global sensitivity analysis,
 * including optimized trajectory sampling (Campolongo et al., 2007) with local
 * optimization (Ruano et al., 2012) and analysis of elementary effects.
 *
 * References:
 * - Morris, M.D. (1991). Factorial Sampling Plans for Preliminary Computational Experiments.
 *   Technometrics, 33(2), 161-174.
 * - Campolongo, F., Cariboni, J., & Saltelli, A. (2007). An effective screening design
 *   for sensitivity analysis of large models. Environmental Modelling & Software, 22(10), 1509-1518.
 * - Ruano, M.V., Ribes, J., Seco, A., Ferrer, J. (2012). An improved sampling strategy
 *   based on trajectory design for application of the Morris method to systems with many
 *   input factors. Environmental Modelling & Software, 37, 103-109.
 *
 * @author Port from Python SALib
 * @version 1.0
 */

#ifndef MORRIS_HPP
#define MORRIS_HPP

#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <utility>
#include <tuple>
#include <set>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace morris {

// ============================================================================
// Type Aliases and Constants
// ============================================================================

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

// ============================================================================
// Matrix Utility Functions
// ============================================================================

/**
 * @brief Create a matrix filled with zeros
 */
inline Matrix zeros(size_t rows, size_t cols) {
    return Matrix(rows, Vector(cols, 0.0));
}

/**
 * @brief Create a matrix filled with ones
 */
inline Matrix ones(size_t rows, size_t cols) {
    return Matrix(rows, Vector(cols, 1.0));
}

/**
 * @brief Create an identity matrix
 */
inline Matrix eye(size_t n) {
    Matrix result = zeros(n, n);
    for (size_t i = 0; i < n; ++i) {
        result[i][i] = 1.0;
    }
    return result;
}

/**
 * @brief Create a lower triangular matrix of ones
 */
inline Matrix tril(size_t rows, size_t cols) {
    Matrix result = zeros(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols && j < i; ++j) {
            result[i][j] = 1.0;
        }
    }
    return result;
}

/**
 * @brief Matrix multiplication C = A * B
 */
inline Matrix matmul(const Matrix& A, const Matrix& B) {
    if (A.empty() || B.empty() || A[0].size() != B.size()) {
        throw std::invalid_argument("Invalid matrix dimensions for multiplication");
    }

    size_t m = A.size();
    size_t n = B[0].size();
    size_t p = B.size();

    Matrix C = zeros(m, n);

    #pragma omp parallel for collapse(2) if(m * n * p > 1000)
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < p; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }

    return C;
}

/**
 * @brief Matrix transpose
 */
inline Matrix transpose(const Matrix& A) {
    if (A.empty()) return Matrix();

    size_t rows = A.size();
    size_t cols = A[0].size();
    Matrix result = zeros(cols, rows);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[j][i] = A[i][j];
        }
    }

    return result;
}

/**
 * @brief Create a diagonal matrix from a vector
 */
inline Matrix diag(const Vector& v) {
    size_t n = v.size();
    Matrix result = zeros(n, n);
    for (size_t i = 0; i < n; ++i) {
        result[i][i] = v[i];
    }
    return result;
}

/**
 * @brief Matrix subtraction C = A - B
 */
inline Matrix matrix_subtract(const Matrix& A, const Matrix& B) {
    if (A.size() != B.size() || (A.size() > 0 && A[0].size() != B[0].size())) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }

    Matrix C = zeros(A.size(), A[0].size());
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[0].size(); ++j) {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
    return C;
}

/**
 * @brief Scalar multiplication of matrix
 */
inline Matrix scalar_mult(double scalar, const Matrix& A) {
    Matrix result = A;
    for (auto& row : result) {
        for (auto& val : row) {
            val *= scalar;
        }
    }
    return result;
}

/**
 * @brief Matrix addition C = A + B
 */
inline Matrix matrix_add(const Matrix& A, const Matrix& B) {
    if (A.size() != B.size() || (A.size() > 0 && A[0].size() != B[0].size())) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }

    Matrix C = zeros(A.size(), A[0].size());
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[0].size(); ++j) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
    return C;
}

/**
 * @brief Euclidean distance between two points
 */
inline double euclidean_distance(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have same size");
    }

    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

/**
 * @brief Sum of distances between all pairs of points in two trajectories (cdist equivalent)
 */
inline double trajectory_distance(const Matrix& traj1, const Matrix& traj2) {
    if (traj1.empty() || traj2.empty()) return 0.0;
    if (traj1[0].size() != traj2[0].size()) {
        throw std::invalid_argument("Trajectories must have same dimension");
    }

    double total_distance = 0.0;
    for (const auto& p1 : traj1) {
        for (const auto& p2 : traj2) {
            total_distance += euclidean_distance(p1, p2);
        }
    }

    return total_distance;
}

// ============================================================================
// Statistical Functions
// ============================================================================

/**
 * @brief Calculate mean of a vector
 */
inline double mean(const Vector& v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

/**
 * @brief Calculate standard deviation with ddof (delta degrees of freedom)
 */
inline double stddev(const Vector& v, int ddof = 0) {
    if (v.size() <= static_cast<size_t>(ddof)) return 0.0;

    double m = mean(v);
    double sum_sq = 0.0;
    for (double val : v) {
        double diff = val - m;
        sum_sq += diff * diff;
    }

    return std::sqrt(sum_sq / (v.size() - ddof));
}

/**
 * @brief Calculate quantile of standard normal distribution (inverse CDF)
 * Using rational approximation from Abramowitz and Stegun
 */
inline double norm_ppf(double p) {
    if (p <= 0.0 || p >= 1.0) {
        throw std::invalid_argument("Probability must be in (0, 1)");
    }

    // Constants for rational approximation
    const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02,
                        -2.759285104469687e+02, 1.383577518672690e+02,
                        -3.066479806614716e+01, 2.506628277459239e+00};
    const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02,
                        -1.556989798598866e+02, 6.680131188771972e+01,
                        -1.328068155288572e+01};
    const double c[] = {-7.784894002430188e-03, -3.223964580411365e-01,
                        -2.400758277161838e+00, -2.549732539343734e+00,
                        4.374664141464968e+00, 2.938163982698783e+00};
    const double d[] = {7.784695709041462e-03, 3.224671290700398e-01,
                        2.445134137142996e+00, 3.754408661907416e+00};

    double q = p - 0.5;
    double r, x;

    if (std::abs(q) <= 0.425) {
        r = 0.180625 - q * q;
        x = q * (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) /
                (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    } else {
        r = (q < 0.0) ? p : 1.0 - p;
        r = std::sqrt(-std::log(r));

        if (r <= 5.0) {
            r -= 1.6;
            x = (((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]) /
                ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1.0);
        } else {
            r -= 5.0;
            x = (((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]) /
                ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1.0);
        }

        if (q < 0.0) x = -x;
    }

    return x;
}

// ============================================================================
// Morris Sampling Functions
// ============================================================================

/**
 * @brief Compute delta parameter based on number of levels
 * delta = p / (2 * (p - 1))
 */
inline double compute_delta(int num_levels) {
    if (num_levels < 2) {
        throw std::invalid_argument("num_levels must be at least 2");
    }
    return static_cast<double>(num_levels) / (2.0 * (num_levels - 1));
}

/**
 * @brief Generate random initial position x* in parameter space
 */
template<typename RNG>
inline Vector generate_x_star(size_t num_params, int num_levels, RNG& rng) {
    double delta = compute_delta(num_levels);
    double bound = 1.0 - delta;
    int num_grid_points = num_levels / 2;

    // Create grid
    Vector grid(num_grid_points);
    for (int i = 0; i < num_grid_points; ++i) {
        grid[i] = i * bound / (num_grid_points - 1);
    }

    // Random selection from grid
    std::uniform_int_distribution<size_t> dist(0, grid.size() - 1);
    Vector x_star(num_params);
    for (size_t i = 0; i < num_params; ++i) {
        x_star[i] = grid[dist(rng)];
    }

    return x_star;
}

/**
 * @brief Generate random permutation matrix P*
 */
template<typename RNG>
inline Matrix generate_p_star(size_t num_groups, RNG& rng) {
    Matrix p_star = eye(num_groups);
    std::shuffle(p_star.begin(), p_star.end(), rng);
    return p_star;
}

/**
 * @brief Generate diagonal matrix D* with random +1 or -1 values
 */
template<typename RNG>
inline Matrix generate_d_star(size_t num_params, RNG& rng) {
    std::uniform_int_distribution<int> dist(0, 1);
    Vector diag_vals(num_params);
    for (size_t i = 0; i < num_params; ++i) {
        diag_vals[i] = dist(rng) ? 1.0 : -1.0;
    }
    return diag(diag_vals);
}

/**
 * @brief Compute the sampling matrix B* for a trajectory
 *
 * B* = x* + (delta/2) * ((2*B*G*P*^T - J)*D* + J)
 */
inline Matrix compute_b_star(
    const Matrix& J,
    const Vector& x_star,
    double delta,
    const Matrix& B,
    const Matrix& G,
    const Matrix& P_star,
    const Matrix& D_star
) {
    // element_a = J[0, :] * x_star (broadcast first row of J by x_star)
    Matrix element_a = J;
    for (size_t j = 0; j < element_a[0].size(); ++j) {
        for (size_t i = 0; i < element_a.size(); ++i) {
            element_a[i][j] = J[i][j] * x_star[j];
        }
    }

    // element_b = (G @ P_star).T
    Matrix GP = matmul(G, P_star);
    Matrix element_b = transpose(GP);

    // element_c = 2.0 * B @ element_b
    Matrix element_c = matmul(B, element_b);
    element_c = scalar_mult(2.0, element_c);

    // element_d = (element_c - J) @ D_star
    Matrix temp = matrix_subtract(element_c, J);
    Matrix element_d = matmul(temp, D_star);

    // b_star = element_a + (delta / 2.0) * (element_d + J)
    Matrix inner = matrix_add(element_d, J);
    inner = scalar_mult(delta / 2.0, inner);
    Matrix b_star = matrix_add(element_a, inner);

    return b_star;
}

/**
 * @brief Generate a single Morris trajectory
 */
template<typename RNG>
inline Matrix generate_trajectory(
    const Matrix& group_membership,
    int num_levels,
    RNG& rng
) {
    size_t num_params = group_membership.size();
    size_t num_groups = group_membership[0].size();

    double delta = compute_delta(num_levels);

    // Matrix B - lower triangular (g+1) x g
    Matrix B = tril(num_groups + 1, num_groups);

    // Matrix P* - random permutation
    Matrix P_star = generate_p_star(num_groups, rng);

    // Matrix J - ones
    Matrix J = ones(num_groups + 1, num_params);

    // Matrix D* - diagonal with random +/-1
    Matrix D_star = generate_d_star(num_params, rng);

    // Vector x* - random initial position
    Vector x_star = generate_x_star(num_params, num_levels, rng);

    // Compute B*
    Matrix B_star = compute_b_star(J, x_star, delta, B, group_membership, P_star, D_star);

    return B_star;
}

/**
 * @brief Generate N Morris trajectories
 *
 * @param num_params Number of parameters
 * @param num_groups Number of groups (or num_params if no grouping)
 * @param N Number of trajectories to generate
 * @param num_levels Number of grid levels (should be even)
 * @param group_membership Group membership matrix (num_params x num_groups)
 * @param seed Random seed
 * @return Matrix of size (N * (num_groups + 1)) x num_params
 */
inline Matrix sample_morris(
    size_t num_params,
    size_t num_groups,
    size_t N,
    int num_levels = 4,
    const Matrix& group_membership = Matrix(),
    unsigned int seed = std::random_device{}()
) {
    std::mt19937 rng(seed);

    // If no group membership provided, assume each parameter is its own group
    Matrix G = group_membership;
    if (G.empty()) {
        G = eye(num_params);
        num_groups = num_params;
    }

    // Validate group membership
    if (G.size() != num_params) {
        throw std::invalid_argument("Group membership rows must equal num_params");
    }

    // Warn if num_levels is odd
    if (num_levels % 2 != 0) {
        // In production code, you might want to use a logging system
        // For now, we'll just proceed with a warning in comments
    }

    // Generate trajectories in parallel
    std::vector<Matrix> trajectories(N);

    #pragma omp parallel
    {
        // Each thread needs its own RNG
        std::mt19937 thread_rng(seed + omp_get_thread_num());

        #pragma omp for
        for (size_t i = 0; i < N; ++i) {
            trajectories[i] = generate_trajectory(G, num_levels, thread_rng);
        }
    }

    // Flatten trajectories into single matrix
    size_t trajectory_size = num_groups + 1;
    Matrix sample_matrix = zeros(N * trajectory_size, num_params);

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < trajectory_size; ++j) {
            sample_matrix[i * trajectory_size + j] = trajectories[i][j];
        }
    }

    return sample_matrix;
}

// ============================================================================
// Local Optimization for Trajectory Selection
// ============================================================================

/**
 * @brief Compute distance matrix between all trajectories
 */
inline Matrix compute_distance_matrix(
    const Matrix& input_sample,
    size_t N,
    size_t num_params,
    size_t num_groups
) {
    (void)num_params;  // Unused but kept for API consistency
    size_t trajectory_size = num_groups + 1;
    Matrix distance_matrix = zeros(N, N);

    // Extract trajectories
    std::vector<Matrix> trajectories(N);
    for (size_t i = 0; i < N; ++i) {
        trajectories[i] = Matrix(trajectory_size);
        for (size_t j = 0; j < trajectory_size; ++j) {
            trajectories[i][j] = input_sample[i * trajectory_size + j];
        }
    }

    // Compute pairwise distances (parallelized)
    #pragma omp parallel for collapse(2) if(N > 10)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = i + 1; j < N; ++j) {
            double dist = trajectory_distance(trajectories[i], trajectories[j]);
            distance_matrix[i][j] = dist;
            distance_matrix[j][i] = dist;
        }
    }

    return distance_matrix;
}

/**
 * @brief Calculate sum of distances for a set of trajectory indices
 */
inline double sum_distances(
    const std::vector<size_t>& indices,
    const Matrix& distance_matrix
) {
    if (indices.size() < 2) return 0.0;

    double total = 0.0;

    // Sum all pairwise distances
    for (size_t i = 0; i < indices.size(); ++i) {
        for (size_t j = i + 1; j < indices.size(); ++j) {
            size_t idx1 = indices[i];
            size_t idx2 = indices[j];
            double dist = distance_matrix[idx1][idx2];
            total += dist * dist;
        }
    }

    return std::sqrt(total);
}

/**
 * @brief Find index with maximum distance sum
 */
inline size_t get_max_sum_ind(
    const std::vector<std::vector<size_t>>& indices_list,
    const Vector& distances
) {
    if (indices_list.size() != distances.size()) {
        throw std::invalid_argument("indices_list and distances must have same size");
    }

    auto max_it = std::max_element(distances.begin(), distances.end());
    return std::distance(distances.begin(), max_it);
}

/**
 * @brief Add new indices to existing combination
 */
inline std::vector<std::vector<size_t>> add_indices(
    const std::vector<size_t>& indices,
    size_t N
) {
    std::set<size_t> existing(indices.begin(), indices.end());
    std::vector<std::vector<size_t>> result;

    for (size_t i = 0; i < N; ++i) {
        if (existing.find(i) == existing.end()) {
            std::vector<size_t> new_indices = indices;
            new_indices.push_back(i);
            result.push_back(new_indices);
        }
    }

    return result;
}

/**
 * @brief Local optimization to find k optimal trajectories from N samples
 *
 * Based on Ruano et al. (2012) algorithm
 */
inline std::vector<size_t> find_local_maximum(
    const Matrix& distance_matrix,
    size_t N,
    size_t k_choices
) {
    if (k_choices >= N || k_choices < 2) {
        throw std::invalid_argument("k_choices must be between 2 and N-1");
    }

    std::vector<std::vector<size_t>> tot_indices_list;
    Vector tot_max_array(k_choices - 1);

    // Loop over k_choices, i starts at 1
    for (size_t i = 1; i < k_choices; ++i) {
        std::vector<std::vector<size_t>> indices_list(N);
        Vector row_maxima_i(N, 0.0);

        // For each row, find best i trajectories
        #pragma omp parallel for if(N > 10)
        for (size_t row_nr = 0; row_nr < N; ++row_nr) {
            // Get indices of i largest distances in this row
            std::vector<std::pair<double, size_t>> row_with_idx;
            for (size_t col = 0; col < N; ++col) {
                if (col != row_nr) {
                    row_with_idx.push_back({distance_matrix[row_nr][col], col});
                }
            }

            std::partial_sort(row_with_idx.begin(),
                            row_with_idx.begin() + i,
                            row_with_idx.end(),
                            [](const auto& a, const auto& b) { return a.first > b.first; });

            std::vector<size_t> indices;
            for (size_t j = 0; j < i; ++j) {
                indices.push_back(row_with_idx[j].second);
            }
            indices.push_back(row_nr);

            indices_list[row_nr] = indices;
            row_maxima_i[row_nr] = sum_distances(indices, distance_matrix);
        }

        // Find the indices belonging to the maximum distance
        size_t i_max_idx = get_max_sum_ind(indices_list, row_maxima_i);
        std::vector<size_t> m_max_ind = indices_list[i_max_idx];

        // Loop 'm'
        size_t m = 1;
        while (m <= k_choices - i - 1) {
            auto m_ind = add_indices(m_max_ind, N);
            size_t len_m_ind = m_ind.size();

            Vector m_maxima(len_m_ind);

            #pragma omp parallel for if(len_m_ind > 10)
            for (size_t n = 0; n < len_m_ind; ++n) {
                m_maxima[n] = sum_distances(m_ind[n], distance_matrix);
            }

            size_t m_max_idx = get_max_sum_ind(m_ind, m_maxima);
            m_max_ind = m_ind[m_max_idx];

            ++m;
        }

        tot_indices_list.push_back(m_max_ind);
        tot_max_array[i - 1] = sum_distances(m_max_ind, distance_matrix);
    }

    size_t tot_max_idx = get_max_sum_ind(tot_indices_list, tot_max_array);
    auto result = tot_indices_list[tot_max_idx];
    std::sort(result.begin(), result.end());

    return result;
}

/**
 * @brief Select optimal trajectories from a larger sample
 */
inline Matrix compute_optimized_trajectories(
    const Matrix& input_sample,
    size_t N,
    size_t num_params,
    size_t num_groups,
    size_t k_choices
) {
    // Compute distance matrix
    Matrix distance_matrix = compute_distance_matrix(input_sample, N, num_params, num_groups);

    // Find optimal trajectory indices
    std::vector<size_t> optimal_indices = find_local_maximum(distance_matrix, N, k_choices);

    // Extract optimal trajectories
    size_t trajectory_size = num_groups + 1;
    Matrix output = zeros(k_choices * trajectory_size, num_params);

    for (size_t i = 0; i < k_choices; ++i) {
        size_t src_idx = optimal_indices[i];
        for (size_t j = 0; j < trajectory_size; ++j) {
            output[i * trajectory_size + j] = input_sample[src_idx * trajectory_size + j];
        }
    }

    return output;
}

// ============================================================================
// Morris Analysis Functions
// ============================================================================

/**
 * @brief Result structure for Morris analysis
 */
struct MorrisResult {
    Vector mu;           ///< Mean elementary effect
    Vector mu_star;      ///< Mean absolute elementary effect
    Vector sigma;        ///< Standard deviation of elementary effect
    Vector mu_star_conf; ///< Confidence interval for mu_star
};

/**
 * @brief Compute elementary effects from model inputs and outputs
 */
inline Matrix compute_elementary_effects(
    const Matrix& X,
    const Vector& Y,
    size_t num_params,
    size_t trajectory_size,
    double delta
) {
    size_t num_trajectories = Y.size() / trajectory_size;

    // Reshape into trajectories
    std::vector<Matrix> input_trajectories(num_trajectories);
    std::vector<Vector> output_trajectories(num_trajectories);

    for (size_t i = 0; i < num_trajectories; ++i) {
        input_trajectories[i] = Matrix(trajectory_size);
        output_trajectories[i] = Vector(trajectory_size);

        for (size_t j = 0; j < trajectory_size; ++j) {
            size_t idx = i * trajectory_size + j;
            input_trajectories[i][j] = X[idx];
            output_trajectories[i][j] = Y[idx];
        }
    }

    // Compute elementary effects
    Matrix elementary_effects = zeros(num_params, num_trajectories);

    #pragma omp parallel for if(num_trajectories > 4)
    for (size_t traj = 0; traj < num_trajectories; ++traj) {
        // For each step in trajectory, find which parameter changed
        for (size_t step = 1; step < trajectory_size; ++step) {
            // Find which parameter changed
            for (size_t param = 0; param < num_params; ++param) {
                double diff = std::abs(input_trajectories[traj][step][param] -
                                      input_trajectories[traj][step-1][param]);

                if (diff > 1e-10) {  // This parameter changed
                    double output_diff = output_trajectories[traj][step] -
                                        output_trajectories[traj][step-1];
                    elementary_effects[param][traj] += output_diff / delta;
                    break;
                }
            }
        }
    }

    return elementary_effects;
}

/**
 * @brief Compute mu_star confidence intervals using bootstrap
 */
template<typename RNG>
inline Vector compute_mu_star_confidence(
    const Matrix& elementary_effects,
    size_t num_resamples,
    double conf_level,
    RNG& rng
) {
    size_t num_params = elementary_effects.size();
    Vector mu_star_conf(num_params);

    #pragma omp parallel
    {
        // Each thread needs its own RNG
        std::mt19937 thread_rng(rng() + omp_get_thread_num());

        #pragma omp for
        for (size_t param = 0; param < num_params; ++param) {
            const Vector& ee = elementary_effects[param];
            size_t n = ee.size();

            std::uniform_int_distribution<size_t> dist(0, n - 1);

            // Bootstrap resampling
            Vector mu_star_resampled(num_resamples);
            for (size_t resample = 0; resample < num_resamples; ++resample) {
                Vector resampled_ee(n);
                for (size_t i = 0; i < n; ++i) {
                    resampled_ee[i] = ee[dist(thread_rng)];
                }

                // Compute mu_star for this resample
                double sum_abs = 0.0;
                for (double val : resampled_ee) {
                    sum_abs += std::abs(val);
                }
                mu_star_resampled[resample] = sum_abs / n;
            }

            // Compute confidence interval
            double std = stddev(mu_star_resampled, 1);
            mu_star_conf[param] = norm_ppf(0.5 + conf_level / 2.0) * std;
        }
    }

    return mu_star_conf;
}

/**
 * @brief Perform Morris sensitivity analysis
 *
 * @param X Model inputs (same as used for sampling)
 * @param Y Model outputs
 * @param num_params Number of parameters
 * @param num_groups Number of groups
 * @param num_levels Number of grid levels used in sampling
 * @param num_resamples Number of bootstrap resamples for confidence intervals
 * @param conf_level Confidence level (e.g., 0.95 for 95%)
 * @param seed Random seed for bootstrap
 * @return MorrisResult containing sensitivity indices
 */
inline MorrisResult analyze(
    const Matrix& X,
    const Vector& Y,
    size_t num_params,
    size_t num_groups,
    int num_levels = 4,
    size_t num_resamples = 100,
    double conf_level = 0.95,
    unsigned int seed = std::random_device{}()
) {
    if (X.empty() || Y.empty()) {
        throw std::invalid_argument("Input matrices cannot be empty");
    }

    if (X.size() != Y.size()) {
        throw std::invalid_argument("X and Y must have same number of rows");
    }

    double delta = compute_delta(num_levels);
    size_t trajectory_size = num_groups + 1;

    // Compute elementary effects
    Matrix elementary_effects = compute_elementary_effects(X, Y, num_params, trajectory_size, delta);

    // Compute sensitivity indices
    MorrisResult result;
    result.mu.resize(num_params);
    result.mu_star.resize(num_params);
    result.sigma.resize(num_params);

    #pragma omp parallel for if(num_params > 4)
    for (size_t i = 0; i < num_params; ++i) {
        const Vector& ee = elementary_effects[i];

        // mu - mean
        result.mu[i] = mean(ee);

        // mu_star - mean of absolute values
        Vector abs_ee(ee.size());
        for (size_t j = 0; j < ee.size(); ++j) {
            abs_ee[j] = std::abs(ee[j]);
        }
        result.mu_star[i] = mean(abs_ee);

        // sigma - standard deviation
        result.sigma[i] = stddev(ee, 1);
    }

    // Compute confidence intervals
    std::mt19937 rng(seed);
    result.mu_star_conf = compute_mu_star_confidence(elementary_effects, num_resamples, conf_level, rng);

    return result;
}

// ============================================================================
// Combined Sampling and Analysis Interface
// ============================================================================

/**
 * @brief Problem definition structure
 */
struct Problem {
    size_t num_vars;                    ///< Number of variables/parameters
    std::vector<std::pair<double, double>> bounds;  ///< Parameter bounds
    Matrix group_membership;            ///< Optional group membership matrix
};

/**
 * @brief Scale samples from [0,1] to actual parameter bounds
 */
inline Matrix scale_samples(const Matrix& samples, const Problem& problem) {
    Matrix scaled = samples;

    for (size_t i = 0; i < scaled.size(); ++i) {
        for (size_t j = 0; j < scaled[i].size(); ++j) {
            double lower = problem.bounds[j].first;
            double upper = problem.bounds[j].second;
            scaled[i][j] = lower + scaled[i][j] * (upper - lower);
        }
    }

    return scaled;
}

/**
 * @brief Complete Morris sampling workflow with optional optimization
 *
 * @param problem Problem definition
 * @param N Number of trajectories to generate
 * @param num_levels Number of grid levels (should be even)
 * @param optimal_trajectories Number of optimal trajectories to select (0 = no optimization)
 * @param seed Random seed
 * @return Scaled sample matrix ready for model evaluation
 */
inline Matrix sample(
    const Problem& problem,
    size_t N,
    int num_levels = 4,
    size_t optimal_trajectories = 0,
    unsigned int seed = std::random_device{}()
) {
    size_t num_groups = problem.group_membership.empty() ?
                        problem.num_vars :
                        problem.group_membership[0].size();

    // Generate initial trajectories
    Matrix sample_matrix = sample_morris(
        problem.num_vars,
        num_groups,
        N,
        num_levels,
        problem.group_membership,
        seed
    );

    // Apply local optimization if requested
    if (optimal_trajectories > 0 && optimal_trajectories < N) {
        sample_matrix = compute_optimized_trajectories(
            sample_matrix,
            N,
            problem.num_vars,
            num_groups,
            optimal_trajectories
        );
    }

    // Scale to actual parameter bounds
    return scale_samples(sample_matrix, problem);
}

} // namespace morris

#endif // MORRIS_HPP
