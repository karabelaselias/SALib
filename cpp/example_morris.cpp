/**
 * @file example_morris.cpp
 * @brief Example demonstrating Morris screening method usage
 *
 * This example shows how to use the Morris screening library for
 * sensitivity analysis of a simple test function.
 */

#include "include/morris.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>

// Example test function: Ishigami function
// f(x1, x2, x3) = sin(x1) + a*sin^2(x2) + b*x3^4*sin(x1)
// This is a common benchmark function for sensitivity analysis
double ishigami(const morris::Vector& x, double a = 7.0, double b = 0.1) {
    if (x.size() < 3) {
        throw std::invalid_argument("Ishigami requires 3 parameters");
    }

    double x1 = x[0];
    double x2 = x[1];
    double x3 = x[2];

    return std::sin(x1) + a * std::sin(x2) * std::sin(x2) +
           b * std::pow(x3, 4) * std::sin(x1);
}

// Simple linear function for testing
double linear_function(const morris::Vector& x) {
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        sum += (i + 1) * x[i];  // Weight increases with index
    }
    return sum;
}

void print_results(const morris::MorrisResult& result, const std::vector<std::string>& param_names) {
    std::cout << "\n=== Morris Sensitivity Analysis Results ===\n\n";
    std::cout << std::setw(15) << "Parameter"
              << std::setw(15) << "mu"
              << std::setw(15) << "mu_star"
              << std::setw(15) << "sigma"
              << std::setw(15) << "mu*_conf\n";
    std::cout << std::string(75, '-') << '\n';

    for (size_t i = 0; i < result.mu.size(); ++i) {
        std::cout << std::setw(15) << param_names[i]
                  << std::setw(15) << std::fixed << std::setprecision(6) << result.mu[i]
                  << std::setw(15) << result.mu_star[i]
                  << std::setw(15) << result.sigma[i]
                  << std::setw(15) << result.mu_star_conf[i]
                  << '\n';
    }
    std::cout << '\n';
}

void example_simple() {
    std::cout << "\n╔════════════════════════════════════════╗\n";
    std::cout << "║  Example 1: Simple Linear Function     ║\n";
    std::cout << "╚════════════════════════════════════════╝\n";

    // Define problem
    morris::Problem problem;
    problem.num_vars = 5;
    problem.bounds = {
        {0.0, 1.0},  // x1
        {0.0, 1.0},  // x2
        {0.0, 1.0},  // x3
        {0.0, 1.0},  // x4
        {0.0, 1.0}   // x5
    };

    std::vector<std::string> param_names = {"x1", "x2", "x3", "x4", "x5"};

    // Morris sampling parameters
    size_t N = 100;                    // Number of trajectories
    int num_levels = 4;                // Grid levels
    size_t optimal_trajectories = 10;  // Select 10 best trajectories
    unsigned int seed = 12345;

    std::cout << "Generating " << N << " trajectories...\n";
    std::cout << "Selecting " << optimal_trajectories << " optimal trajectories...\n";

    // Generate sample
    morris::Matrix X = morris::sample(problem, N, num_levels, optimal_trajectories, seed);

    std::cout << "Generated " << X.size() << " sample points\n";

    // Evaluate model
    std::cout << "Evaluating model...\n";
    morris::Vector Y(X.size());

    #pragma omp parallel for
    for (size_t i = 0; i < X.size(); ++i) {
        Y[i] = linear_function(X[i]);
    }

    // Perform Morris analysis
    std::cout << "Performing Morris analysis...\n";
    size_t num_groups = problem.num_vars;  // No grouping
    morris::MorrisResult result = morris::analyze(
        X, Y, problem.num_vars, num_groups, num_levels,
        100,    // num_resamples for confidence intervals
        0.95,   // confidence level
        seed
    );

    // Print results
    print_results(result, param_names);

    std::cout << "Expected: Higher weights (x5 > x4 > x3 > x2 > x1) should have higher mu_star\n";
}

void example_ishigami() {
    std::cout << "\n╔════════════════════════════════════════╗\n";
    std::cout << "║  Example 2: Ishigami Function          ║\n";
    std::cout << "╚════════════════════════════════════════╝\n";

    // Define problem - Ishigami function has 3 parameters on [-π, π]
    morris::Problem problem;
    problem.num_vars = 3;
    const double pi = 3.14159265358979323846;
    problem.bounds = {
        {-pi, pi},  // x1
        {-pi, pi},  // x2
        {-pi, pi}   // x3
    };

    std::vector<std::string> param_names = {"x1", "x2", "x3"};

    // Morris sampling parameters
    size_t N = 200;                    // Number of trajectories
    int num_levels = 4;                // Grid levels
    size_t optimal_trajectories = 20;  // Select 20 best trajectories
    unsigned int seed = 42;

    std::cout << "Generating " << N << " trajectories...\n";
    std::cout << "Selecting " << optimal_trajectories << " optimal trajectories...\n";

    // Generate sample
    morris::Matrix X = morris::sample(problem, N, num_levels, optimal_trajectories, seed);

    std::cout << "Generated " << X.size() << " sample points\n";

    // Evaluate Ishigami function
    std::cout << "Evaluating Ishigami function...\n";
    morris::Vector Y(X.size());

    #pragma omp parallel for
    for (size_t i = 0; i < X.size(); ++i) {
        Y[i] = ishigami(X[i]);
    }

    // Perform Morris analysis
    std::cout << "Performing Morris analysis...\n";
    size_t num_groups = problem.num_vars;
    morris::MorrisResult result = morris::analyze(
        X, Y, problem.num_vars, num_groups, num_levels,
        100,    // num_resamples
        0.95,   // confidence level
        seed
    );

    // Print results
    print_results(result, param_names);

    std::cout << "Note: For Ishigami function:\n";
    std::cout << "  - x1 and x2 should have high mu_star (main effects)\n";
    std::cout << "  - x3 has no main effect but high interaction (high sigma, low mu_star)\n";
}

void example_with_groups() {
    std::cout << "\n╔════════════════════════════════════════╗\n";
    std::cout << "║  Example 3: Morris with Groups         ║\n";
    std::cout << "╚════════════════════════════════════════╝\n";

    // Define problem with 6 parameters in 3 groups
    morris::Problem problem;
    problem.num_vars = 6;
    problem.bounds = {
        {0.0, 1.0}, {0.0, 1.0},  // Group 1
        {0.0, 1.0}, {0.0, 1.0},  // Group 2
        {0.0, 1.0}, {0.0, 1.0}   // Group 3
    };

    // Define group membership matrix (6 params x 3 groups)
    // Group 1: params 0,1
    // Group 2: params 2,3
    // Group 3: params 4,5
    problem.group_membership = morris::zeros(6, 3);
    problem.group_membership[0][0] = 1.0;  // param 0 -> group 0
    problem.group_membership[1][0] = 1.0;  // param 1 -> group 0
    problem.group_membership[2][1] = 1.0;  // param 2 -> group 1
    problem.group_membership[3][1] = 1.0;  // param 3 -> group 1
    problem.group_membership[4][2] = 1.0;  // param 4 -> group 2
    problem.group_membership[5][2] = 1.0;  // param 5 -> group 2

    std::vector<std::string> param_names = {"x1", "x2", "x3", "x4", "x5", "x6"};

    std::cout << "Parameters grouped as: [x1,x2], [x3,x4], [x5,x6]\n";

    size_t N = 50;
    int num_levels = 4;
    size_t optimal_trajectories = 10;
    unsigned int seed = 999;

    std::cout << "Generating " << N << " trajectories...\n";

    // Generate sample
    morris::Matrix X = morris::sample(problem, N, num_levels, optimal_trajectories, seed);

    std::cout << "Generated " << X.size() << " sample points\n";

    // Simple model: sum all parameters with increasing weights
    std::cout << "Evaluating model...\n";
    morris::Vector Y(X.size());

    #pragma omp parallel for
    for (size_t i = 0; i < X.size(); ++i) {
        Y[i] = linear_function(X[i]);
    }

    // Perform Morris analysis
    std::cout << "Performing Morris analysis...\n";
    size_t num_groups = 3;  // We have 3 groups
    morris::MorrisResult result = morris::analyze(
        X, Y, problem.num_vars, num_groups, num_levels, 100, 0.95, seed
    );

    // Print results
    print_results(result, param_names);

    std::cout << "Note: Results show sensitivity averaged within each group\n";
}

int main(int argc, char* argv[]) {
    (void)argc;  // Unused
    (void)argv;  // Unused
    std::cout << "╔════════════════════════════════════════════════════╗\n";
    std::cout << "║   Morris Screening Method - C++ Implementation     ║\n";
    std::cout << "║   Header-only library with OpenMP parallelization  ║\n";
    std::cout << "╚════════════════════════════════════════════════════╝\n";

    #ifdef _OPENMP
    std::cout << "\nOpenMP is ENABLED (parallel execution)\n";
    std::cout << "Maximum threads available: " << omp_get_max_threads() << '\n';
    #else
    std::cout << "\nOpenMP is DISABLED (serial execution)\n";
    std::cout << "Compile with -fopenmp to enable parallelization\n";
    #endif

    try {
        // Run examples
        example_simple();
        example_ishigami();
        example_with_groups();

        std::cout << "\n✓ All examples completed successfully!\n\n";

    } catch (const std::exception& e) {
        std::cerr << "\n✗ Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
