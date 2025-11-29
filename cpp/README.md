# Morris Screening Method - C++ Header-Only Library

A high-performance, header-only C++14 implementation of the Morris screening method for global sensitivity analysis, with OpenMP parallelization.

## Features

- **Header-only**: Just include `morris.hpp` - no compilation or linking required
- **C++14 compatible**: Works with older compilers, no modern C++ features required
- **Minimal dependencies**: Only uses C++ standard library and OpenMP
- **OpenMP parallelized**: Automatic parallelization of key computational bottlenecks
- **Complete implementation**:
  - Morris trajectory sampling
  - Local optimization for trajectory selection (Ruano et al., 2012)
  - Elementary effects calculation
  - Sensitivity indices (μ, μ*, σ)
  - Bootstrap confidence intervals
  - Support for parameter grouping

## Quick Start

### Basic Usage

```cpp
#include "include/morris.hpp"

// 1. Define your problem
morris::Problem problem;
problem.num_vars = 3;
problem.bounds = {
    {0.0, 10.0},  // x1 bounds
    {-5.0, 5.0},  // x2 bounds
    {0.0, 1.0}    // x3 bounds
};

// 2. Generate sample points
morris::Matrix X = morris::sample(
    problem,
    100,   // N trajectories
    4,     // num_levels
    10     // select 10 optimal trajectories
);

// 3. Evaluate your model
morris::Vector Y(X.size());
for (size_t i = 0; i < X.size(); ++i) {
    Y[i] = your_model_function(X[i]);
}

// 4. Analyze results
morris::MorrisResult result = morris::analyze(
    X, Y,
    problem.num_vars,
    problem.num_vars,  // num_groups (= num_vars if no grouping)
    4                   // num_levels
);

// 5. Access sensitivity indices
for (size_t i = 0; i < result.mu_star.size(); ++i) {
    std::cout << "Parameter " << i << ": mu* = " << result.mu_star[i] << '\n';
}
```

## Compilation

### Compile the Example

#### With OpenMP (recommended for parallelization):

```bash
# GCC/G++
g++ -std=c++14 -fopenmp -O3 example_morris.cpp -o morris_example

# Clang++
clang++ -std=c++14 -fopenmp -O3 example_morris.cpp -o morris_example
```

#### Without OpenMP (serial execution):

```bash
g++ -std=c++14 -O3 example_morris.cpp -o morris_example
```

### Using the Makefile

```bash
make          # Build with OpenMP
make clean    # Clean build artifacts
make run      # Build and run example
```

### Run the Example

```bash
./morris_example
```

## API Reference

### Core Data Structures

#### `morris::Problem`
```cpp
struct Problem {
    size_t num_vars;                                    // Number of parameters
    std::vector<std::pair<double, double>> bounds;      // Parameter bounds
    Matrix group_membership;                            // Optional: group membership matrix
};
```

#### `morris::MorrisResult`
```cpp
struct MorrisResult {
    Vector mu;           // Mean elementary effect
    Vector mu_star;      // Mean absolute elementary effect
    Vector mu_star_conf; // Confidence interval for mu_star
    Vector sigma;        // Standard deviation (interactions/non-linearity)
};
```

### Main Functions

#### Sampling

```cpp
Matrix sample(
    const Problem& problem,
    size_t N,                      // Number of trajectories
    int num_levels = 4,            // Grid levels (should be even)
    size_t optimal_trajectories = 0, // 0 = no optimization, >0 = select k best
    unsigned int seed = random
);
```

Generates Morris sample points scaled to parameter bounds.

#### Analysis

```cpp
MorrisResult analyze(
    const Matrix& X,               // Model inputs
    const Vector& Y,               // Model outputs
    size_t num_params,             // Number of parameters
    size_t num_groups,             // Number of groups
    int num_levels = 4,            // Grid levels (must match sampling)
    size_t num_resamples = 100,    // Bootstrap resamples
    double conf_level = 0.95,      // Confidence level
    unsigned int seed = random
);
```

Computes Morris sensitivity indices.

### Interpreting Results

- **μ* (mu_star)**: Overall importance of parameter
  - Higher values indicate greater influence on model output
  - Use for ranking parameters

- **σ (sigma)**: Interactions and non-linearity
  - High σ with high μ*: non-linear effects or interactions
  - High σ with low μ: interactions with other parameters

- **μ (mu)**: Directional effect
  - Compare with μ*: if μ ≈ 0 but μ* is high, effects cancel out (different signs)

- **mu_star_conf**: Confidence interval width
  - Indicates uncertainty in μ* estimate
  - Narrow intervals = more reliable estimates

## Advanced Usage

### Parameter Grouping

Group related parameters to reduce dimensionality:

```cpp
morris::Problem problem;
problem.num_vars = 6;
problem.bounds = /* ... 6 bounds ... */;

// Define 3 groups with 2 parameters each
problem.group_membership = morris::zeros(6, 3);
problem.group_membership[0][0] = 1.0;  // param 0 -> group 0
problem.group_membership[1][0] = 1.0;  // param 1 -> group 0
problem.group_membership[2][1] = 1.0;  // param 2 -> group 1
// ... etc
```

### Custom Random Seed

For reproducible results:

```cpp
unsigned int seed = 12345;
auto X = morris::sample(problem, N, num_levels, k_optimal, seed);
auto result = morris::analyze(X, Y, num_params, num_groups, num_levels, 100, 0.95, seed);
```

### Manual Workflow (Advanced)

For more control over the process:

```cpp
// 1. Generate trajectories without scaling
Matrix unscaled = morris::sample_morris(
    num_params, num_groups, N, num_levels, group_membership, seed
);

// 2. Apply local optimization
Matrix optimized = morris::compute_optimized_trajectories(
    unscaled, N, num_params, num_groups, k_choices
);

// 3. Scale to bounds
Matrix scaled = morris::scale_samples(optimized, problem);

// 4. ... evaluate model, then analyze
```

## Performance Tips

1. **Use OpenMP**: Compile with `-fopenmp` for parallel execution
   - Trajectory generation parallelized over trajectories
   - Distance matrix computation parallelized
   - Elementary effects calculation parallelized
   - Bootstrap resampling parallelized

2. **Optimize trajectory selection**:
   - For exploratory analysis: N=100-500, optimal_trajectories=10-20
   - For final analysis: N=1000+, optimal_trajectories=50-100
   - Local optimization is much faster than brute force

3. **Choose num_levels wisely**:
   - Default: 4 (fast, usually sufficient)
   - Higher values (6, 8): more precision but slower
   - Always use even numbers

4. **Balance sample size vs optimization**:
   - Generating 1000 trajectories and selecting 100 optimal > generating 100 trajectories
   - Local optimization scales well to large N

## Theoretical Background

### Morris Method

The Morris method (Morris, 1991) is a one-at-a-time (OAT) screening method that:
- Efficiently identifies important parameters
- Requires fewer model evaluations than variance-based methods
- Provides qualitative sensitivity measures

### Optimized Trajectories

- **Campolongo et al. (2007)**: Select trajectories that maximize space-filling
- **Ruano et al. (2012)**: Local optimization for faster trajectory selection

### Sample Size

Number of model evaluations = N * (num_groups + 1) where:
- N = number of trajectories (or optimal_trajectories if optimization used)
- num_groups = number of groups (or num_params if no grouping)

Example: 10 optimal trajectories from 100, 5 parameters:
- Initial: 100 * 6 = 600 evaluations to generate samples
- Final: 10 * 6 = 60 evaluations for analysis

## References

1. **Morris, M.D. (1991)**. "Factorial Sampling Plans for Preliminary Computational Experiments."
   *Technometrics*, 33(2), 161-174.

2. **Campolongo, F., Cariboni, J., & Saltelli, A. (2007)**. "An effective screening design for
   sensitivity analysis of large models." *Environmental Modelling & Software*, 22(10), 1509-1518.

3. **Ruano, M.V., Ribes, J., Seco, A., Ferrer, J. (2012)**. "An improved sampling strategy based
   on trajectory design for application of the Morris method to systems with many input factors."
   *Environmental Modelling & Software*, 37, 103-109.

## Integration Guide

### Integrating into Your Project

Since this is a header-only library, integration is simple:

1. Copy `include/morris.hpp` to your project
2. Include it in your code: `#include "path/to/morris.hpp"`
3. Compile with C++14: `-std=c++14`
4. Optional: Enable OpenMP: `-fopenmp`

### Example Integration

```cpp
// your_project.cpp
#include "morris.hpp"
#include "your_model.hpp"

int main() {
    // Define sensitivity analysis problem
    morris::Problem problem;
    problem.num_vars = your_model::num_parameters();
    problem.bounds = your_model::get_parameter_bounds();

    // Generate sample
    auto X = morris::sample(problem, 100, 4, 10);

    // Evaluate your model
    morris::Vector Y(X.size());
    #pragma omp parallel for
    for (size_t i = 0; i < X.size(); ++i) {
        Y[i] = your_model::evaluate(X[i]);
    }

    // Analyze
    auto result = morris::analyze(X, Y, problem.num_vars, problem.num_vars, 4);

    // Use results to identify important parameters
    your_model::calibrate(result.mu_star);

    return 0;
}
```

## License

This implementation is derived from the Python SALib library and maintains compatibility with SALib's algorithms and test cases.

## Troubleshooting

### OpenMP not found
- Install OpenMP: `sudo apt-get install libomp-dev` (Ubuntu/Debian)
- Or compile without it (serial execution)

### Compilation errors with C++14
- Ensure compiler supports C++14: GCC 5+, Clang 3.4+, MSVC 2015+
- Check flag: `-std=c++14` (not `-std=c++11`)

### Results differ from Python SALib
- Check that num_levels matches between sampling and analysis
- Verify random seeds are the same
- Small numerical differences are expected due to different RNG implementations

## Contact

For issues, questions, or contributions related to this C++ port, please refer to the main SALib repository.
