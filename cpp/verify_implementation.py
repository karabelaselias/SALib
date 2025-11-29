#!/usr/bin/env python3
"""
Verify C++ Morris implementation matches Python SALib
"""
import numpy as np
import sys
sys.path.insert(0, '/home/user/SALib/src')

from SALib.sample import morris as morris_sample
from SALib.analyze import morris as morris_analyze

# Test 1: Simple function without groups
print("=" * 60)
print("Test 1: Linear function (no groups)")
print("=" * 60)

problem = {
    'num_vars': 5,
    'names': ['x1', 'x2', 'x3', 'x4', 'x5'],
    'bounds': [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
}

# Use same seed as C++ example
X = morris_sample.sample(problem, N=100, num_levels=4,
                         optimal_trajectories=10, seed=12345)
Y = np.sum(X * np.arange(1, 6), axis=1)

Si = morris_analyze.analyze(problem, X, Y, num_levels=4, seed=12345)

print("\nPython SALib results:")
print(f"{'Parameter':<15} {'mu':<15} {'mu_star':<15} {'sigma':<15}")
print("-" * 60)
for i, name in enumerate(problem['names']):
    print(f"{name:<15} {Si['mu'][i]:<15.6f} {Si['mu_star'][i]:<15.6f} {Si['sigma'][i]:<15.6f}")

print("\nExpected: mu_star should increase (x5 > x4 > x3 > x2 > x1)")

# Test 2: Ishigami function
print("\n" + "=" * 60)
print("Test 2: Ishigami function")
print("=" * 60)

problem_ishi = {
    'num_vars': 3,
    'names': ['x1', 'x2', 'x3'],
    'bounds': [[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]]
}

X = morris_sample.sample(problem_ishi, N=200, num_levels=4,
                         optimal_trajectories=20, seed=42)

# Ishigami function
def ishigami(x):
    a = 7.0
    b = 0.1
    return np.sin(x[:, 0]) + a * np.sin(x[:, 1])**2 + b * x[:, 2]**4 * np.sin(x[:, 0])

Y = ishigami(X)

Si = morris_analyze.analyze(problem_ishi, X, Y, num_levels=4, seed=42)

print("\nPython SALib results:")
print(f"{'Parameter':<15} {'mu':<15} {'mu_star':<15} {'sigma':<15}")
print("-" * 60)
for i, name in enumerate(problem_ishi['names']):
    print(f"{name:<15} {Si['mu'][i]:<15.6f} {Si['mu_star'][i]:<15.6f} {Si['sigma'][i]:<15.6f}")

print("\nExpected: x1, x2 high mu_star; x3 low mu_star but high sigma")

# Test 3: With groups
print("\n" + "=" * 60)
print("Test 3: Parameter groups")
print("=" * 60)

problem_groups = {
    'num_vars': 6,
    'names': ['x1', 'x2', 'x3', 'x4', 'x5', 'x6'],
    'bounds': [[0, 1]] * 6,
    'groups': ['G1', 'G1', 'G2', 'G2', 'G3', 'G3']
}

X = morris_sample.sample(problem_groups, N=50, num_levels=4,
                         optimal_trajectories=10, seed=999)

Y = np.sum(X * np.arange(1, 7), axis=1)

Si = morris_analyze.analyze(problem_groups, X, Y, num_levels=4, seed=999)

print("\nPython SALib results:")
print(f"{'Parameter':<15} {'mu':<15} {'mu_star':<15} {'sigma':<15}")
print("-" * 60)
for i, name in enumerate(problem_groups['names']):
    print(f"{name:<15} {Si['mu'][i]:<15.6f} {Si['mu_star'][i]:<15.6f} {Si['sigma'][i]:<15.6f}")

print("\nExpected: Parameters in same group should have identical effects")
print("  Group 1 (x1, x2): same mu_star")
print("  Group 2 (x3, x4): same mu_star")
print("  Group 3 (x5, x6): same mu_star")
