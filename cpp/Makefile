# Makefile for Morris Screening C++ Example

CXX = g++
CXXFLAGS = -std=c++14 -Wall -Wextra -O3
OPENMP_FLAGS = -fopenmp
TARGET = morris_example
SOURCE = example_morris.cpp
HEADER = include/morris.hpp

# Default target with OpenMP
all: $(TARGET)

# Build with OpenMP (parallel)
$(TARGET): $(SOURCE) $(HEADER)
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) $(SOURCE) -o $(TARGET)
	@echo "Built with OpenMP support (parallel execution)"

# Build without OpenMP (serial)
serial: $(SOURCE) $(HEADER)
	$(CXX) $(CXXFLAGS) $(SOURCE) -o $(TARGET)
	@echo "Built without OpenMP (serial execution)"

# Run the example
run: $(TARGET)
	./$(TARGET)

# Clean build artifacts
clean:
	rm -f $(TARGET)

# Build with debug symbols
debug: CXXFLAGS = -std=c++14 -Wall -Wextra -g
debug: $(TARGET)

# Show compiler and OpenMP info
info:
	@echo "Compiler: $(CXX)"
	@$(CXX) --version
	@echo ""
	@echo "Checking OpenMP support:"
	@echo '#include <omp.h>' | $(CXX) -fopenmp -x c++ -E - > /dev/null 2>&1 && \
		echo "  ✓ OpenMP is available" || echo "  ✗ OpenMP is NOT available"

.PHONY: all serial run clean debug info
