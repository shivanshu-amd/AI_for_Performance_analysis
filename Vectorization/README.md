# Identify Missing vectorization opportunity in Benchmark for AOCC, LLVM, and GCC



LLM-based system for identifying vectorization candidate loops in C/C++ programs with multi-compiler verification.

## Overview

This project uses on-prem LLM (Large Language Model) analysis with multi-compiler verification to identify vectorization opportunities in C/C++ code. The system analyzes complete nested loop structures and provides expert-level vectorization assessment.

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
````

### Basic Usage

```bash
# Analyze all loops in a C/C++ file
python src/multi_compiler_verifier.py --file your_code.c

# Analyze only loops in a specific function
python src/multi_compiler_verifier.py --file your_code.c --function matrix_multiply

# Quick test with neural network kernels
python src/multi_compiler_verifier.py --file data/test_neural_network.c

# Analyze only specific function in complex file
python src/multi_compiler_verifier.py --file data/complex_template.c --function lstm_cell

# Test specific patterns
python src/multi_compiler_verifier.py --file data/test_image_processing.c
python src/multi_compiler_verifier.py --file data/test_tiled_loops.c
python src/multi_compiler_verifier.py --file data/test_nested_3level.c

# Use custom configuration
python src/multi_compiler_verifier.py --file your_code.c --config path/to/config.yaml
```

**Note**: The tool automatically:
- Extracts all loops with function names
- Analyzes nested loop structures completely
- Tests with all configured compilers
- Generates 3 output files with timestamps
- Can filter analysis to specific functions (useful for large files)

## Project Structure

```
AI Hackathon_final/
├── requirements.txt              # Python dependencies
├── Doc/
│   └── README.md                 # This file
│
├── config/                       # Configuration files
│   └── compiler_config.yaml     # Compiler and LLM settings
│
├── src/                          # Source code
│   ├── multi_compiler_verifier.py    # Main entry point (✅ Use this)
│   ├── llm_vectorization_checker.py  # On-prem LLM checker
│   └── loop_analyzer.py              # Loop extraction & analysis
│
└── data/                         # Test files
    ├── test_neural_network.c     # Neural network kernels
    ├── test_image_processing.c   # Image processing (blur, brightness)
    ├── test_tiled_loops.c        # 6-level nested tiled matrix multiplication
    ├── test_nested_3level.c      # 3D stencil computation
    └── complex_template.c        # Comprehensive multi-domain test suite
                                  # (32 functions: matrix, weather, NN, 
                                  #  image processing, stencils, encoders)
```

## Features

### LLM Verification

- **Expert Analysis**: On-prem LLM acts as compiler optimization engineer
- **Multi-Compiler Support**: Tests with GCC, Clang, AOCC, Intel compilers simultaneously
- **Complete Nested Context**: Analyzes full nested loop structures (not just innermost)
- **Smart Function Detection**: Automatically identifies function names (no more "unknown")
- **Function Filtering**: Focus analysis on specific functions with `--function` flag
- **Table Format Output**: Clean, organized loop reporting with file/function/loop columns
- **Auto-Generated Reports**: 3 output files per analysis (loops, detailed report, CSV)
- **On-Prem LLM Only**: Privacy-preserving, secure analysis

### Test Files Coverage

**`test_neural_network.c`** - Neural network kernels:
- Dense layer forward pass (2-level nested loop)
- ReLU activation (single loop)
- 2D convolution (4-level nested loop)

**`test_image_processing.c`** - Image processing operations:
- Horizontal blur (2-level nested loop)
- Brightness adjustment (single loop)

**`test_tiled_loops.c`** - Cache-optimized algorithms:
- Tiled matrix multiplication (6-level nested loop)
- Tests compiler's ability to vectorize complex nested structures

**`test_nested_3level.c`** - Multi-dimensional stencil:
- 3D 7-point stencil computation (3-level nested loop)
- Tests vectorization of 3D array operations

**`complex_template.c`** - Comprehensive multi-domain test suite with 32 functions:
- **Matrix Operations**: Multiplication, transpose, triangular solve
- **Weather Simulation**: Heat diffusion, pressure calculation, wind velocity
- **Neural Networks**: Dense layers, ReLU, LSTM cells, batch normalization
- **Image Processing**: Convolution, Gaussian blur, Sobel, median filter
- **Stencil Computations**: 3D/2D stencils, Jacobi iteration
- **Encoders/Decoders**: Run-length, Huffman, Base64, XOR cipher, S-box
- **Mixed Complexity**: Pointer chasing, reductions, histograms
- **Multi-Loop Functions**: Particle simulation, financial analysis, gradient descent, graph processing

## Configuration

Edit `config/compiler_config.yaml`:

```yaml
compilers:
  - name: gcc
    module_path: /path/to/modules
    module_name: gcc/15.release
    compiler_command: gcc
    flags: -O3 -ftree-vectorize -fopt-info-vec-all
    
  - name: clang
    module_path: /path/to/modules
    module_name: llvm/22_20Oct20
    compiler_command: clang
    flags: -O3 -Rpass=loop-vectorize -Rpass-missed=loop-vectorize
    
  - name: aocc
    module_path: /path/to/modules
    module_name: aocc/5.1_1920
    compiler_command: clang
    flags: -O3 -Rpass=loop-vectorize
    
  - name: intel
    module_path: /path/to/modules
    module_name: oneapi/2025.2
    compiler_command: icpx
    flags: -O3 -qopt-report=5 -qopt-report-phase=vec

# LLM Configuration (On-Prem Only)
llm:
  model: default
  api_key: your_api_key_here
```

## Output Files

When you run analysis, three files are automatically generated:

1. **Loops File** (`filename_loops_timestamp.txt` or `filename_functionname_loops_timestamp.txt`): 
   - Complete nested loop structures (outermost to innermost)
   - Loop IDs with function names and line numbers
   - Nesting level details

2. **Text Report** (`filename_timestamp.txt` or `filename_functionname_timestamp.txt`): 
   - Detailed LLM analysis with complete reasoning
   - Multi-compiler verification results
   - Summary statistics and agreement metrics

3. **CSV Report** (`filename_timestamp.csv` or `filename_functionname_timestamp.csv`): 
   - Structured data for further processing
   - LLM predictions vs compiler results
   - Ready for data analysis tools

4. **Compiler Optimization Reports** (one per compiler):
   - Without function filter: `filename_compilername.optrpt` (e.g., `test_neural_network_gcc.optrpt`)
   - With function filter: `filename_functionname_compilername.optrpt` (e.g., `test_neural_network_relu_activation_gcc.optrpt`)
   - Contains raw compiler vectorization reports
   - Generated for each configured compiler (gcc, clang, aocc, intel)

**Note**: When using `--function` flag, the function name is automatically included in all output filenames (including optimization reports) for easy identification.

### Console Output Format

Step 1 displays loops in a clean table:
```
File Name                 Function Name        Loop Detected
--------------------------------------------------------------------------------------------------------------
complex_template.cpp      matrix_multiply      3 level nested loop (line #17), innermost at line #20
--------------------------------------------------------------------------------------------------------------
complex_template.cpp      relu_activation      1 level nested loop (line #139)
--------------------------------------------------------------------------------------------------------------
```

## Usage Examples

### Example 1: Neural Network Kernels (Quick Test)

```bash
python src/multi_compiler_verifier.py --file data/test_neural_network.c
```

**Output**: 3 loops analyzed (dense layer, ReLU, convolution)

### Example 2: Image Processing Operations

```bash
python src/multi_compiler_verifier.py --file data/test_image_processing.c
```

**Output**: 2 functions with blur and brightness adjustment loops

### Example 3: Complex Nested Loops

```bash
# 6-level nested tiled matrix multiplication
python src/multi_compiler_verifier.py --file data/test_tiled_loops.c

# 3D stencil computation
python src/multi_compiler_verifier.py --file data/test_nested_3level.c
```

**Output**: Deep nesting analysis with complete loop context

### Example 4: Comprehensive Multi-Domain Analysis

```bash
python src/multi_compiler_verifier.py --file data/complex_template.c
```

**Output**: 32 functions, ~38 loop structures analyzed across multiple domains

### Example 5: Function-Specific Analysis

```bash
# Analyze only loops in the lstm_cell function
python src/multi_compiler_verifier.py --file data/complex_template.c --function lstm_cell

# Compare vectorization of specific function
python src/multi_compiler_verifier.py --file data/test_neural_network.c --function conv2d_forward
```

**Output**: Filters and analyzes only loops within the specified function

**Use case**: When you have a large file with many functions but want to focus on optimizing a specific function

### Example 6: With Custom Configuration

```bash
python src/multi_compiler_verifier.py --file data/complex_template.c --config config/my_config.yaml
```

## Dependencies

- `openai>=1.0.0` - LLM Integration
- `pyyaml>=6.0` - Configuration parsing
- `pycparser>=2.21` - C/C++ code parsing

## Key Improvements

Recent enhancements (January 2026):

- ✅ **Complete Nested Loop Analysis**: Sends full loop context to LLM (not just innermost)
- ✅ **Smart Function Detection**: Automatically extracts function names from code
- ✅ **Function Filtering**: Analyze loops in specific functions with `--function` flag
- ✅ **Clean Table Output**: Organized loop reporting by file/function
- ✅ **On-Prem LLM Only**: Simplified to single secure endpoint
- ✅ **Suppressed Error Messages**: Clean output without parser warnings
- ✅ **Multi-Loop Functions**: Added comprehensive test suite with multiple loops per function

## Summary

This system provides:

- ✅ On-prem LLM-based vectorization analysis
- ✅ Simultaneous multi-compiler verification (4 compilers)
- ✅ Comprehensive test suite with 32+ functions and multiple test files
- ✅ Complete nested loop context analysis (up to 6 levels deep)
- ✅ Automated report generation (3 formats)
- ✅ Production-ready for AMD internal use

Perfect for identifying vectorization opportunities in C/C++ code with enterprise-grade accuracy!
