/**
 * Complex Multi-Domain Template for Vectorization Analysis
 * Contains functions from various computational domains with mixed vectorization potential
 */

#include <math.h>
#include <string.h>
#include <stdlib.h>

// ============================================================================
// MATRIX OPERATIONS
// ============================================================================

/**
 * Matrix multiplication - Dense matrix operation
 */
void matrix_multiply(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/**
 * Matrix transpose - Switches rows and columns
 */
void matrix_transpose(float* input, float* output, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

/**
 * Triangular matrix solver - Forward substitution
 */
void triangular_solve(float* L, float* b, float* x, int n) {
    for (int i = 0; i < n; i++) {
        float sum = b[i];
        for (int j = 0; j < i; j++) {
            sum -= L[i * n + j] * x[j];  // Dependency on previous x[j]
        }
        x[i] = sum / L[i * n + i];
    }
}

// ============================================================================
// WEATHER SIMULATION
// ============================================================================

/**
 * Temperature diffusion using 2D stencil
 */
void heat_diffusion_2d(float* temp, float* temp_new, int width, int height, float alpha) {
    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            int idx = i * width + j;
            temp_new[idx] = temp[idx] + alpha * (
                temp[idx - 1] + temp[idx + 1] +
                temp[idx - width] + temp[idx + width] -
                4.0f * temp[idx]
            );
        }
    }
}

/**
 * Atmospheric pressure calculation with temperature and humidity effects
 */
void calculate_pressure(float* temperature, float* humidity, float* pressure, 
                       int size, float base_pressure) {
    for (int i = 0; i < size; i++) {
        float temp_factor = 1.0f - (temperature[i] - 273.15f) / 273.15f;
        float humid_factor = 1.0f + 0.1f * humidity[i];
        
        if (temperature[i] < 273.15f) {
            // Ice conditions - different formula
            pressure[i] = base_pressure * temp_factor * 0.9f;
        } else if (temperature[i] > 310.0f) {
            // High temperature conditions
            pressure[i] = base_pressure * temp_factor * humid_factor * 1.1f;
        } else {
            // Normal conditions
            pressure[i] = base_pressure * temp_factor * humid_factor;
        }
    }
}

/**
 * Wind velocity update with pressure gradient and Coriolis effects
 */
void update_wind_velocity(float* wind_u, float* wind_v, float* pressure, 
                         int width, int height, float dt) {
    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            int idx = i * width + j;
            
            // Pressure gradient affects wind
            float dp_dx = (pressure[idx + 1] - pressure[idx - 1]) / 2.0f;
            float dp_dy = (pressure[idx + width] - pressure[idx - width]) / 2.0f;
            
            // Coriolis effect (non-linear coupling)
            float coriolis_u = 0.0001f * wind_v[idx];
            float coriolis_v = -0.0001f * wind_u[idx];
            
            wind_u[idx] -= dt * (dp_dx + coriolis_u);
            wind_v[idx] -= dt * (dp_dy + coriolis_v);
        }
    }
}

// ============================================================================
// NEURAL NETWORK OPERATIONS
// ============================================================================

/**
 * Dense layer forward pass
 */
void dense_layer(float* input, float* weights, float* bias, float* output,
                int input_size, int output_size) {
    for (int i = 0; i < output_size; i++) {
        float sum = bias[i];
        for (int j = 0; j < input_size; j++) {
            sum += weights[i * input_size + j] * input[j];
        }
        output[i] = sum;
    }
}

/**
 * ReLU activation function
 */
void relu_activation(float* data, int size) {
    for (int i = 0; i < size; i++) {
        if (data[i] < 0.0f) {
            data[i] = 0.0f;
        }
    }
}

/**
 * LSTM cell computation with gates
 */
void lstm_cell(float* input, float* hidden, float* cell_state,
              float* weights_i, float* weights_f, float* weights_c, float* weights_o,
              int size) {
    for (int i = 0; i < size; i++) {
        // Input gate
        float i_gate = 0.0f;
        for (int j = 0; j < size; j++) {
            i_gate += weights_i[i * size + j] * (input[j] + hidden[j]);
        }
        i_gate = 1.0f / (1.0f + expf(-i_gate));  // Sigmoid
        
        // Forget gate
        float f_gate = 0.0f;
        for (int j = 0; j < size; j++) {
            f_gate += weights_f[i * size + j] * (input[j] + hidden[j]);
        }
        f_gate = 1.0f / (1.0f + expf(-f_gate));
        
        // Cell candidate
        float c_candidate = 0.0f;
        for (int j = 0; j < size; j++) {
            c_candidate += weights_c[i * size + j] * (input[j] + hidden[j]);
        }
        c_candidate = tanhf(c_candidate);
        
        // Update cell state (dependency on previous cell_state)
        cell_state[i] = f_gate * cell_state[i] + i_gate * c_candidate;
        
        // Output gate
        float o_gate = 0.0f;
        for (int j = 0; j < size; j++) {
            o_gate += weights_o[i * size + j] * (input[j] + hidden[j]);
        }
        o_gate = 1.0f / (1.0f + expf(-o_gate));
        
        // Update hidden state
        hidden[i] = o_gate * tanhf(cell_state[i]);
    }
}

/**
 * Batch normalization with learnable parameters
 */
void batch_normalize(float* data, int batch_size, int features, 
                    float* gamma, float* beta, float epsilon) {
    for (int f = 0; f < features; f++) {
        // Calculate mean
        float mean = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            mean += data[b * features + f];
        }
        mean /= batch_size;
        
        // Calculate variance
        float variance = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            float diff = data[b * features + f] - mean;
            variance += diff * diff;
        }
        variance /= batch_size;
        
        // Normalize
        float std_dev = sqrtf(variance + epsilon);
        for (int b = 0; b < batch_size; b++) {
            int idx = b * features + f;
            data[idx] = gamma[f] * (data[idx] - mean) / std_dev + beta[f];
        }
    }
}

// ============================================================================
// IMAGE PROCESSING
// ============================================================================

/**
 * 2D Convolution operation
 */
void convolution_2d(float* input, float* kernel, float* output,
                   int img_height, int img_width, int k_size) {
    int pad = k_size / 2;
    
    for (int i = pad; i < img_height - pad; i++) {
        for (int j = pad; j < img_width - pad; j++) {
            float sum = 0.0f;
            
            for (int ki = 0; ki < k_size; ki++) {
                for (int kj = 0; kj < k_size; kj++) {
                    int img_i = i - pad + ki;
                    int img_j = j - pad + kj;
                    sum += input[img_i * img_width + img_j] * kernel[ki * k_size + kj];
                }
            }
            
            output[i * img_width + j] = sum;
        }
    }
}

/**
 * Gaussian blur horizontal pass
 */
void gaussian_blur_horizontal(float* input, float* output, int width, int height,
                             float* kernel, int kernel_size) {
    int pad = kernel_size / 2;
    
    for (int i = 0; i < height; i++) {
        for (int j = pad; j < width - pad; j++) {
            float sum = 0.0f;
            for (int k = 0; k < kernel_size; k++) {
                sum += input[i * width + (j - pad + k)] * kernel[k];
            }
            output[i * width + j] = sum;
        }
    }
}

/**
 * Sobel edge detection with gradient computation
 */
void sobel_edge_detection(unsigned char* input, unsigned char* output, 
                         int width, int height) {
    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            int idx = i * width + j;
            
            // Horizontal gradient
            int gx = -input[idx - width - 1] + input[idx - width + 1]
                    -2*input[idx - 1] + 2*input[idx + 1]
                    -input[idx + width - 1] + input[idx + width + 1];
            
            // Vertical gradient  
            int gy = -input[idx - width - 1] - 2*input[idx - width] - input[idx - width + 1]
                    +input[idx + width - 1] + 2*input[idx + width] + input[idx + width + 1];
            
            // Magnitude
            int magnitude = (int)sqrtf((float)(gx*gx + gy*gy));
            output[idx] = (magnitude > 255) ? 255 : magnitude;
        }
    }
}

/**
 * Median filter 3x3 window
 */
void median_filter_3x3(unsigned char* input, unsigned char* output, 
                      int width, int height) {
    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            unsigned char window[9];
            int idx = 0;
            
            // Collect 3x3 window
            for (int di = -1; di <= 1; di++) {
                for (int dj = -1; dj <= 1; dj++) {
                    window[idx++] = input[(i + di) * width + (j + dj)];
                }
            }
            
            // Bubble sort to find median
            for (int k = 0; k < 9; k++) {
                for (int m = k + 1; m < 9; m++) {
                    if (window[k] > window[m]) {
                        unsigned char temp = window[k];
                        window[k] = window[m];
                        window[m] = temp;
                    }
                }
            }
            
            output[i * width + j] = window[4];  // Median value
        }
    }
}

// ============================================================================
// STENCIL COMPUTATIONS
// ============================================================================

/**
 * 3D 7-point stencil computation
 */
void stencil_3d_7point(float* grid, float* grid_new, 
                       int nx, int ny, int nz, float factor) {
    for (int k = 1; k < nz - 1; k++) {
        for (int j = 1; j < ny - 1; j++) {
            for (int i = 1; i < nx - 1; i++) {
                int idx = k * nx * ny + j * nx + i;
                
                grid_new[idx] = factor * (
                    grid[idx - 1] + grid[idx + 1] +
                    grid[idx - nx] + grid[idx + nx] +
                    grid[idx - nx*ny] + grid[idx + nx*ny] -
                    6.0f * grid[idx]
                );
            }
        }
    }
}

/**
 * 2D 9-point stencil with diagonal terms
 */
void stencil_2d_9point(double* in, double* out, int width, int height, double coef) {
    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            int idx = i * width + j;
            
            double center = in[idx];
            double sum = 0.0;
            
            // Cardinal directions
            sum += 4.0 * (in[idx-1] + in[idx+1] + in[idx-width] + in[idx+width]);
            
            // Diagonals
            sum += 2.0 * (in[idx-width-1] + in[idx-width+1] + 
                         in[idx+width-1] + in[idx+width+1]);
            
            out[idx] = coef * sum + (1.0 - 20.0*coef) * center;
        }
    }
}

/**
 * Iterative Jacobi solver for PDEs
 */
void jacobi_iteration(float* phi, float* phi_new, float* rhs,
                     int nx, int ny, float dx, float dy) {
    float dx2 = dx * dx;
    float dy2 = dy * dy;
    float factor = 0.5 / (1.0f/dx2 + 1.0f/dy2);
    
    for (int i = 1; i < ny - 1; i++) {
        for (int j = 1; j < nx - 1; j++) {
            int idx = i * nx + j;
            
            float laplacian = (phi[idx-1] - 2.0f*phi[idx] + phi[idx+1]) / dx2
                            + (phi[idx-nx] - 2.0f*phi[idx] + phi[idx+nx]) / dy2;
            
            phi_new[idx] = factor * (rhs[idx] - laplacian);
        }
    }
}

// ============================================================================
// ENCODER/DECODER OPERATIONS
// ============================================================================

/**
 * Run-length encoding compression
 */
int run_length_encode(unsigned char* input, unsigned char* output, 
                     int input_size) {
    int out_idx = 0;
    int i = 0;
    
    while (i < input_size) {
        unsigned char current = input[i];
        int count = 1;
        
        // Count consecutive identical values
        while (i + count < input_size && input[i + count] == current && count < 255) {
            count++;
        }
        
        output[out_idx++] = current;
        output[out_idx++] = (unsigned char)count;
        i += count;
    }
    
    return out_idx;
}

/**
 * Huffman tree traversal for decoding
 */
void huffman_decode_bits(unsigned char* encoded, int* tree, unsigned char* decoded,
                        int encoded_bits, int root) {
    int current_node = root;
    int decoded_idx = 0;
    
    for (int i = 0; i < encoded_bits; i++) {
        int byte_idx = i / 8;
        int bit_idx = i % 8;
        int bit = (encoded[byte_idx] >> bit_idx) & 1;
        
        // Traverse tree based on bit
        if (bit == 0) {
            current_node = tree[current_node * 2];  // Left child
        } else {
            current_node = tree[current_node * 2 + 1];  // Right child
        }
        
        // Check if leaf node (contains decoded character)
        if (current_node < 256) {
            decoded[decoded_idx++] = (unsigned char)current_node;
            current_node = root;  // Reset to root
        }
    }
}

/**
 * Base64 encoding
 */
void base64_encode(unsigned char* input, char* output, int input_len) {
    const char* base64_chars = 
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    
    int i = 0;
    int j = 0;
    
    while (i < input_len) {
        unsigned char byte1 = input[i++];
        unsigned char byte2 = (i < input_len) ? input[i++] : 0;
        unsigned char byte3 = (i < input_len) ? input[i++] : 0;
        
        unsigned int triple = (byte1 << 16) | (byte2 << 8) | byte3;
        
        output[j++] = base64_chars[(triple >> 18) & 0x3F];
        output[j++] = base64_chars[(triple >> 12) & 0x3F];
        output[j++] = base64_chars[(triple >> 6) & 0x3F];
        output[j++] = base64_chars[triple & 0x3F];
    }
}

/**
 * XOR cipher with key stream
 */
void xor_cipher(unsigned char* data, unsigned char* key, int data_len, int key_len) {
    for (int i = 0; i < data_len; i++) {
        data[i] ^= key[i % key_len];
    }
}

/**
 * AES-like S-box substitution
 */
void sbox_substitution(unsigned char* data, unsigned char* sbox, int data_len) {
    for (int i = 0; i < data_len; i++) {
        data[i] = sbox[data[i]];
    }
}

// ============================================================================
// MIXED COMPLEXITY OPERATIONS
// ============================================================================

/**
 * Pointer chasing with indirect indexing
 */
void pointer_chase(int* data, int* indices, float* values, int size) {
    for (int i = 0; i < size; i++) {
        int idx = indices[i];
        values[i] = (float)data[idx];
        indices[i] = data[idx] % size;  // Update for next iteration
    }
}

/**
 * Reduction with conditional accumulation
 */
float conditional_sum(float* data, int size, float threshold) {
    float sum = 0.0f;
    
    for (int i = 0; i < size; i++) {
        if (data[i] > threshold) {
            sum += data[i] * data[i];  // Square of values above threshold
        } else {
            sum += data[i];  // Linear for values below threshold
        }
    }
    
    return sum;
}

/**
 * Histogram computation from image data
 */
void compute_histogram(unsigned char* image, int* histogram, int size) {
    // Initialize histogram
    for (int i = 0; i < 256; i++) {
        histogram[i] = 0;
    }
    
    // Count pixel values
    for (int i = 0; i < size; i++) {
        histogram[image[i]]++;  // Race condition in parallel execution
    }
}

// ============================================================================
// ADDITIONAL MULTI-LOOP FUNCTIONS
// ============================================================================

/**
 * Image normalization with min-max scaling
 */
void image_normalize(float* image, int width, int height, int channels) {
    // Find min and max for each channel
    for (int c = 0; c < channels; c++) {
        float min_val = image[c];
        float max_val = image[c];
        
        for (int i = 0; i < height * width; i++) {
            int idx = i * channels + c;
            if (image[idx] < min_val) min_val = image[idx];
            if (image[idx] > max_val) max_val = image[idx];
        }
        
        // Normalize
        float range = max_val - min_val;
        if (range > 0.0f) {
            for (int i = 0; i < height * width; i++) {
                int idx = i * channels + c;
                image[idx] = (image[idx] - min_val) / range;
            }
        }
    }
}

/**
 * Multi-stage particle physics simulation
 */
void particle_simulation(float* pos_x, float* pos_y, float* vel_x, float* vel_y,
                        float* force_x, float* force_y, int n_particles, float dt) {
    // Stage 1: Reset forces
    for (int i = 0; i < n_particles; i++) {
        force_x[i] = 0.0f;
        force_y[i] = 0.0f;
    }
    
    // Stage 2: Calculate pairwise forces
    for (int i = 0; i < n_particles; i++) {
        for (int j = i + 1; j < n_particles; j++) {
            float dx = pos_x[j] - pos_x[i];
            float dy = pos_y[j] - pos_y[i];
            float dist = sqrtf(dx*dx + dy*dy);
            
            if (dist > 0.01f) {
                float force = 1.0f / (dist * dist);
                force_x[i] += force * dx / dist;
                force_y[i] += force * dy / dist;
                force_x[j] -= force * dx / dist;
                force_y[j] -= force * dy / dist;
            }
        }
    }
    
    // Stage 3: Update velocities
    for (int i = 0; i < n_particles; i++) {
        vel_x[i] += force_x[i] * dt;
        vel_y[i] += force_y[i] * dt;
    }
    
    // Stage 4: Update positions
    for (int i = 0; i < n_particles; i++) {
        pos_x[i] += vel_x[i] * dt;
        pos_y[i] += vel_y[i] * dt;
    }
}

/**
 * Financial data processing with multiple passes
 */
void financial_analysis(float* prices, float* volumes, float* indicators,
                       int n_stocks, int n_days) {
    // Calculate moving averages
    for (int s = 0; s < n_stocks; s++) {
        float sum = 0.0f;
        for (int d = 0; d < 20 && d < n_days; d++) {
            sum += prices[s * n_days + d];
        }
        indicators[s * 3] = sum / 20.0f;
    }
    
    // Calculate volatility
    for (int s = 0; s < n_stocks; s++) {
        float mean = indicators[s * 3];
        float variance = 0.0f;
        
        for (int d = 0; d < n_days; d++) {
            float diff = prices[s * n_days + d] - mean;
            variance += diff * diff;
        }
        indicators[s * 3 + 1] = sqrtf(variance / n_days);
    }
    
    // Calculate volume-weighted average
    for (int s = 0; s < n_stocks; s++) {
        float sum_pv = 0.0f;
        float sum_v = 0.0f;
        
        for (int d = 0; d < n_days; d++) {
            int idx = s * n_days + d;
            sum_pv += prices[idx] * volumes[idx];
            sum_v += volumes[idx];
        }
        
        if (sum_v > 0.0f) {
            indicators[s * 3 + 2] = sum_pv / sum_v;
        }
    }
}

/**
 * Machine learning gradient descent with multiple operations
 */
void gradient_descent_step(float* weights, float* gradients, float* momentum,
                          float* data, float* labels, int n_samples, int n_features,
                          float learning_rate, float momentum_factor) {
    // Initialize gradients
    for (int f = 0; f < n_features; f++) {
        gradients[f] = 0.0f;
    }
    
    // Calculate predictions and accumulate gradients
    for (int s = 0; s < n_samples; s++) {
        float prediction = 0.0f;
        
        for (int f = 0; f < n_features; f++) {
            prediction += weights[f] * data[s * n_features + f];
        }
        
        float error = prediction - labels[s];
        
        for (int f = 0; f < n_features; f++) {
            gradients[f] += error * data[s * n_features + f];
        }
    }
    
    // Update weights with momentum
    for (int f = 0; f < n_features; f++) {
        momentum[f] = momentum_factor * momentum[f] + learning_rate * gradients[f];
        weights[f] -= momentum[f];
    }
}

/**
 * Graph algorithms - adjacency matrix operations
 */
void graph_processing(int* adj_matrix, int* degrees, float* pagerank,
                     int n_nodes, int n_iterations) {
    // Calculate node degrees
    for (int i = 0; i < n_nodes; i++) {
        degrees[i] = 0;
        for (int j = 0; j < n_nodes; j++) {
            degrees[i] += adj_matrix[i * n_nodes + j];
        }
    }
    
    // Initialize PageRank
    for (int i = 0; i < n_nodes; i++) {
        pagerank[i] = 1.0f / n_nodes;
    }
    
    // PageRank iterations
    for (int iter = 0; iter < n_iterations; iter++) {
        float* new_pagerank = (float*)malloc(n_nodes * sizeof(float));
        
        for (int i = 0; i < n_nodes; i++) {
            new_pagerank[i] = 0.15f / n_nodes;
            
            for (int j = 0; j < n_nodes; j++) {
                if (adj_matrix[j * n_nodes + i] && degrees[j] > 0) {
                    new_pagerank[i] += 0.85f * pagerank[j] / degrees[j];
                }
            }
        }
        
        for (int i = 0; i < n_nodes; i++) {
            pagerank[i] = new_pagerank[i];
        }
        
        free(new_pagerank);
    }
}

/**
 * Signal processing - FFT preparation with bit reversal
 */
void fft_prepare_and_scale(float* real, float* imag, int* bit_reverse, int n) {
    // Bit reversal permutation
    for (int i = 0; i < n; i++) {
        int j = bit_reverse[i];
        if (j > i) {
            float temp_r = real[i];
            float temp_i = imag[i];
            real[i] = real[j];
            imag[i] = imag[j];
            real[j] = temp_r;
            imag[j] = temp_i;
        }
    }
    
    // Hamming window application
    for (int i = 0; i < n; i++) {
        float window = 0.54f - 0.46f * cosf(2.0f * 3.14159f * i / (n - 1));
        real[i] *= window;
        imag[i] *= window;
    }
    
    // Normalize
    float scale = 1.0f / sqrtf((float)n);
    for (int i = 0; i < n; i++) {
        real[i] *= scale;
        imag[i] *= scale;
    }
}
