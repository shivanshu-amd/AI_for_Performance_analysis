void matmul_tiled(float *A, float *B, float *C, int N, int tile_size) {
    for (int i0 = 0; i0 < N; i0 += tile_size) {
        for (int j0 = 0; j0 < N; j0 += tile_size) {
            for (int k0 = 0; k0 < N; k0 += tile_size) {
                for (int i = i0; i < i0 + tile_size && i < N; i++) {
                    for (int j = j0; j < j0 + tile_size && j < N; j++) {
                        float sum = C[i*N + j];
                        for (int k = k0; k < k0 + tile_size && k < N; k++) {
                            sum += A[i*N + k] * B[k*N + j];
                        }
                        C[i*N + j] = sum;
                    }
                }
            }
        }
    }
}
