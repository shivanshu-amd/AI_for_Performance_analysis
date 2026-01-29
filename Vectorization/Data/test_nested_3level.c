void stencil_3d(float *input, float *output, int nx, int ny, int nz) {
    for (int k = 1; k < nz - 1; k++) {
        for (int j = 1; j < ny - 1; j++) {
            for (int i = 1; i < nx - 1; i++) {
                int idx = k * (nx * ny) + j * nx + i;
                output[idx] = (input[idx-1] + input[idx+1] +
                              input[idx-nx] + input[idx+nx] +
                              input[idx-nx*ny] + input[idx+nx*ny]) / 6.0f;
            }
        }
    }
}
