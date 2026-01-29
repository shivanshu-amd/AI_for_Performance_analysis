void blur_horizontal(float *input, float *output, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 1; x < width - 1; x++) {
            int idx = y * width + x;
            output[idx] = (input[idx-1] + input[idx] + input[idx+1]) / 3.0f;
        }
    }
}

void adjust_brightness(float *image, float factor, int size) {
    for (int i = 0; i < size; i++) {
        image[i] *= factor;
    }
}
