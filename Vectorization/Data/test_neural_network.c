void dense_forward(float *weights, float *input, float *bias, float *output,
                   int input_size, int output_size) {
    for (int i = 0; i < output_size; i++) {
        float sum = bias[i];
        for (int j = 0; j < input_size; j++) {
            sum += weights[i * input_size + j] * input[j];
        }
        output[i] = sum;
    }
}

void relu_activation(float *data, int size) {
    for (int i = 0; i < size; i++) {
        if (data[i] < 0.0f) {
            data[i] = 0.0f;
        }
    }
}

void conv2d_forward(float *input, float *kernel, float *output,
                    int in_h, int in_w, int k_h, int k_w,
                    int out_h, int out_w) {
    for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
            float sum = 0.0f;
            for (int kh = 0; kh < k_h; kh++) {
                for (int kw = 0; kw < k_w; kw++) {
                    int ih = oh + kh;
                    int iw = ow + kw;
                    sum += input[ih * in_w + iw] * kernel[kh * k_w + kw];
                }
            }
            output[oh * out_w + ow] = sum;
        }
    }
}
