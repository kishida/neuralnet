__kernel void backword_delta(
    int outputSize,
    __global const int* dropout,
    __global const float* delta,
    __global const float* diffed,
    __global const float* weight,
    __global float* newDelta,
    int count
){
    int i = get_global_id(0);
    if(i >= count){
        return;
    }
    float nd = 0;
    for (int j = 0; j < outputSize; ++j) {
        if (dropout[j] != 1) {
            continue;
        }
        float d = diffed[j] * delta[j];
        nd += d *  weight[i * outputSize + j];//in[i] *;
    }
    newDelta[i] = nd; 
}

__kernel void backword_weight(
    int outputSize,
    float learningRate,
    __global const int* dropout,
    __global const float* input,
    __global const float* delta,
    __global const float* diffed,
    __global const float* weight,
    __global float* weightDelta,
    int count
){
    int ij = get_global_id(0);
    if(ij >= count){
        return;
    }
    int i = ij / outputSize;
    int j = ij % outputSize;
    if (dropout[j] != 1) {
        return;
    }
    float d = diffed[j] * delta[j];
    weightDelta[ij] += d * input[i] * learningRate;

}

__kernel void backword_bias(
    int outputSize,
    float learningRate,
    __global const int* dropout,
    __global const float* delta,
    __global const float* diffed,
    __global float* biasDelta,
    int count
){
    int j = get_global_id(0);
    if(j >= count){
        return;
    }
    if (dropout[j] != 1) {
        return;
    }
    biasDelta[j] += diffed[j] * delta[j] * learningRate;
}

__kernel void joinFilter(
    float weightDecay,
    float learningRate,
    int count,
    int filterCount,
    __global float* filter,
    __global const float* filterDelta,
    int len
){
    int f = get_global_id(0);
    if(f >= len){
        return;
    }
    filter[f] += filterDelta[f] / count
        - (f < filterCount ? weightDecay : 0) * learningRate * filter[f];
}