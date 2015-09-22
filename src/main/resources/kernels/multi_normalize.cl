__kernel void average(
    int inputChannels,
    int inputWidth,
    int inputHeight,
    int size,
    float threshold,
    __global const float* input,
    __global float* averages,
    __global float* stds,
    int len
){
    int xy = get_global_id(0);
    if(xy >= len){
        return;
    }
    int x = xy / inputHeight;
    int y = xy % inputHeight;

    float total = 0;
    int count = 0;
    for(int i = 0; i < size; ++i){
        int xx = x + i - size / 2;
        if(xx < 0 || xx >= inputWidth){
            continue;
        }
        for(int j = 0; j < size; ++j){
            int yy = y + j - size / 2;
            if(yy < 0 || yy >= inputHeight){
                continue;
            }
            for(int ch = 0; ch < inputChannels; ++ch){
                total += input[ch * inputHeight * inputWidth + xy];
                ++count;
            }
        }
    }
    float average = total / count;
    float variance = 0;
    for(int i = 0; i < size; ++i){
        int xx = x + i - size / 2;
        if(xx < 0 || xx >= inputWidth){
            continue;
        }
        for(int j = 0; j < size; ++j){
            int yy = y + j - size / 2;
            if(yy < 0 || yy >= inputHeight){
                continue;
            }
            for(int ch = 0; ch < inputChannels; ++ch){
                float data = input[ch * inputHeight * inputWidth + xy];
                variance += (data - average) * (data - average);
            }
        }
    }
    averages[xy] = average;
    stds[xy] = max(threshold, sqrt(variance / count));

}

__kernel void forward(
    int inputChannels,
    int inputWidth,
    int inputHeight,
    __global const float* input,
    __global const float* averages,
    __global const float* stds,
    __global float* result,
    int count
){
    int chxy = get_global_id(0);
    if(chxy >= count){
        return;
    }
    int xy = chxy % (inputWidth * inputHeight);
    result[chxy] = (input[chxy] - averages[xy]) / stds[xy];
}