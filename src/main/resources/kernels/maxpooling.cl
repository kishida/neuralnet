__kernel void forward(
    int inputWidth,
    int inputHeight,
    int outputWidth,
    int outputHeight,
    int size,
    int stride,
    __global const float* data,
    __global float* result,
    int count
){
    int chxy = get_global_id(0);
    if(chxy >= count){
        return;
    }
    
    int ch = chxy / (outputWidth * outputHeight);
    int x = (chxy % (outputWidth * outputHeight)) / outputHeight;
    int y = chxy % outputHeight;

    float max = -INFINITY;
    for (int i = 0; i < size; ++i) {
        int xx = x * stride + i - size / 2;
        if (xx < 0 || xx >= inputWidth) {
            continue;
        }
        for (int j = 0; j < size; ++j) {
            int yy = y * stride + j - size / 2;
            if (yy < 0 || yy >= inputHeight) {
                continue;
            }
            float d = data[ch * inputWidth * inputHeight + xx * inputHeight + yy];
            if (max < d) {
                max = d;
            }
        }
    }
    result[chxy] = max;

}

__kernel void backword(
    int inputWidth,
    int inputHeight,
    int outputWidth,
    int outputHeight,
    int size,
    int stride,
    __global const float* input,
    __global const float* delta,
    __global float* newDelta,
    int count
){
    int chxy = get_global_id(0);
    if(chxy >= count){
        return;
    }
    
    int ch = chxy / (inputWidth * inputHeight);
    int xi = (chxy % (inputWidth * inputHeight)) / inputHeight;
    int yi = chxy % inputHeight;

    float nd = 0;
    for(int x = max(0, (xi - size / 2) / stride - 1); 
            x < min(outputWidth, (xi + size / 2) / stride + 1); ++x){
        for(int y = max(0, (yi - size / 2) / stride - 1);
                y < min(outputHeight, (yi + size / 2) / stride + 1); ++y){
            float max = -INFINITY;
            int maxX = 0;
            int maxY = 0;
            for (int i = 0; i < size; ++i) {
                int xx = x * stride + i - size / 2;
                if (xx < 0 || xx >= inputWidth) {
                    continue;
                }
                for (int j = 0; j < size; ++j) {
                    int yy = y * stride + j - size / 2;
                    if (yy < 0 || yy >= inputHeight) {
                        continue;
                    }
                    float d = input[ch * inputWidth * inputHeight + xx * inputWidth + yy];
                    if (max < d) {
                        max = d;
                        maxX = xx;
                        maxY = yy;
                    }
                }
            }
            if(maxX == xi && maxY == yi){
                nd += delta[ch * outputWidth * outputHeight + x * outputHeight + y];
            }
        }
    }
    newDelta[chxy] = nd;
}