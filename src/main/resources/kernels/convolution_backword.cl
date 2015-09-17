__kernel void delta(
   int inputWidth, 
   int inputHeight, 
   int filterSize, 
   int outputChannels, 
   int stride, 
   int outputWidth, 
   int outputHeight, 
   __global float *result, 
   __global float *delta, 
   __global float *filter, 
   int inputChannels, 
   __global float *newDelta
){
    int chxxyy = get_global_id(0);
    int ch = chxxyy / (inputWidth * inputHeight);
    int xx = (chxxyy % (inputWidth * inputHeight)) / inputHeight;
    int yy = chxxyy % inputHeight;
    int sizeHalf = filterSize / 2;
    float tempDelta = 0.0f;
    for (int f = 0; f<outputChannels; f++){
       for (int i = 0; i<filterSize; i++){
          int x = ((xx - i) + sizeHalf) / stride;
          if ((((xx - i) + sizeHalf) % stride)==0 && x>=0 && x<outputWidth){
             for (int j = 0; j<filterSize; j++){
                int y = ((yy - j) + sizeHalf) / stride;
                if ((((yy - j) + sizeHalf) % stride)==0 && y>=0 && y<outputHeight){
                   int fxy = (((f * outputWidth) * outputHeight) + (x * outputHeight)) + y;
                   float d = (result[fxy]>=0.0f)?delta[fxy]:0.0f;
                   tempDelta = tempDelta + (d * filter[((((((f * inputChannels) * filterSize) * filterSize) + ((ch * filterSize) * filterSize)) + (i * filterSize)) + j)]);
                }
             }
          }
       }
    }
    newDelta[chxxyy]  = tempDelta;
}

__kernel void filter(
   int inputChannels, 
   int filterSize, 
   int outputWidth, 
   int outputHeight, 
   __global float *result, 
   __global float *delta, 
   int stride, 
   int inputWidth, 
   int inputHeight, 
   float learningRate, 
   __global float *input, 
   __global float *filter
){
    int fchij = get_global_id(0);
    int f = fchij / ((inputChannels * filterSize) * filterSize);
    int ch = (fchij % ((inputChannels * filterSize) * filterSize)) / (filterSize * filterSize);
    int i = (fchij % (filterSize * filterSize)) / filterSize;
    int j = fchij % filterSize;
    float df = 0.0f;
    for (int x = 0; x<outputWidth; x++){
        for (int y = 0; y<outputHeight; y++){
            int fxy = (((f * outputWidth) * outputHeight) + (x * outputHeight)) + y;
            float d = (result[fxy]>=0.0f)?delta[fxy]:0.0f;
            int xx = x * stride + i - filterSize / 2;
            if (xx >= 0 && xx < inputWidth){
                int yy = y * stride + j - filterSize / 2;
                if (yy >= 0 && yy < inputHeight){
                    df = df + d * learningRate * 
                        input[ch * inputWidth * inputHeight + xx * inputHeight + yy];
                }
            }
        }
    }
    filter[fchij] = filter[fchij] + df;
}

__kernel void bias(
    __global float *result, 
    __global float *delta, 
    __global float *tempBiasDelta, 
    float learningRate
){
    int fxy = get_global_id(0);
    float d = result[fxy]>=0.0f ? delta[fxy] : 0.0f;
    tempBiasDelta[fxy]  = learningRate * d;
}

__kernel void biasAfter(
    int outputWidth,
    int outputHeight,
    __global float *tempBiasDelta,
    __global float *biasDelta
){
    int f = get_global_id(0);
    float b = 0;
    for(int xy = 0; xy < outputWidth * outputHeight; ++xy){
        b += tempBiasDelta[f * outputWidth * outputHeight + xy];
    }
    biasDelta[f] += b;
}
