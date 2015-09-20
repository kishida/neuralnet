__kernel void forward(
   int outputHeight, 
   int outputWidth, 
   int inputChannels, 
   int filterSize, 
   int stride, 
   int inputWidth, 
   int inputHeight, 
   __global const float *input, 
   __global const float *filter, 
   __global float *result, 
   __global const float *bias, 
   int count
){
   int fxy = get_global_id(0);
   if(fxy >= count){
      return;
   }
   int f = fxy / (outputHeight * outputWidth);
   int x = (fxy % (outputHeight * outputWidth)) / outputHeight;
   int y = fxy % outputHeight;
   float r = 0.0f;
   for (int ch = 0; ch<inputChannels; ch++){
      for (int i = 0; i<filterSize; i++){
         int xx = ((x * stride) + i) - (filterSize / 2);
         if (xx>=0 && xx<inputWidth){
            for (int j = 0; j<filterSize; j++){
               int yy = ((y * stride) + j) - (filterSize / 2);
               if (yy>=0 && yy<inputHeight){
                  r += input[ch * inputWidth * inputHeight + xx * inputHeight + yy] * 
                          filter[f * inputChannels * filterSize * filterSize +
                                 ch * filterSize * filterSize + i * filterSize + j];
               }
            }
         }
      }
   }
   float rs = r + bias[f];
   result[fxy] = rs >= 0 ? rs : 0;
}

__kernel void localNormalize(
   int outputWidth, 
   int outputHeight, 
   int outputChannels, 
   __global float *result, 
   int count
){
   int chxy = get_global_id(0);
   if(chxy >= count){
      return;
   }
   float k = 2;
   float a = 1.0E-4f;
   float b = 0.75f;
   int n = 5;
   int ch = chxy / (outputWidth * outputHeight);
   int xy = chxy % (outputWidth * outputHeight);
   float sum = 0.0f;
   
   for (int lp = max(0, ch - n / 2); lp <= min(outputChannels - 1, ch + n / 2); lp++){
      sum += result[lp * outputWidth * outputHeight + xy] * 
             result[lp * outputWidth * outputHeight + xy];
   }
   result[chxy] /= pow(k + a * sum, b);
}
