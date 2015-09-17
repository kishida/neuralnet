void kishida_cnn_kernels_ConvolutionForwardKernel__proc(This *this, int fxy){
   int f = fxy / (this->outputHeight * this->outputWidth);
   int x = (fxy % (this->outputHeight * this->outputWidth)) / this->outputHeight;
   int y = fxy % this->outputHeight;
   float r = 0.0f;
   for (int ch = 0; ch<this->inputChannels; ch++){
      for (int i = 0; i<this->filterSize; i++){
         int xx = ((x * this->stride) + i) - (this->filterSize / 2);
         if (xx>=0 && xx<this->inputWidth){
            for (int j = 0; j<this->filterSize; j++){
               int yy = ((y * this->stride) + j) - (this->filterSize / 2);
               if (yy>=0 && yy<this->inputHeight){
                  r = r + (this->input[((((ch * this->inputWidth) * this->inputHeight) + (xx * this->inputHeight)) + yy)] * this->filter[((((((f * this->inputChannels) * this->filterSize) * this->filterSize) + ((ch * this->filterSize) * this->filterSize)) + (i * this->filterSize)) + j)]);
               }
            }
         }
      }
   }
   this->result[fxy]  = r + this->bias[f];
   return;
}
__kernel void run(
   int outputHeight, 
   int outputWidth, 
   int inputChannels, 
   int filterSize, 
   int stride, 
   int inputWidth, 
   int inputHeight, 
   __global float *input, 
   __global float *filter, 
   __global float *result, 
   __global float *bias, 
   int passid
){
   This thisStruct;
   This* this=&thisStruct;
   this->outputHeight = outputHeight;
   this->outputWidth = outputWidth;
   this->inputChannels = inputChannels;
   this->filterSize = filterSize;
   this->stride = stride;
   this->inputWidth = inputWidth;
   this->inputHeight = inputHeight;
   this->input = input;
   this->filter = filter;
   this->result = result;
   this->bias = bias;
   this->passid = passid;
   {
      int fixy = get_global_id(0);
      kishida_cnn_kernels_ConvolutionForwardKernel__proc(this, fixy);
      return;
   }
}