__kernel void relu(
  __global float *result,
  int count
){
   int i = get_global_id(0);
   if(i >= count){
      return;
   }
   float r = result[i];
   result[i] = r >= 0 ? r : 0;
}

__kernel void relu_diff(
  __global const float* result,
  __global float* diff,
  int count
){
   int i = get_global_id(0);
   if(i >= count){
      return;
   }
   diff[i] = result[i] >= 0 ? 1 : 0;
}

__kernel void softmax_before(
  __global const float* result,
  __global float* exped,
  int count
){
   int i = get_global_id(0);
   if(i >= count){
      return;
   }
   exped[i] = exp(min(700.0f, result[i]));
}  

__kernel void softmax(
  __global const float* exped,
  __global float* result,
  int count
){
   int i = get_global_id(0);
   if(i >= count){
      return;
   }
   float sum = 0;
   for(int j = 0; j < count; ++j){
      sum += exped[j];
   }
   result[i] = exped[i] / sum;
}

__kernel void softmax_diff(
  __global const float* result,
  __global float* diff,
  int count
){
   int i = get_global_id(0);
   if(i >= count){
      return;
   }
   float r = result[i];
   diff[i] = r * (1 - r);
}
