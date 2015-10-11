__kernel void forward(
   int inSize, 
   int out, 
   __global const int *dropout, 
   __global const float *in, 
   __global float *weight, 
   __global float *result, 
   int count
){
    int j = get_global_id(0);
    if(j >= count){
       return;
    }
    float r = 0;
    for (int i = 0; i<inSize; i++){
       r += in[i] * weight[i * out + j];
    }
    result[j] = r * dropout[j];// + bias[j];
}
