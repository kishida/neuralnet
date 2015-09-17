void kishida_cnn_kernels_FullyForwardKernel__proc(This *this, int j){
   if (this->dropout[j]==1){
      for (int i = 0; i<this->inSize; i++){
         this->result[j]  = this->result[j] + (this->in[i] * this->weight[((i * this->out) + j)]);
      }
      this->result[j]  = this->result[j] + this->bias[j];
   }
   return;
}
__kernel void run(
   __global int *dropout, 
   int inSize, 
   __global float *result, 
   __global float *in, 
   __global float *weight, 
   int out, 
   __global float *bias, 
   int passid
){
   This thisStruct;
   This* this=&thisStruct;
   this->dropout = dropout;
   this->inSize = inSize;
   this->result = result;
   this->in = in;
   this->weight = weight;
   this->out = out;
   this->bias = bias;
   this->passid = passid;
   {
      int j = get_global_id(0);
      kishida_cnn_kernels_FullyForwardKernel__proc(this, j);
      return;
   }
}
