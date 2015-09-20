/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.kernels;

import com.amd.aparapi.Kernel;
import java.util.stream.IntStream;

/**
 *
 * @author naoki
 */
public class FullyForwardKernel extends Kernel{
    public static final FullyForwardKernel INSTANCE = new FullyForwardKernel();

    public FullyForwardKernel() {
    }

    @Override
    public void run() {
        int j = getGlobalId();
        proc(j);
    }

    private void proc(int j){
        if(dropout[j] == 1){
            float r = 0;
            for (int i = 0; i < inSize; ++i) {
                r += in[i] * weight[i * out + j];
            }
            result[j] = r + bias[j];
        }else{
            result[j] = 0;
        }
    }
    int out;
    int inSize;
    int[] dropout;
    float[] in;
    float[] result;
    float[] weight;
    float[] bias;
    public void forward(int out, int[] dropout, float[] in, float[] result, float[] weight, float[] bias, boolean useGpu){
        this.inSize = in.length;
        this.dropout = dropout;
        this.out = out;
        this.in = in;
        this.result = result;
        this.weight = weight;
        this.bias = bias;
        if(useGpu){
            execute(out);
        }else{
            IntStream.range(0, out).parallel().forEach(j -> {
                proc(j);
            });
        }
    }
}
