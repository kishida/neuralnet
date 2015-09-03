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
public class ConvolutionBackwordBiasKernel extends Kernel {
    public static ConvolutionBackwordBiasKernel INSTANCE = new ConvolutionBackwordBiasKernel();

    private ConvolutionBackwordBiasKernel() {
        //setExplicit(true);
    }

    @Override
    public void run() {
        int fxy = getGlobalId();
        proc(fxy);
    }
    float[] result;
    float[] delta;
    float localEp;
    float[] tempBiasDelta;

    private void proc(int fxy) {
        float d = result[fxy] >= 0 ? delta[fxy] : 0;
        // float d = (result[fxy] >= 0 ? 1 : 0) * delta[fxy]; GPUで*delta[fxy]が無視された・・・
        tempBiasDelta[fxy] = localEp * d;
    }

    public void backwordBias(float[] delta, float[] result,
            int outputChannels, int outputWidth, int outputHeight,
            float[] bias, float ep, float[] tempBiasDelta, boolean useGpu) {
        this.delta = delta;
        this.result = result;
        this.localEp = ep;// / outputWidth;// * outputHeight);
        this.tempBiasDelta = tempBiasDelta;
        if (useGpu) {
            //put(delta);
            //put(result);
            execute(outputChannels * outputWidth * outputHeight);
            //get(tempBiasDelta);
            IntStream.range(0, outputChannels).parallel().forEach(f -> {
                for (int xy = 0; xy < outputWidth * outputHeight; ++xy) {
                    bias[f] += tempBiasDelta[f * outputWidth * outputHeight + xy];
                }
            });
        } else {
            IntStream.range(0, outputChannels).parallel().forEach(f -> {
                for (int xy = 0; xy < outputWidth * outputHeight; ++xy) {
                    proc(f * outputWidth * outputHeight + xy);
                    bias[f] += tempBiasDelta[f * outputWidth * outputHeight + xy];
                }
            });
        }
    }

}
