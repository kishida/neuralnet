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
        setExplicit(true);
    }

    @Override
    public void run() {
        int fxy = getGlobalId();
        proc(fxy);
    }
    double[] result;
    double[] delta;
    double localEp;
    double[] biasDelta;

    private void proc(int fxy) {
        double d = (result[fxy] > 0 ? 1 : 0) * delta[fxy];
        biasDelta[fxy] = localEp * d;
    }

    public void backwordBias(double[] delta, double[] result,
            int outputChannels, int outputWidth, int outputHeight,
            double[] bias, double ep, double[] biasDelta, boolean useGpu) {
        this.delta = delta;
        this.result = result;
        this.localEp = ep / (outputWidth * outputHeight);
        this.biasDelta = biasDelta;
        if (useGpu) {
            put(delta);
            put(result);
            execute(outputChannels * outputWidth * outputHeight);
            get(biasDelta);
            IntStream.range(0, outputChannels).parallel().forEach(f -> {
                for (int xy = 0; xy < outputWidth * outputHeight; ++xy) {
                    bias[f] += biasDelta[f * outputWidth * outputHeight + xy];
                }
            });
        } else {
            IntStream.range(0, outputChannels).parallel().forEach(f -> {
                for (int xy = 0; xy < outputWidth * outputHeight; ++xy) {
                    proc(f * outputWidth * outputHeight + xy);
                    bias[f] += biasDelta[f * outputWidth * outputHeight + xy];
                }
            });
        }
    }

}
