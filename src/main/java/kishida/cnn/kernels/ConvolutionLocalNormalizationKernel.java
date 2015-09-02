/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.kernels;

import com.amd.aparapi.Kernel;
import java.util.Arrays;
import java.util.stream.IntStream;

/**
 *
 * @author naoki
 */
public class ConvolutionLocalNormalizationKernel extends Kernel{
    public static final ConvolutionLocalNormalizationKernel INSTANCE = new ConvolutionLocalNormalizationKernel();

    public ConvolutionLocalNormalizationKernel() {
    }

    public void localNormalization(double[] result, int outputChannels, int outputWidth, int outputHeight, boolean useGpu){
        this.result = result;
        this.outputChannels = outputChannels;
        this.outputWidth = outputWidth;
        this.outputHeight = outputHeight;
        if(useGpu && outputWidth * outputHeight > 500){
            execute(outputWidth * outputHeight);
            //throw new UnsupportedOperationException("because I dont know how to use private memory.");
        }else{
            IntStream.range(0, outputWidth).parallel().forEach(x -> {
                for(int y = 0; y < outputHeight; ++y){
                    procCpu(x * outputHeight + y, new double[n]);
                }
            });
        }
    }


    @Override
    public void run() {
        int xy = getGlobalId();
        procGpu(xy);
    }

    double[] result;
    int outputWidth;
    int outputHeight;
    int outputChannels;
    static final int n = 5;

    @PrivateMemorySpace(n) double[] sigma = new double[n]; // not work

    public void procGpu(int xy){
        final int k = 2;
        final double a = 0.0001;
        final double b = 0.75;
        int lp = 0;
        for(; lp < n / 2; ++lp){
            sigma[lp] =
                    result[lp * outputWidth * outputHeight + xy] *
                    result[lp * outputWidth * outputHeight + xy];
        }
        for(int ch = 0; ch < outputChannels; ++ch){
            sigma[lp % n] = lp >= outputChannels ? 0 :
                    result[lp * outputWidth * outputHeight + xy] *
                    result[lp * outputWidth * outputHeight + xy];
            lp = lp + 1;
            double sum = 0;
            for(int i = 0; i < n; ++i){
                sum += sigma[i];
            }
            result[ch * outputWidth * outputHeight + xy] = result[ch * outputWidth * outputHeight + xy] /
                    pow(k + a * sum, b);
        }

    }

    public void procCpu(int xy, double[] sig){
        final int k = 2;
        final double a = 0.0001;
        final double b = 0.75;
        int lp = 0;
        for(; lp < n / 2; ++lp){
            sig[lp] =
                    result[lp * outputWidth * outputHeight + xy] *
                    result[lp * outputWidth * outputHeight + xy];
        }
        for(int ch = 0; ch < outputChannels; ++ch){
            sig[lp % 5] = lp >= outputChannels ? 0 :
                    result[lp * outputWidth * outputHeight + xy] *
                    result[lp * outputWidth * outputHeight + xy];
            lp = lp + 1;
            double sum = Arrays.stream(sig).sum();
            result[ch * outputWidth * outputHeight + xy] = result[ch * outputWidth * outputHeight + xy] /
                    pow(k + a * sum, b);
        }

    }

}
