/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.kernels;

import com.aparapi.Kernel;
import java.util.stream.IntStream;

/**
 *
 * @author naoki
 */
public class ConvolutionLocalNormalizationKernel extends Kernel{
    public static final ConvolutionLocalNormalizationKernel INSTANCE = new ConvolutionLocalNormalizationKernel();

    public ConvolutionLocalNormalizationKernel() {
    }

    public void localNormalization(float[] result, int outputChannels, int outputWidth, int outputHeight, boolean useGpu){
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
                    procCpu(x * outputHeight + y, new float[n]);
                }
            });
        }
    }


    @Override
    public void run() {
        int xy = getGlobalId();
        procGpu(xy);
    }

    float[] result;
    int outputWidth;
    int outputHeight;
    int outputChannels;
    static final int n = 5;

    @PrivateMemorySpace(n) float[] sigma = new float[n]; // not work

    public void procGpu(int xy){
        final int k = 2;
        final float a = 0.0001f;
        final float b = 0.75f;
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
            float sum = 0;
            for(int i = 0; i < n; ++i){
                sum += sigma[i];
            }
            result[ch * outputWidth * outputHeight + xy] = result[ch * outputWidth * outputHeight + xy] /
                    pow(k + a * sum, b);
        }

    }

    public void procCpu(int xy, float[] sig){
        final int k = 2;
        final float a = 0.0001f;
        final float b = 0.75f;
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
            //float sum = (float)ConvolutionalNet.summary(sig).getSum();
            float sum = 0;
            for(float d : sig){
                sum += d;
            }
            result[ch * outputWidth * outputHeight + xy] = result[ch * outputWidth * outputHeight + xy] /
                    pow(k + a * sum, b);
        }

    }

}
