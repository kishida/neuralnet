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
public class ConvolutionLocalNormalizationKernel extends Kernel{
    public static final ConvolutionLocalNormalizationKernel INSTANCE = new ConvolutionLocalNormalizationKernel();

    public ConvolutionLocalNormalizationKernel() {
    }

    public void localNormalization(float[] result, int outputChannels, int outputWidth, int outputHeight, boolean useGpu){
        this.result = result;
        this.outputChannels = outputChannels;
        this.outputWidth = outputWidth;
        this.outputHeight = outputHeight;
        if(useGpu){
            execute(outputChannels * outputWidth * outputHeight);
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
        int chxy = getGlobalId();
        procGpu(chxy);
    }

    float[] result;
    int outputWidth;
    int outputHeight;
    int outputChannels;
    static final int n = 5;

    //@PrivateMemorySpace(n) float[] sigma = new float[n]; // not work

    public void procGpu(int chxy){
        final int k = 2;
        final float a = 0.0001f;
        final float b = 0.75f;
        int ch = chxy / (outputWidth * outputHeight);
        int xy = chxy % (outputWidth * outputHeight);

        float sum = 0;
        for(int lp = max(0, ch - n / 2); lp <= min(outputChannels - 1, ch + n / 2); ++lp){
            sum += result[lp * outputWidth * outputHeight + xy] *
                   result[lp * outputWidth * outputHeight + xy];
        }
        result[chxy] = result[chxy] /
                pow(k + a * sum, b);

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
