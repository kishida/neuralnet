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
public class ConvolutionBackwordFilterKernel extends Kernel {
    public static ConvolutionBackwordFilterKernel INSTANCE = new ConvolutionBackwordFilterKernel();

    private ConvolutionBackwordFilterKernel() {
        setExplicit(true);
    }

    @Override
    public void run() {
        int fchij = getGlobalId();
        proc(fchij);
    }

    private void proc(int fchij) {
        int f = fchij / (inputChannels * filterSize * filterSize);
        int ch = (fchij % (inputChannels * filterSize * filterSize)) / (filterSize * filterSize);
        int i = (fchij % (filterSize * filterSize)) / filterSize;
        int j = fchij % filterSize;
        float df = 0;
        for (int x = 0; x < outputWidth; ++x) {
            for (int y = 0; y < outputHeight; ++y) {
                int fxy = f * outputWidth * outputHeight + x * outputHeight + y;
                float d = result[fxy] >= 0 ? delta[fxy] : 0;
                int xx = x * stride + i - filterSize / 2;
                if (xx >= 0 && xx < inputWidth) {
                    int yy = y * stride + j - filterSize / 2;
                    if (yy >= 0 && yy < inputHeight) {
                        df += d * learningRate * input[ch * inputWidth * inputHeight + xx * inputHeight + yy];
                    }
                }
            }
        }
        filter[fchij] += df;
    }
    float[] input;
    float[] result;
    float[] delta;
    int inputChannels;
    int inputWidth;
    int inputHeight;
    float[] filter;
    int outputChannels;
    int outputWidth;
    int outputHeight;
    int filterSize;
    int stride;
    float learningRate;

    public void backword(float[] delta, float[] result,
            float[] input, int inputChannels, int inputWidth, int inputHeight,
            float[] filter, int outputChannels, int outputWidth, int outputHeight,
            int filterSize, int stride, float learningRate, boolean useGpu) {
        this.input = input;
        this.delta = delta;
        this.inputChannels = inputChannels;
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.filter = filter;
        this.outputChannels = outputChannels;
        this.outputWidth = outputWidth;
        this.outputHeight = outputHeight;
        this.filterSize = filterSize;
        this.stride = stride;
        this.result = result;
        this.learningRate = learningRate;// / outputWidth;// * outputHeight);
        if (useGpu) {
            put(delta);
            put(filter);
            put(input);
            put(result);
            execute(outputChannels * inputChannels * filterSize * filterSize);
            get(filter);
        } else {
            IntStream.range(0, outputChannels).parallel().forEach((f) -> {
                for (int chij = 0; chij < inputChannels * filterSize * filterSize; ++chij) {
                    proc(f * inputChannels * filterSize * filterSize + chij);
                }
            });
        }
    }

}
