/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.kernels;

import com.aparapi.Kernel;
import java.util.Arrays;
import java.util.stream.IntStream;

/**
 *
 * @author naoki
 */
public class ConvolutionBackwordKernel extends Kernel {
    public static ConvolutionBackwordKernel INSTANCE = new ConvolutionBackwordKernel();

    private ConvolutionBackwordKernel() {
        setExplicit(true);
    }

    @Override
    public void run() {
        int fxy = getGlobalId();
        proc(fxy);
    }

    private void proc(int fxy) {
        float d = result[fxy] >= 0 ? delta[fxy] : 0;
        int f = fxy / (outputWidth * outputHeight);
        int x = (fxy % (outputWidth * outputHeight)) / outputHeight;
        int y = fxy % outputHeight;
        for (int ch = 0; ch < inputChannels; ++ch) {
            for (int i = 0; i < filterSize; ++i) {
                int xx = x * stride + i - filterSize / 2;
                if (xx >= 0 && xx < inputWidth) {
                    for (int j = 0; j < filterSize; ++j) {
                        int yy = y * stride + j - filterSize / 2;
                        if (yy >= 0 && yy < inputHeight) {
                            float dxinp = d * input[ch * inputWidth * inputHeight + xx * inputHeight + yy];
                            int fcij = f * inputChannels * filterSize * filterSize +
                                    ch * filterSize * filterSize + i * filterSize + j;
                            tempDelta[f * inputChannels * inputWidth * inputHeight +
                                    ch * inputWidth * inputHeight + xx * inputHeight + yy] += dxinp * filter[fcij];
                            filterDelta[fcij] += dxinp * learningRate;
                        }
                    }
                }
            }
        }
        tempBiasDelta[fxy] = learningRate * d;
    }
    float[] input;
    float[] result;
    int inputChannels;
    int inputWidth;
    int inputHeight;
    float[] filter;
    int outputChannels;
    int outputWidth;
    int outputHeight;
    int filterSize;
    int stride;
    float[] bias;
    float[] delta;
    float learningRate;
    float[] tempDelta;
    float[] filterDelta;
    float[] biasDelta;
    float[] tempBiasDelta;

    public float[] backward(float[] delta, float[] result,
            float[] input, int inputChannels, int inputWidth, int inputHeight,
            float[] filter, int outputChannels, int outputWidth, int outputHeight,
            float[] filterDelta, float[] biasDelta,
            int filterSize, int stride, float[] bias, float learningRate, boolean useGpu) {
        this.delta = delta;
        this.input = input;
        this.inputChannels = inputChannels;
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.filter = filter;
        this.outputChannels = outputChannels;
        this.outputWidth = outputWidth;
        this.outputHeight = outputHeight;
        this.filterSize = filterSize;
        this.stride = stride;
        this.bias = bias;
        this.result = result;
        this.tempDelta = new float[outputChannels * inputChannels * inputWidth * inputHeight];
        this.learningRate = learningRate;// / (outputWidth * outputHeight);
        this.biasDelta = biasDelta;
        this.filterDelta = filterDelta;
        this.tempBiasDelta = Arrays.copyOf(result, result.length);
        if (useGpu) {
            put(filter);
            put(delta);
            put(filterDelta);
            put(input);
            put(result);
            put(tempDelta);
            execute(outputChannels * outputWidth * outputHeight);
            get(filterDelta);
            get(tempDelta);
            get(tempBiasDelta);
        } else {
            IntStream.range(0, outputChannels).parallel().forEach(f -> {
                for (int xy = 0; xy < outputWidth * outputHeight; ++xy) {
                    proc(f * outputWidth * outputHeight + xy);
                }
            });
        }
        float[] newDelta = new float[input.length];
        IntStream.range(0, outputChannels).parallel().forEach(f -> {
            for (int chxy = 0; chxy < inputChannels * inputWidth * inputHeight; ++chxy) {
                newDelta[chxy] += tempDelta[f * inputChannels * inputWidth * inputHeight + chxy];
            }
            for (int xy = 0; xy < outputWidth * outputHeight; ++xy) {
                biasDelta[f] += tempBiasDelta[f * outputWidth * outputHeight + xy];
            }
        });
        return newDelta;
    }

}
