/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.kernels;

import com.aparapi.Kernel;
import java.util.stream.IntStream;
import kishida.cnn.activation.ActivationFunction;

/**
 *
 * @author naoki
 */
public class ConvolutionForwardKernel extends Kernel {
    public static ConvolutionForwardKernel INSTANCE = new ConvolutionForwardKernel();

    private ConvolutionForwardKernel() {
        setExplicit(true);
    }

    @Override
    public void run() {
        int fixy = getGlobalId();
        proc(fixy);
    }

    private void proc(int fxy) {
        int f = fxy / (outputHeight * outputWidth);
        int x = (fxy % (outputHeight * outputWidth)) / outputHeight;
        int y = fxy % outputHeight;
        float r = 0; // 毎回resultを足すよりもまとめて足したほうがGPUの場合に速くなる。
        for (int ch = 0; ch < inputChannels; ++ch) {
            for (int i = 0; i < filterSize; ++i) {
                int xx = x * stride + i - filterSize / 2;
                if (xx >= 0 && xx < inputWidth) {
                    for (int j = 0; j < filterSize; ++j) {
                        int yy = y * stride + j - filterSize / 2;
                        if (yy >= 0 && yy < inputHeight) {
                            r += input[ch * inputWidth * inputHeight + xx * inputHeight + yy] *
                                    filter[f * inputChannels * filterSize * filterSize +
                                    ch * filterSize * filterSize + i * filterSize + j];
                        }
                    }
                }
            }
        }
        result[fxy] = r + bias[f];
    }
    float[] result;
    float[] input;
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

    public float[] forward(float[] input, int inputChannels, int inputWidth, int inputHeight,
            float[] filter, int outputChannels, int outputWidth, int outputHeight, float[] result,
            int filterSize, int stride, float[] bias, ActivationFunction activation, boolean useGpu) {
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
        if (useGpu) {
            put(input);
            put(filter);
            put(bias);
            execute(outputChannels * outputWidth * outputHeight);
            get(result);
        } else {
            IntStream.range(0, outputChannels * outputWidth * outputHeight).parallel().forEach(fxy -> {
                proc(fxy);
            });
        }
        IntStream.range(0, result.length).parallel().forEach((fi) -> {
            result[fi] = activation.apply(result[fi]);
        });
        return result;
    }

}
