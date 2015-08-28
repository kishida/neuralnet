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
        double d = (result[fxy] > 0 ? 1 : 0) * delta[fxy];
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
                            double dxinp = d * input[ch * inputWidth * inputHeight + xx * inputHeight + yy];
                            int fcij = f * inputChannels * filterSize * filterSize +
                                    ch * filterSize * filterSize + i * filterSize + j;
                            tempDelta[f * inputChannels * inputWidth * inputHeight +
                                    ch * inputWidth * inputHeight + xx * inputHeight + yy] += dxinp * oldfilter[fcij];
                            filter[fcij] += dxinp * localEp;
                        }
                    }
                }
            }
        }
        //bias[f] += localEp * d;
        biasDelta[fxy] = localEp * d;
    }
    double[] input;
    double[] result;
    int inputChannels;
    int inputWidth;
    int inputHeight;
    double[] filter;
    int outputChannels;
    int outputWidth;
    int outputHeight;
    int filterSize;
    int stride;
    double[] bias;
    double[] delta;
    double[] oldfilter;
    double localEp;
    double[] tempDelta;
    double[] biasDelta;

    public double[] backward(double[] delta, double[] result,
            double[] input, int inputChannels, int inputWidth, int inputHeight,
            double[] filter, int outputChannels, int outputWidth, int outputHeight,
            int filterSize, int stride, double[] bias, double ep, boolean useGpu) {
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
        this.oldfilter = Arrays.copyOf(filter, filter.length);
        this.tempDelta = new double[outputChannels * inputChannels * inputWidth * inputHeight];
        this.localEp = ep / (outputWidth * outputHeight);
        this.biasDelta = Arrays.copyOf(result, result.length);
        if (useGpu) {
            put(filter);
            put(delta);
            put(oldfilter);
            put(input);
            put(result);
            put(tempDelta);
            execute(outputChannels * outputWidth * outputHeight);
            get(filter);
            get(tempDelta);
            get(biasDelta);
        } else {
            IntStream.range(0, outputChannels).parallel().forEach(f -> {
                for (int xy = 0; xy < outputWidth * outputHeight; ++xy) {
                    proc(f * outputWidth * outputHeight + xy);
                }
            });
        }
        double[] newDelta = new double[input.length];
        IntStream.range(0, outputChannels).parallel().forEach(f -> {
            for (int chxy = 0; chxy < inputChannels * inputWidth * inputHeight; ++chxy) {
                newDelta[chxy] += tempDelta[f * inputChannels * inputWidth * inputHeight + chxy];
            }
            for (int xy = 0; xy < outputWidth * outputHeight; ++xy) {
                bias[f] += biasDelta[f * outputWidth * outputHeight + xy];
            }
        });
        return newDelta;
    }

}