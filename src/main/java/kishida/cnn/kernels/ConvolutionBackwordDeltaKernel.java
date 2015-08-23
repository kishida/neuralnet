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
public class ConvolutionBackwordDeltaKernel extends Kernel {
    public static ConvolutionBackwordDeltaKernel INSTANCE = new ConvolutionBackwordDeltaKernel();

    private ConvolutionBackwordDeltaKernel() {
        setExplicit(true);
    }

    @Override
    public void run() {
        int fxy = getGlobalId();
        proc(fxy);
    }

    private void proc(int chxxyy) {
        int ch = chxxyy / (inputWidth * inputHeight);
        int xx = (chxxyy % (inputWidth * inputHeight)) / inputHeight;
        int yy = chxxyy % inputHeight;
        int sizeHalf = filterSize / 2;
        double tempDelta = 0;
        for (int f = 0; f < outputChannels; ++f) {
            for (int i = 0; i < filterSize; ++i) {
                int x = (xx - i + sizeHalf) / stride;
                if ((xx - i + sizeHalf) % stride == 0 && // yy == y * stride + j -sizeHalf だとなぜかGPUで動かない
                x >= 0 && x < outputWidth) {
                    for (int j = 0; j < filterSize; ++j) {
                        int y = (yy - j + sizeHalf) / stride;
                        if ((yy - j + sizeHalf) % stride == 0 && y >= 0 && y < outputHeight) {
                            int fxy = f * outputWidth * outputHeight + x * outputHeight + y;
                            double d = (result[fxy] > 0 ? 1 : 0) * delta[fxy];
                            tempDelta += d * input[chxxyy] *
                                    filter[f * inputChannels * filterSize * filterSize +
                                    ch * filterSize * filterSize + i * filterSize + j];
                        }
                    }
                }
            }
        }
        newDelta[chxxyy] = tempDelta;
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
    double[] delta;
    double[] newDelta;

    public double[] backword(double[] input, double[] delta, double[] result,
            int inputChannels, int inputWidth, int inputHeight,
            double[] filter, int outputChannels, int outputWidth, int outputHeight,
            int filterSize, int stride, boolean useGpu) {
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
        this.newDelta = new double[inputChannels * inputWidth * inputHeight];
        if (useGpu) {
            put(filter);
            put(delta);
            put(result);
            put(input);
            execute(inputChannels * inputWidth * inputHeight);
            get(newDelta);
        } else {
            IntStream.range(0, inputChannels * inputWidth).parallel().forEach(chx -> {
                for (int y = 0; y < inputHeight; ++y) {
                    proc(chx * inputHeight + y);
                }
            });
        }
        return newDelta;
    }

}
