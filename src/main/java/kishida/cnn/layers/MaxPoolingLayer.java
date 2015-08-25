/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.layers;

import java.util.Arrays;
import java.util.stream.IntStream;
import kishida.cnn.activation.LinearFunction;

/**
 *
 * @author naoki
 */
public class MaxPoolingLayer extends ImageNeuralLayer {
    int size;
    int stride;

    public MaxPoolingLayer(String name, int size, int stride, ImageNeuralLayer preLayer) {
        this(name, size, stride, preLayer.outputChannels, preLayer.outputWidth, preLayer.outputHeight);
        this.preLayer = preLayer;
    }

    public MaxPoolingLayer(String name, int size, int stride, int channels, int inputWidth, int inputHeight) {
        super(name, new LinearFunction(),
                channels, inputWidth, inputHeight,
                channels, inputWidth / stride, inputHeight / stride);
        this.size = size;
        this.stride = stride;
        result = new double[outputChannels * outputWidth * outputHeight];
        newDelta = new double[channels * inputWidth * inputHeight];
    }

    /** プーリング(max) */
    @Override
    public double[] forward(double[] data) {
        IntStream.range(0, inputChannels).parallel().forEach(ch -> {
            for (int x = 0; x < outputWidth; ++x) {
                for (int y = 0; y < outputHeight; ++y) {
                    double max = Double.NEGATIVE_INFINITY;
                    for (int i = 0; i < size; ++i) {
                        int xx = x * stride + i - size / 2;
                        if (xx < 0 || xx >= inputWidth) {
                            continue;
                        }
                        for (int j = 0; j < size; ++j) {
                            int yy = y * stride + j - size / 2;
                            if (yy < 0 || yy >= inputHeight) {
                                continue;
                            }
                            double d = data[ch * inputWidth * inputHeight + xx * inputHeight + yy];
                            if (max < d) {
                                max = d;
                            }
                        }
                    }
                    result[ch * outputWidth * outputHeight + x * outputHeight + y] = max;
                }
            }
        });
        return result;
    }

    double[] newDelta;
    @Override
    public double[] backward(double[] in, double[] delta) {
        Arrays.fill(newDelta, 0);
        IntStream.range(0, inputChannels).parallel().forEach(ch -> {
            for (int x = 0; x < outputWidth; ++x) {
                for (int y = 0; y < outputHeight; ++y) {
                    double max = Double.NEGATIVE_INFINITY;
                    int maxX = 0;
                    int maxY = 0;
                    for (int i = 0; i < size; ++i) {
                        int xx = x * stride + i - size / 2;
                        if (xx < 0 || xx >= inputWidth) {
                            continue;
                        }
                        for (int j = 0; j < size; ++j) {
                            int yy = y * stride + j - size / 2;
                            if (yy < 0 || yy >= inputHeight) {
                                continue;
                            }
                            double d = in[ch * inputWidth * inputHeight + xx * inputWidth + yy];
                            if (max < d) {
                                max = d;
                                maxX = xx;
                                maxY = yy;
                            }
                        }
                    }
                    int chxy = ch * outputWidth * outputHeight + x * outputHeight + y;
                    newDelta[ch * inputWidth * inputHeight + maxX * inputHeight + maxY] +=
                             delta[chxy];
                }
            }
        });
        return newDelta;
    }

    @Override
    public String toString() {
        return String.format("Max pooling:%s size:%dx%d stride:%d in:%dx%dx%d out %dx%dx%d",
                name, this.size, this.size, this.stride,
                this.inputWidth, this.inputHeight, this.inputChannels,
                this.outputWidth, this.outputHeight, this.outputChannels);
    }

}
