/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.layers;

import java.util.stream.IntStream;
import kishida.cnn.activation.LinearFunction;
import kishida.cnn.kernels.NormalizeKernel;

/**
 *
 * @author naoki
 */
public class NormalizeLayer extends ImageNeuralLayer {
    float[] averages;
    float[] rates;
    int size;
    float threshold;
    boolean useGpu;

    public NormalizeLayer(String name, int size, float threshold, ImageNeuralLayer preLayer, boolean useGpu) {
        this(name, size, threshold,
                preLayer.outputChannels, preLayer.outputWidth, preLayer.outputHeight, preLayer, useGpu);
    }

    public NormalizeLayer(String name, int size, float threshold,
            int channels, int width, int height, ImageNeuralLayer preLayer, boolean useGpu) {
        super(name, new LinearFunction(), channels, width, height, channels, width, height);
        this.preLayer = preLayer;
        this.size = size;
        this.threshold = threshold;
        this.useGpu = useGpu;
    }

    @Override
    public float[] forward(float[] in) {
        averages = new float[in.length];
        rates = new float[in.length];
        result = NormalizeKernel.INSTANCE.normalize(in, inputChannels, inputWidth, inputHeight,
                size, averages, rates, threshold, useGpu);
        return result;
    }

    @Override
    public float[] backward(float[] in, float[] delta) {
        float[] result = new float[delta.length];
        IntStream.range(0, delta.length).parallel()
                .forEach(ch -> result[ch] = delta[ch] * rates[ch] + averages[ch]);
        return result;
    }

    @Override
    public String toString() {
        return String.format("Normalize:%s size:%dx%d stride:1 in:%dx%dx%d out %dx%dx%d",
                name, this.size, this.size,
                this.inputWidth, this.inputHeight, this.inputChannels,
                this.outputWidth, this.outputHeight, this.outputChannels);
    }

}
