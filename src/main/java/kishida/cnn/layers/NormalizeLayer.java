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
    double[] averages;
    double[] rates;
    int size;
    double threshold;
    boolean useGpu;

    public NormalizeLayer(String name, int size, double threshold, ImageNeuralLayer preLayer, boolean useGpu) {
        this(name, size, threshold,
                preLayer.outputChannels, preLayer.outputWidth, preLayer.outputHeight, preLayer, useGpu);
    }

    public NormalizeLayer(String name, int size, double threshold,
            int channels, int width, int height, ImageNeuralLayer preLayer, boolean useGpu) {
        super(name, new LinearFunction(), channels, width, height, channels, width, height);
        this.preLayer = preLayer;
        this.size = size;
        this.threshold = threshold;
        this.useGpu = useGpu;
    }

    @Override
    public double[] forward(double[] in) {
        averages = new double[in.length];
        rates = new double[in.length];
        result = NormalizeKernel.INSTANCE.normalize(in, inputChannels, inputWidth, inputHeight,
                size, averages, rates, threshold, useGpu);
        return result;
    }

    @Override
    public double[] backward(double[] in, double[] delta) {
        return IntStream.range(0, delta.length).parallel()
                .mapToDouble(ch -> delta[ch] * rates[ch] + averages[ch]).toArray();
    }

    @Override
    public String toString() {
        return String.format("Normalize:%s size:%dx%d stride:1 in:%dx%dx%d out %dx%dx%d",
                name, this.size, this.size,
                this.inputWidth, this.inputHeight, this.inputChannels,
                this.outputWidth, this.outputHeight, this.outputChannels);
    }

}
