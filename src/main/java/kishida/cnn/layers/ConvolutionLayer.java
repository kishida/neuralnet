/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.layers;

import com.amd.aparapi.Kernel;
import java.util.Arrays;
import java.util.DoubleSummaryStatistics;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import kishida.cnn.ConvolutionalNet;
import kishida.cnn.activation.RetifierdLinear;
import kishida.cnn.kernels.ConvolutionBackwordBiasKernel;
import kishida.cnn.kernels.ConvolutionBackwordDeltaKernel;
import kishida.cnn.kernels.ConvolutionBackwordFilterKernel;
import kishida.cnn.kernels.ConvolutionBackwordKernel;
import kishida.cnn.kernels.ConvolutionForwardKernel;

/** 畳み込み層 */
public class ConvolutionLayer extends ImageNeuralLayer {
    double[] filter;
    double[] bias;
    int stride;
    int filterSize;
    boolean useGpu;
    double ep;

    public ConvolutionLayer(String name, ImageNeuralLayer preLayer,
            int filterCount, int size, int stride, double ep, boolean useGpu) {
        this(name, preLayer, preLayer.outputChannels, preLayer.outputWidth, preLayer.outputWidth,
                filterCount, size, stride, ep, useGpu);
    }

    public ConvolutionLayer(String name, ImageNeuralLayer preLayer,
            int channel, int width, int height, int filterCount, int size, int stride, double ep, boolean useGpu) {
        super(name, new RetifierdLinear(), channel, width, height, filterCount, width / stride, height / stride);
        this.ep = ep;
        this.preLayer = preLayer;
        this.filter = IntStream.range(0, size * size * channel * filterCount)
                .mapToDouble(d -> (ConvolutionalNet.random.nextDouble() + ConvolutionalNet.random.nextDouble() +
                        ConvolutionalNet.random.nextDouble() + ConvolutionalNet.random.nextDouble() +
                        ConvolutionalNet.random.nextDouble() + ConvolutionalNet.random.nextDouble() +
                        ConvolutionalNet.random.nextDouble() + ConvolutionalNet.random.nextDouble() +
                        ConvolutionalNet.random.nextDouble() + ConvolutionalNet.random.nextDouble() +
                        ConvolutionalNet.random.nextDouble() + ConvolutionalNet.random.nextDouble())
                        / size / size / channel).toArray();
        this.bias = DoubleStream.generate(() -> 0).limit(filterCount).toArray();
        this.stride = stride;
        this.filterSize = size;
        this.useGpu = useGpu;
        this.result = new double[outputChannels * outputWidth * outputHeight];
        this.biasDelta = new double[result.length];
    }

    /** 畳み込みフィルタを適用する */
    @Override
    public double[] forward(double[] img) {
        result = ConvolutionForwardKernel.INSTANCE.forward(img, inputChannels, inputWidth, inputHeight,
                filter, outputChannels, outputWidth, outputHeight, result, filterSize, stride, bias, activation, useGpu);
        return result;
    }

    double[] biasDelta;

    /** 畳み込み層の学習 */
    @Override
    public double[] backward(double[] input, double[] delta) {
        if (useGpu) {
            // GPUバージョン
            double[] newDelta = ConvolutionBackwordDeltaKernel.INSTANCE.backword(input, delta, result,
                    inputChannels, inputWidth, inputHeight,
                    filter, outputChannels, outputWidth, outputHeight, filterSize, stride, useGpu);
            ConvolutionBackwordFilterKernel.INSTANCE.backword(delta, result,
                    input, inputChannels, inputWidth, inputHeight,
                    filter, outputChannels, outputWidth, outputHeight, filterSize, stride, ep, useGpu);
            ConvolutionBackwordBiasKernel.INSTANCE.backwordBias(delta, result,
                    outputChannels, outputWidth, outputHeight, bias, ep, biasDelta, useGpu);
            if (ConvolutionBackwordDeltaKernel.INSTANCE.getExecutionMode() != Kernel.EXECUTION_MODE.GPU ||
                    ConvolutionBackwordFilterKernel.INSTANCE.getExecutionMode() != Kernel.EXECUTION_MODE.GPU ||
                    ConvolutionBackwordBiasKernel.INSTANCE.getExecutionMode() != Kernel.EXECUTION_MODE.GPU) {
                useGpu = false;
            }
            if (!useGpu) {
                System.out.println("Can't use GPU on " + name);
                System.out.println("delta" + ConvolutionBackwordDeltaKernel.INSTANCE.getExecutionMode());
                System.out.println("filter" + ConvolutionBackwordFilterKernel.INSTANCE.getExecutionMode());
                System.out.println("bias" + ConvolutionBackwordBiasKernel.INSTANCE.getExecutionMode());
            }
            return newDelta;
        } else {
            // CPUバージョン
            return ConvolutionBackwordKernel.INSTANCE.backward(delta, result,
                    input, inputChannels, inputWidth, inputHeight,
                    filter, outputChannels, outputWidth, outputHeight, filterSize, stride, bias, ep, false);
        }
    }

    public double[] getFilter() {
        return filter;
    }

    public double[] getBias() {
        return bias;
    }

    @Override
    public String toString() {
        DoubleSummaryStatistics sum = Arrays.stream(filter).summaryStatistics();
        return String.format("Convolutional:%s filter:%dx%d x%d stride:%d in:%dx%dx%d out %dx%dx%d%n"
                + "Filter %.2f-%.2f ave:%.2f filtertotal:%.2f",
                name, filterSize, filterSize, outputChannels, stride,
                inputWidth, inputHeight, inputChannels, outputWidth, outputHeight, outputChannels,
                sum.getMin(), sum.getMax(), sum.getAverage(), sum.getSum() / inputChannels / outputChannels);
    }

}
