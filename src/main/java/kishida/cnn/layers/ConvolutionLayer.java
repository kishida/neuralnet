/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.layers;

import com.amd.aparapi.Kernel;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import kishida.cnn.ConvolutionalNet;
import kishida.cnn.activation.RetifierdLinear;

/** 畳み込み層 */
public class ConvolutionLayer extends ImageNeuralLayer {
    double[] filter;
    double[] bias;
    int stride;
    int filterSize;
    boolean useGpu;

    public ConvolutionLayer(String name, ImageNeuralLayer preLayer, int filterCount, int size, int stride, boolean useGpu) {
        this(name, preLayer.outputChannels, preLayer.outputWidth, preLayer.outputWidth, filterCount, size, stride, useGpu);
        this.preLayer = preLayer;
    }

    public ConvolutionLayer(String name, int channel, int width, int height, int filterCount, int size, int stride, boolean useGpu) {
        super(name, new RetifierdLinear(), channel, width, height, filterCount, width / stride, height / stride);
        this.filter = IntStream.range(0, size * size * channel * filterCount).mapToDouble((int d) -> (ConvolutionalNet.random.nextDouble() + ConvolutionalNet.random.nextDouble() + ConvolutionalNet.random.nextDouble() + ConvolutionalNet.random.nextDouble() + ConvolutionalNet.random.nextDouble() + ConvolutionalNet.random.nextDouble() + ConvolutionalNet.random.nextDouble() + ConvolutionalNet.random.nextDouble() + ConvolutionalNet.random.nextDouble() + ConvolutionalNet.random.nextDouble() + ConvolutionalNet.random.nextDouble() + ConvolutionalNet.random.nextDouble() - 6 - 0.5) / size / size / channel).toArray();
        this.bias = DoubleStream.generate(() -> .1).limit(filterCount).toArray();
        this.stride = stride;
        this.filterSize = size;
        this.useGpu = useGpu;
    }

    /** 畳み込みフィルタを適用する */
    @Override
    public double[] forward(double[] img) {
        result = ConvolutionalNet.convolutionForwardKernel.forward(img, inputChannels, inputWidth, inputHeight, filter, outputChannels, outputWidth, outputHeight, filterSize, stride, bias, activation, useGpu);
        return result;
    }

    /** 畳み込み層の学習 */
    @Override
    public double[] backward(double[] input, double[] delta) {
        if (useGpu) {
            // GPUバージョン
            double[] newDelta = ConvolutionalNet.convolutionBackwordDeltaKernel.backword(input, delta, result, inputChannels, inputWidth, inputHeight, filter, outputChannels, outputWidth, outputHeight, filterSize, stride, useGpu);
            ConvolutionalNet.convolutionBackwordFilterKernel.backword(delta, result, input, inputChannels, inputWidth, inputHeight, filter, outputChannels, outputWidth, outputHeight, filterSize, stride, useGpu);
            ConvolutionalNet.convolutionBackwordBiasKernel.backwordBias(delta, result, outputChannels, outputWidth, outputHeight, bias, useGpu);
            if (ConvolutionalNet.convolutionBackwordDeltaKernel.getExecutionMode() != Kernel.EXECUTION_MODE.GPU || ConvolutionalNet.convolutionBackwordFilterKernel.getExecutionMode() != Kernel.EXECUTION_MODE.GPU || ConvolutionalNet.convolutionBackwordBiasKernel.getExecutionMode() != Kernel.EXECUTION_MODE.GPU) {
                useGpu = false;
            }
            if (!useGpu) {
                System.out.println("Can't use GPU on " + name);
                System.out.println("delta" + ConvolutionalNet.convolutionBackwordDeltaKernel.getExecutionMode());
                System.out.println("filter" + ConvolutionalNet.convolutionBackwordFilterKernel.getExecutionMode());
                System.out.println("bias" + ConvolutionalNet.convolutionBackwordBiasKernel.getExecutionMode());
            }
            return newDelta;
            /*
            return convolutionBackwordKernel.backword(delta, result,
            input, inputChannels, inputWidth, inputHeight,
            filter, outputChannels, outputWidth, outputHeight,
            filterSize, stride, bias, act, useGpu);
             */
        } else {
            // CPUバージョン
            return ConvolutionalNet.convolutionBackwordKernel.backward(delta, result, input, inputChannels, inputWidth, inputHeight, filter, outputChannels, outputWidth, outputHeight, filterSize, stride, bias, false);
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
        return String.format("Convolutional:%s filter:%dx%d x%d stride:%d in:%dx%dx%d out %dx%dx%d", name, this.filterSize, this.filterSize, this.outputChannels, this.stride, this.inputWidth, this.inputHeight, this.inputChannels, this.outputWidth, this.outputHeight, this.outputChannels);
    }

}
