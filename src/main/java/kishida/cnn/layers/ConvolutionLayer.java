/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.layers;

import com.amd.aparapi.Kernel;
import java.util.Arrays;
import java.util.DoubleSummaryStatistics;
import java.util.stream.IntStream;
import kishida.cnn.ConvolutionalNet;
import kishida.cnn.activation.RetifierdLinear;
import kishida.cnn.kernels.ConvolutionBackwordBiasKernel;
import kishida.cnn.kernels.ConvolutionBackwordDeltaKernel;
import kishida.cnn.kernels.ConvolutionBackwordFilterKernel;
import kishida.cnn.kernels.ConvolutionBackwordKernel;
import kishida.cnn.kernels.ConvolutionForwardKernel;
import kishida.cnn.kernels.ConvolutionLocalNormalizationKernel;

/** 畳み込み層 */
public class ConvolutionLayer extends ImageNeuralLayer implements LerningLayer{
    float[] filter;
    float[] bias;
    float[] filterDelta;
    float[] biasDelta;
    int stride;
    int filterSize;
    boolean useGpu;
    float learningRate;

    public ConvolutionLayer(String name, ImageNeuralLayer preLayer,
            int filterCount, int size, int stride, float initBias, float learningRate, boolean useGpu) {
        this(name, preLayer, preLayer.outputChannels, preLayer.outputWidth, preLayer.outputWidth,
                filterCount, size, stride, initBias, learningRate, useGpu);
    }

    public ConvolutionLayer(String name, ImageNeuralLayer preLayer,
            int channel, int width, int height, int filterCount, int size, int stride, float initBias, float learningRate, boolean useGpu) {
        super(name, new RetifierdLinear(), channel, width, height, filterCount, width / stride, height / stride);
        this.learningRate = learningRate;
        this.preLayer = preLayer;
        this.filter = ConvolutionalNet.createGaussianArray(size * size * channel * filterCount, 0.01f);
        this.bias = new float[filterCount];
        Arrays.fill(bias, initBias);
        this.filterDelta = new float[filter.length];
        this.biasDelta = new float[bias.length];
        this.stride = stride;
        this.filterSize = size;
        this.useGpu = useGpu;
        this.result = new float[outputChannels * outputWidth * outputHeight];
        this.tempDelta = new float[result.length];
    }

    /** 畳み込みフィルタを適用する */
    @Override
    public float[] forward(float[] img) {
        result = ConvolutionForwardKernel.INSTANCE.forward(img, inputChannels, inputWidth, inputHeight,
                filter, outputChannels, outputWidth, outputHeight, result, filterSize, stride, bias, activation, useGpu);
        //localNormalization(result);
        ConvolutionLocalNormalizationKernel.INSTANCE.localNormalization(result,
                outputChannels, outputWidth, outputHeight, false);
        return result;
    }

    private void localNormalization(float[] result){
        final int n = 5;
        final int k = 2;
        final float a = 0.0001f;
        final float b = 0.75f;
        // resultをコピーするほうが楽だけど、メモリを節約するため
        final float[] sigma = new float[n];
        for(int x = 0; x < outputWidth; ++x){
            for(int y = 0; y < outputHeight; ++y){
                int xy = x * outputHeight + y;
                Arrays.fill(sigma, 0);
                int lp = 0;
                for(; lp < n / 2; ++lp){
                    sigma[lp] = result[lp * outputWidth * outputHeight + xy] * result[lp * outputWidth * outputHeight + xy];
                }
                for(int ch = 0; ch < outputChannels; ++ch){
                    sigma[lp % 5] = lp >= outputChannels ? 0 :
                            result[lp * outputWidth * outputHeight + xy] * result[lp * outputWidth * outputHeight + xy];
                    lp = lp + 1;
                    float sum = ConvolutionalNet.floatSum(sigma);
                    result[ch * outputWidth * outputHeight + xy] = result[ch * outputWidth * outputHeight + xy] /
                            (float)Math.pow(k + a * sum, b);
                }
            }
        }
    }

    float[] tempDelta;

    /** 畳み込み層の学習 */
    @Override
    public float[] backward(float[] input, float[] delta) {
        if (useGpu) {
            // GPUバージョン
            float[] newDelta = ConvolutionBackwordDeltaKernel.INSTANCE.backword(input, delta, result,
                    inputChannels, inputWidth, inputHeight,
                    filter, outputChannels, outputWidth, outputHeight, filterSize, stride, useGpu);
            ConvolutionBackwordFilterKernel.INSTANCE.backword(delta, result,
                    input, inputChannels, inputWidth, inputHeight,
                    filterDelta, outputChannels, outputWidth, outputHeight, filterSize, stride, learningRate, useGpu);
            ConvolutionBackwordBiasKernel.INSTANCE.backwordBias(delta, result,
                    outputChannels, outputWidth, outputHeight, biasDelta, learningRate, tempDelta, useGpu);
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
                    filter, outputChannels, outputWidth, outputHeight,
                    filterDelta, biasDelta,
                    filterSize, stride, bias, learningRate, false);
        }
    }

    @Override
    public void prepareBatch(float momentam) {
        IntStream.range(0, filterDelta.length).parallel().forEach(i -> filterDelta[i] = filterDelta[i] * momentam);
        IntStream.range(0, biasDelta.length).parallel().forEach(i -> biasDelta[i] = biasDelta[i] * momentam);
    }

    @Override
    public void joinBatch(int count, float weightDecay, float learningRate) {
        IntStream.range(0, filter.length).parallel().forEach(i -> filter[i] +=  filterDelta[i] / count
                - weightDecay * learningRate * filter[i]);
        IntStream.range(0, bias.length).parallel().forEach(i -> bias[i] += biasDelta[i] / count);
    }

    public float[] getFilter() {
        return filter;
    }

    public float[] getBias() {
        return bias;
    }

    @Override
    public String toString() {
        DoubleSummaryStatistics sum = ConvolutionalNet.summary(filter);
        return String.format("Convolutional:%s filter:%dx%d x%d stride:%d in:%dx%dx%d out %dx%dx%d%n"
                + "Filter %.2f-%.2f ave:%.2f filtertotal:%.2f",
                name, filterSize, filterSize, outputChannels, stride,
                inputWidth, inputHeight, inputChannels, outputWidth, outputHeight, outputChannels,
                sum.getMin(), sum.getMax(), sum.getAverage(), sum.getSum() / inputChannels / outputChannels);
    }

    @Override
    public DoubleSummaryStatistics getWeightStatistics() {
        return ConvolutionalNet.summary(filter);
    }

    @Override
    public DoubleSummaryStatistics getBiasStatistics() {
        return ConvolutionalNet.summary(bias);
    }

}
