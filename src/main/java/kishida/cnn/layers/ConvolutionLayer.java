/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.layers;

import com.amd.aparapi.Kernel;
import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.Arrays;
import java.util.DoubleSummaryStatistics;
import java.util.stream.IntStream;
import kishida.cnn.activation.ActivationFunction;
import kishida.cnn.activation.RectifiedLinear;
import kishida.cnn.kernels.ConvolutionBackwordBiasKernel;
import kishida.cnn.kernels.ConvolutionBackwordDeltaKernel;
import kishida.cnn.kernels.ConvolutionBackwordFilterKernel;
import kishida.cnn.kernels.ConvolutionBackwordKernel;
import kishida.cnn.kernels.ConvolutionForwardKernel;
import kishida.cnn.kernels.ConvolutionLocalNormalizationKernel;
import kishida.cnn.util.FloatUtil;
import lombok.Getter;
import lombok.Setter;

/** 畳み込み層 */
public class ConvolutionLayer extends ImageNeuralLayer implements LerningLayer{

    @JsonInclude(JsonInclude.Include.NON_NULL)
    @Getter
    float[] filter;
    @JsonInclude(JsonInclude.Include.NON_NULL)
    @Getter
    float[] bias;
    @JsonInclude(JsonInclude.Include.NON_NULL)
    @Getter
    float[] filterDelta;
    @JsonInclude(JsonInclude.Include.NON_NULL)
    @Getter
    float[] biasDelta;
    @Getter
    int stride;
    @Getter
    int filterSize;
    private ActivationFunction activation;
    @Getter @Setter
    boolean useGpu;
    @Getter
    float initBias;
    float[] tempDelta;

    public ConvolutionLayer(String name,
            int filterCount, int size, int stride, float initBias, boolean useGpu) {
        this(name, size, filterCount, stride, null, null, initBias, null, null, useGpu);
    }

    @JsonCreator
    public ConvolutionLayer(
            @JsonProperty("name") String name,
            @JsonProperty("filterSize") int filterSize,
            @JsonProperty("filterCount") int filterCount,
            @JsonProperty("stride") int stride,
            @JsonProperty("filter") float[] filter,
            @JsonProperty("filterDelta") float[] filterDelta,
            @JsonProperty("initBias") float initBias,
            @JsonProperty("bias") float[] bias,
            @JsonProperty("biasDelta") float[] biasDelta,
            @JsonProperty("useGpu") boolean useGpu) {
        super(name);
        this.filterSize = filterSize;
        this.outputChannels = filterCount;
        this.stride = stride;
        this.filter = filter;
        this.filterDelta = filterDelta;
        this.initBias = initBias;
        this.bias = bias;
        this.biasDelta = biasDelta;
        this.activation = new RectifiedLinear();
        this.useGpu = useGpu;
    }

    public int getFilterCount() {
        return super.getOutputChannels();
    }

    @Override
    public final void setPreLayer(NeuralLayer preLayer) {
        super.setPreLayer(preLayer);
        outputWidth = inputWidth / stride;
        outputHeight = inputHeight / stride;

        if(filter == null){
            this.filter = FloatUtil.createGaussianArray(filterSize * filterSize *
                inputChannels * outputChannels, 0.01f, parent.getRandom());
        }
        if(filterDelta == null){
            this.filterDelta = new float[filter.length];
        }

        // コンストラクタで処理できるけど、JSONデータ出力で省略できるように。
        if(bias == null){
            this.bias = FloatUtil.createArray(outputChannels, initBias);
        }
        if(biasDelta == null){
            this.biasDelta = new float[this.bias.length];
        }

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
                    float sum = FloatUtil.floatSum(sigma);
                    result[ch * outputWidth * outputHeight + xy] = result[ch * outputWidth * outputHeight + xy] /
                            (float)Math.pow(k + a * sum, b);
                }
            }
        }
    }

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
                    filterDelta, outputChannels, outputWidth, outputHeight, filterSize, stride, parent.getLearningRate(), useGpu);
            ConvolutionBackwordBiasKernel.INSTANCE.backwordBias(delta, result,
                    outputChannels, outputWidth, outputHeight, biasDelta, parent.getLearningRate(), tempDelta, useGpu);
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
                    filterSize, stride, bias, parent.getLearningRate(), false);
        }
    }

    @Override
    public void prepareBatch() {
        float momentam = parent.getMomentam();
        IntStream.range(0, filterDelta.length).parallel().forEach(i -> filterDelta[i] = filterDelta[i] * momentam);
        IntStream.range(0, biasDelta.length).parallel().forEach(i -> biasDelta[i] = biasDelta[i] * momentam);
    }

    @Override
    public void joinBatch() {
        float count = parent.getMiniBatch();
        IntStream.range(0, filter.length).parallel().forEach(i -> filter[i] +=  filterDelta[i] / count
                - parent.getWeightDecay() * parent.getLearningRate() * filter[i]);
        IntStream.range(0, bias.length).parallel().forEach(i -> bias[i] += biasDelta[i] / count);
    }

    @Override
    public String toString() {
        DoubleSummaryStatistics sum = FloatUtil.summary(filter);
        return String.format("%s:Convolutional filter:%dx%d x%d stride:%d in:%dx%dx%d out %dx%dx%d",
                name, filterSize, filterSize, outputChannels, stride,
                inputWidth, inputHeight, inputChannels, outputWidth, outputHeight, outputChannels);
    }

    @Override
    public DoubleSummaryStatistics getWeightStatistics() {
        return FloatUtil.summary(filter);
    }

    @Override
    public DoubleSummaryStatistics getBiasStatistics() {
        return FloatUtil.summary(bias);
    }

}
