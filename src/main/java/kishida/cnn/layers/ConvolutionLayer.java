/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.layers;

import com.amd.aparapi.Kernel;
import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.jogamp.opencl.CLBuffer;
import java.nio.FloatBuffer;
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
import kishida.cnn.opencl.ConvolutionBackwordCL;
import kishida.cnn.opencl.ConvolutionForwardCL;
import kishida.cnn.opencl.OpenCL;
import kishida.cnn.util.FloatUtil;
import lombok.Getter;
import lombok.Setter;

/** 畳み込み層 */
public class ConvolutionLayer extends ImageNeuralLayer implements LerningLayer, FullGpuEnabled{

    @JsonInclude(JsonInclude.Include.NON_NULL)
    float[] filter;

    @JsonInclude(JsonInclude.Include.NON_NULL)
    float[] bias;
    @JsonInclude(JsonInclude.Include.NON_NULL)
    float[] filterDelta;
    @JsonInclude(JsonInclude.Include.NON_NULL)
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
    float[] newDelta;

    CLBuffer<FloatBuffer> bufFilter;
    CLBuffer<FloatBuffer> bufBias;
    CLBuffer<FloatBuffer> bufFilterDelta;
    CLBuffer<FloatBuffer> bufBiasDelta;

    @JsonIgnore
    @Getter
    CLBuffer<FloatBuffer> bufResult;

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
        this.newDelta = new float[inputChannels * inputWidth * inputHeight];

        if(true){
            this.bufFilter = OpenCL.createReadWriteBuffer(filter);
            this.bufBias = OpenCL.createReadWriteBuffer(bias);
            this.bufFilterDelta = OpenCL.createReadWriteBuffer(filterDelta);
            this.bufBiasDelta = OpenCL.createReadWriteBuffer(biasDelta);
            this.bufResult = OpenCL.createReadWriteBuffer(result.length);
            OpenCL.getQueue()
                    .putWriteBuffer(bufFilter, false)
                    .putWriteBuffer(bufBias, false)
                    .putWriteBuffer(bufFilterDelta, false)
                    .putWriteBuffer(bufBiasDelta, false);
        }
    }

    public float[] getFilter() {
        if(bufFilter != null){
            OpenCL.getQueue().putReadBuffer(bufFilter, true);
            bufFilter.getBuffer().get(filter).rewind();
        }
        return filter;
    }

    public float[] getBias() {
        if(bufBias != null){
            OpenCL.getQueue().putReadBuffer(bufBias, true);
            bufBias.getBuffer().get(bias).rewind();
        }
        return bias;
    }

    public float[] getFilterDelta() {
        if(bufFilterDelta != null){
            OpenCL.getQueue().putReadBuffer(bufFilterDelta, true);
            bufFilterDelta.getBuffer().get(filterDelta).rewind();
        }
        return filterDelta;
    }

    public float[] getBiasDelta() {
        if(bufBiasDelta != null){
            OpenCL.getQueue().putReadBuffer(bufBiasDelta, true);
            bufBiasDelta.getBuffer().get(biasDelta).rewind();
        }
        return biasDelta;
    }

    @Override
    public float[] getResult() {
        if(bufResult != null){
            OpenCL.getQueue().putReadBuffer(bufResult, true);
            bufResult.getBuffer().get(result).rewind();
        }
        return result;
    }

    /** 畳み込みフィルタを適用する */
    @Override
    public float[] forward(float[] img) {
        if(true){
            if(false){
                // aparapi
                result = ConvolutionForwardKernel.INSTANCE.forward(img, inputChannels, inputWidth, inputHeight,
                        filter, outputChannels, outputWidth, outputHeight, result, filterSize, stride, bias, activation, false);
                //localNormalization(result);
                ConvolutionLocalNormalizationKernel.INSTANCE.localNormalization(result,
                        outputChannels, outputWidth, outputHeight, false);
            } else{
                // JOCL
                if(true){
                    ConvolutionForwardCL.INSTANCE.forward(img, inputChannels, inputWidth, inputHeight,
                            bufFilter, outputChannels, outputWidth, outputHeight, result, filterSize, stride, bufBias);
                } else {
                    ConvolutionForwardCL.INSTANCE.forward(img, inputChannels, inputWidth, inputHeight,
                            filter, outputChannels, outputWidth, outputHeight, result, filterSize, stride, bias);
                }
            }
        }else {
            //CPU
            result = ConvolutionForwardKernel.INSTANCE.forward(img, inputChannels, inputWidth, inputHeight,
                    filter, outputChannels, outputWidth, outputHeight, result, filterSize, stride, bias, activation, false);
            //localNormalization(result);
            ConvolutionLocalNormalizationKernel.INSTANCE.localNormalization(result,
                    outputChannels, outputWidth, outputHeight, false);
        }
        return result;
    }

    @Override
    public void forward(CLBuffer<FloatBuffer> input) {
        ConvolutionForwardCL.INSTANCE.forward(input,
                inputChannels, inputWidth, inputHeight,
                bufFilter, outputChannels, outputWidth, outputHeight,
                bufResult, filterSize, stride, bufBias);
    }

    /** 畳み込み層の学習 */
    @Override
    public float[] backward(float[] input, float[] delta) {
        if (useGpu) {
            // GPUバージョン
            if(false){
                // aparapi
                ConvolutionBackwordDeltaKernel.INSTANCE.backword(delta, result,
                        inputChannels, inputWidth, inputHeight,
                        filter, outputChannels, outputWidth, outputHeight,
                        filterSize, stride, newDelta, useGpu);
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
            }else{
                // JOCL
                if(true){
                    ConvolutionBackwordCL.INSTANCE.backward(
                            delta, result, input,
                            inputChannels, inputWidth, inputHeight,
                            bufFilter, outputChannels, outputWidth, outputHeight,
                            bufFilterDelta, bufBiasDelta, filterSize, stride, newDelta, parent.getLearningRate());
                }else{
                    ConvolutionBackwordCL.INSTANCE.backward(
                            delta, result, input,
                            inputChannels, inputWidth, inputHeight,
                            filter, outputChannels, outputWidth, outputHeight,
                            filterDelta, biasDelta, filterSize, stride, newDelta, parent.getLearningRate());
                }
            }
            return newDelta;
        } else {
            // CPUバージョン
            return ConvolutionBackwordKernel.INSTANCE.backward(delta, result,
                    input, inputChannels, inputWidth, inputHeight,
                    filter, outputChannels, outputWidth, outputHeight,
                    filterDelta, biasDelta,
                    filterSize, stride, parent.getLearningRate(), false);
        }
    }

    @Override
    public void prepareBatch() {
        if(useGpu){
            ConvolutionBackwordCL.INSTANCE.prepare(parent.getMomentam(),
                    filterDelta.length, biasDelta.length, bufFilterDelta, bufBiasDelta);
        }else{
            float momentam = parent.getMomentam();
            IntStream.range(0, filterDelta.length).parallel().forEach(i -> filterDelta[i] = filterDelta[i] * momentam);
            IntStream.range(0, biasDelta.length).parallel().forEach(i -> biasDelta[i] = biasDelta[i] * momentam);
        }
    }

    @Override
    public void joinBatch() {
        if(useGpu){
            ConvolutionBackwordCL.INSTANCE.join(
                    parent.getWeightDecay(), parent.getLearningRate(),
                    filter.length, bias.length,
                    parent.getMiniBatch(),
                    bufFilter, bufFilterDelta, bufBias, bufBiasDelta);
            /*
            bufFilter.getBuffer().put(filter).rewind();
            bufBias.getBuffer().put(bias).rewind();
            OpenCL.getQueue()
                    .putWriteBuffer(bufFilter, false)
                    .putWriteBuffer(bufBias, false);
                    */
        }else{
            float count = parent.getMiniBatch();
            IntStream.range(0, filter.length).parallel().forEach(
                    i -> filter[i] +=  filterDelta[i] / count
                    - parent.getWeightDecay() * parent.getLearningRate() * filter[i]);
            IntStream.range(0, bias.length).parallel().forEach(
                    i -> bias[i] += biasDelta[i] / count);
        }
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
