/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.layers;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.jogamp.opencl.CLBuffer;
import java.nio.FloatBuffer;
import java.util.DoubleSummaryStatistics;
import java.util.stream.IntStream;
import kishida.cnn.activation.ActivationFunction;
import kishida.cnn.kernels.FullyForwardKernel;
import kishida.cnn.opencl.FullyBackwordCL;
import kishida.cnn.opencl.FullyForwardCL;
import kishida.cnn.opencl.OpenCL;
import kishida.cnn.util.FloatUtil;
import lombok.Getter;
import lombok.Setter;

/**
 *
 * @author naoki
 */
public class FullyConnect extends NeuralLayer implements LerningLayer{
    @JsonInclude(JsonInclude.Include.NON_NULL)
    private float[]weight;
    @JsonInclude(JsonInclude.Include.NON_NULL)
    private float[] bias;
    @JsonInclude(JsonInclude.Include.NON_NULL)
    @Getter
    private float[]weightDelta;
    @JsonInclude(JsonInclude.Include.NON_NULL)
    @Getter
    private float[] biasDelta;

    @JsonProperty
    private int outputSize;
    private int inputSize;
    private int[] dropout;
    @Getter
    private float dropoutRate = 1;
    @Getter @Setter
    private boolean useGpu;
    private float[] newDelta;
    private float[] diffed;

    private ActivationFunction activation;
    @Getter
    private float initBias;

    CLBuffer<FloatBuffer> bufWeight;
    CLBuffer<FloatBuffer> bufBias;

    public FullyConnect(String name, int outputSize, float initBias, float dropoutRate, ActivationFunction activation, boolean useGpu) {
        this(name, outputSize, null, null, initBias, null, null, dropoutRate, null, activation, useGpu);
    }

    @JsonCreator
    public FullyConnect(
            @JsonProperty("name") String name,
            @JsonProperty("outputSize") int outputSize,
            @JsonProperty("weight") float[] weight,
            @JsonProperty("bias") float[] bias,
            @JsonProperty("initBias") float initBias,
            @JsonProperty("weightDelta") float[] weightDelta,
            @JsonProperty("biasDelta") float[] biasDelta,
            @JsonProperty("dropoutRate") float dropoutRate,
            @JsonProperty("activation") String activationName,
            @JsonProperty("activationObj") ActivationFunction activation,
            @JsonProperty("useGpu") boolean useGpu) {
        super(name);
        this.name = name;
        if(activation != null){
            this.activation = activation;
        }else{
            try {
                this.activation = (ActivationFunction) FullyConnect.class.forName(
                        ActivationFunction.class.getPackage().getName() + "." + activationName).newInstance();
            } catch (ClassNotFoundException | InstantiationException | IllegalAccessException ex) {
                throw new RuntimeException(ex);
            }
        }
        this.outputSize = outputSize;
        this.weight = weight;
        this.weightDelta = weightDelta;
        this.bias = bias;
        this.initBias = initBias;
        this.biasDelta = biasDelta;
        this.result = new float[outputSize];
        this.dropout = IntStream.generate(() -> 1).limit(outputSize).toArray();
        this.dropoutRate = dropoutRate;
        this.useGpu = useGpu;
        this.diffed = new float[outputSize];
    }

    @Override
    public final void setPreLayer(NeuralLayer preLayer) {
        this.preLayer = preLayer;
        this.inputSize = preLayer.getOutputSize();
        if(this.weight == null){
            this.weight = FloatUtil.createGaussianArray(
                    inputSize * outputSize, 0.01f, parent.getRandom());
        }
        if(this.weightDelta == null){
            this.weightDelta = new float[inputSize * outputSize];
        }
        this.newDelta = new float[inputSize];

        // 実際はコンストラクタで処理できるけど、JSONにデータ出力したくないときのために。
        if(bias == null){
            this.bias = FloatUtil.createArray(outputSize, initBias);
        }
        if(biasDelta == null){
            this.biasDelta = new float[outputSize];
        }
        if(useGpu){
            bufWeight = OpenCL.createReadWriteBuffer(weight);
            bufBias = OpenCL.createReadWriteBuffer(bias);
            OpenCL.getQueue()
                    .putWriteBuffer(bufWeight, false)
                    .putWriteBuffer(bufBias, false);
        }
    }

    public float[] getWeight() {
        if(bufWeight != null){
            OpenCL.getQueue().putReadBuffer(bufWeight, true);
            bufWeight.getBuffer().get(weight).rewind();
        }
        return weight;
    }

    public float[] getBias() {
        if(bufBias != null){
            OpenCL.getQueue().putReadBuffer(bufBias, true);
            bufBias.getBuffer().get(bias).rewind();
        }
        return bias;
    }

    @JsonProperty("activationObj")
    public ActivationFunction getActivation() {
        return activation;
    }

    @JsonProperty("activation")
    public String getActivationName(){
        return activation.getClass().getSimpleName();
    }

    public void prepareDropout() {
        dropout = parent.getRandom().doubles(outputSize)
                .mapToInt(d -> d < dropoutRate ? 1 : 0).toArray();
    }

    @Override
    public float[] forward(float[] in) {
        prepareDropout();
        if(useGpu){
            if(true){
                FullyForwardKernel.INSTANCE.forward(outputSize, dropout, in, result, weight, bias, useGpu);
                activation.applyAfter(result);
            }else{
                FullyForwardCL.INSTANCE.forward(inputSize, outputSize, dropout, in, bufWeight, bufBias, result, activation);
            }
        }else{
            FullyForwardKernel.INSTANCE.forward(outputSize, dropout, in, result, weight, bias, useGpu);
            activation.applyAfter(result);
        }
        return result;
    }

    @Override
    public float[] backward(float[] in, float[] delta) {
        if(useGpu && false){
            FullyBackwordCL.INSTANCE.backword(inputSize, outputSize,
                    dropout, in, delta, result, bufWeight, weightDelta, biasDelta, newDelta,
                    parent.getLearningRate(), activation);
        }else{
            for(int i = 0; i < result.length; ++i){
                    diffed[i] = activation.diff(result[i]);
            }
            IntStream.range(0, in.length).parallel().forEach((i) -> {
                float nd = 0;
                for (int j = 0; j < outputSize; ++j) {
                    if (dropout[j] != 1) {
                        continue;
                    }
                    float d = diffed[j] * delta[j];
                    nd += d *  weight[i * outputSize + j];//in[i] *;
                    weightDelta[i * outputSize + j] += d * in[i] * parent.getLearningRate();
                }
                newDelta[i] = nd;
            });
            IntStream.range(0, outputSize).parallel().filter(j -> dropout[j] == 1).forEach(j -> {
                biasDelta[j] += diffed[j] * delta[j] * parent.getLearningRate();
            });
        }
        return newDelta;
    }

    @Override
    public void prepareBatch() {
        float momentam = parent.getMomentam();
        IntStream.range(0, weightDelta.length).forEach(i -> weightDelta[i] = weightDelta[i] * momentam);
        IntStream.range(0, biasDelta.length).parallel().forEach(i -> biasDelta[i] = biasDelta[i] * momentam);
    }

    @Override
    public void joinBatch() {
        IntStream.range(0, weight.length).parallel().forEach(ij -> {
                weight[ij] += weightDelta[ij] / parent.getMiniBatch()
                        - weight[ij] * parent.getWeightDecay() * parent.getLearningRate();
        });
        IntStream.range(0, bias.length).parallel().forEach(i -> {
            bias[i] += biasDelta[i] / parent.getMiniBatch();
        });
        if(bufWeight != null){
            bufWeight.getBuffer().put(weight).rewind();
            bufBias.getBuffer().put(bias).rewind();
            OpenCL.getQueue()
                    .putWriteBuffer(bufWeight, false)
                    .putWriteBuffer(bufBias, false);
        }
    }

    @Override
    public int getOutputSize() {
        return outputSize;
    }

    @Override
    public String toString() {
        return String.format("%s:Fully connect %d->%d dropout:%.2f", name, inputSize, outputSize, dropoutRate);
    }

    @Override
    public DoubleSummaryStatistics getWeightStatistics() {
        return FloatUtil.summary(weight);
    }

    @Override
    public DoubleSummaryStatistics getBiasStatistics() {
        return FloatUtil.summary(bias);
    }

}
