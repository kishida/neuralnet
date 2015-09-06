/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.layers;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.DoubleSummaryStatistics;
import java.util.stream.IntStream;
import kishida.cnn.ConvolutionalNet;
import kishida.cnn.activation.ActivationFunction;
import kishida.cnn.kernels.FullyForwardKernel;
import kishida.cnn.util.FloatUtil;
import lombok.Getter;

/**
 *
 * @author naoki
 */
public class FullyConnect extends NeuralLayer implements LerningLayer{
    @Getter
    private float[]weight;
    @Getter
    private float[] bias;
    @Getter
    private float[]weightDelta;
    @Getter
    private float[] biasDelta;

    @JsonProperty
    private int outputSize;
    @Getter
    private int inputSize;
    private int[] dropout;
    @Getter
    private float dropoutRate = 1;
    private float learningRate;
    @Getter
    private boolean useGpu;
    @Getter
    private ActivationFunction activation;

    public FullyConnect(String name, NeuralLayer preLayer, int out, float initBias, float dropoutRate, ActivationFunction activation, float learningRate, boolean useGpu) {
        this(name, preLayer.getOutputSize(), out,initBias, dropoutRate, activation, learningRate, useGpu);
        setPreLayer(preLayer);
    }

    public FullyConnect(String name, int in, int out, float initBias, float dropoutRate, ActivationFunction activation, float learningRate, boolean useGpu) {
        this(name, in, out,
                FloatUtil.createGaussianArray(in * out, 0.01f, ConvolutionalNet.random),
                FloatUtil.createArray(out, initBias),
                dropoutRate, learningRate, activation, useGpu);
    }

    public FullyConnect(String name, int in, int out, float[] weight,
            float[] bias, float dropoutRate, float learningRate,
            ActivationFunction activation, boolean useGpu) {
        this(name, out, weight, bias, 0,
                new float[in * out],  new float[out],
                dropoutRate, learningRate,
                activation, useGpu);
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
            @JsonProperty("learningRate") float learningRate,
            @JsonProperty("activation") ActivationFunction activation,
            @JsonProperty("useGpu") boolean useGpu) {
        super(name);
        this.name = name;
        this.activation = activation;
        this.outputSize = outputSize;
        this.weight = weight;
        this.weightDelta = weightDelta;
        if(bias == null){
            this.bias = FloatUtil.createArray(outputSize, initBias);
        }else{
            this.bias = bias;
        }
        if(biasDelta == null){
            this.biasDelta = new float[outputSize];
        }else{
            this.biasDelta = biasDelta;
        }
        this.learningRate = learningRate;
        this.dropout = IntStream.generate(() -> 1).limit(outputSize).toArray();
        this.dropoutRate = dropoutRate;
        this.useGpu = useGpu;
    }

    public final void setPreLayer(NeuralLayer preLayer) {
        this.preLayer = preLayer;
        this.inputSize = preLayer.getOutputSize();
        if(this.weight == null){
            this.weight = FloatUtil.createGaussianArray(
                    inputSize * outputSize, 0.01f, ConvolutionalNet.random);
        }
        if(this.weightDelta == null){
            this.weightDelta = new float[inputSize * outputSize];
        }
    }

    public void prepareDropout() {
        dropout = ConvolutionalNet.random.doubles(outputSize)
                .mapToInt(d -> d < dropoutRate ? 1 : 0).toArray();
    }

    @Override
    public float[] forward(float[] in) {
        prepareDropout();
        result = new float[outputSize];

        FullyForwardKernel.INSTANCE.forward(outputSize, dropout, in, result, weight, bias, useGpu);
        /*
        IntStream.range(0, out).parallel().filter(j -> dropout[j] == 1).forEach(j -> {
            for (int i = 0; i < in.length; ++i) {
                result[j] += in[i] * weight[i * out + j];
            }
            result[j] += bias[j];
        });*/
        activation.applyAfter(result);
        return result;
    }

    @Override
    public float[] backward(float[] in, float[] delta) {
        /*
        float[][] oldweight = Arrays.stream(weight).parallel()
        .map(row -> Arrays.copyOf(row, row.length))
        .toArray(float[][]::new);*/
        float[] newDelta = new float[in.length];
        float[] diffed = new float[result.length];
        for(int i = 0; i < result.length; ++i){
                diffed[i] = activation.diff(result[i]);
        }
        IntStream.range(0, in.length).parallel().forEach((i) -> {
            for (int j = 0; j < outputSize; ++j) {
                if (dropout[j] != 1) {
                    continue;
                }
                float d = diffed[j] * delta[j];
                newDelta[i] += d *  weight[i * outputSize + j];//in[i] *;
                weightDelta[i * outputSize + j] += d * in[i] * learningRate;
            }
        });
        IntStream.range(0, outputSize).parallel().filter(j -> dropout[j] == 1).forEach(j -> {
            biasDelta[j] += diffed[j] * delta[j] * learningRate;
        });
        return newDelta;
    }

    @Override
    public void prepareBatch(float momentam) {
        IntStream.range(0, weightDelta.length).forEach(i -> weightDelta[i] = weightDelta[i] * momentam);
        IntStream.range(0, biasDelta.length).parallel().forEach(i -> biasDelta[i] = biasDelta[i] * momentam);
    }

    @Override
    public void joinBatch(int count, float weightDecay, float learningRate) {
        IntStream.range(0, weight.length).parallel().forEach(ij -> {
                weight[ij] += weightDelta[ij] / count
                        - weight[ij] * weightDecay * learningRate;
        });
        IntStream.range(0, bias.length).parallel().forEach(i -> {
            bias[i] += biasDelta[i] / count;
        });
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
