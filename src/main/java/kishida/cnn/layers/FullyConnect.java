/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.layers;

import java.util.Arrays;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import kishida.cnn.ConvolutionalNet;
import kishida.cnn.activation.ActivationFunction;

/**
 *
 * @author naoki
 */
public class FullyConnect extends NeuralLayer {
    double[][] weight;
    double[] bias;
    double[][] weightDelta;
    double[] biasDelta;
    int out;
    int in;
    int[] dropout;
    double dropoutRate = 1;
    double localEp;

    public FullyConnect(String name, NeuralLayer preLayer, int out, double initBias, double dropoutRate, ActivationFunction activation, double ep) {
        this(name, preLayer.getOutputSize(), out,initBias, dropoutRate, activation, ep);
        this.preLayer = preLayer;
    }

    public FullyConnect(String name, int in, int out, double initBias, double dropoutRate, ActivationFunction activation, double ep) {
        this(name, in, out,
                Stream.generate(() -> IntStream.range(0, out)// ここをparallelにすると、nextDoubleで同期化されて遅くなる
                        .mapToDouble(d -> ConvolutionalNet.random.nextGaussian() * 0.01)
                        .toArray()).limit(in).toArray(double[][]::new),
                DoubleStream.generate(() -> initBias).limit(out).toArray()
                , dropoutRate, ep, activation);
    }

    public FullyConnect(String name, int in, int out, double[][] weight,
            double[] bias, double dropoutRate, double localEp, ActivationFunction activation) {
        super(name, activation);
        this.name = name;
        this.in = in;
        this.out = out;
        this.weight = weight;
        this.bias = bias;
        this.weightDelta = new double[in][out];
        this.biasDelta = new double[out];
        this.localEp = localEp;
        this.dropout = IntStream.generate(() -> 1).limit(out).toArray();
        this.dropoutRate = dropoutRate;
        this.weight = weight;
        this.bias = bias;
    }

    public void prepareDropout() {
        dropout = ConvolutionalNet.random.doubles(out).mapToInt((d) -> d < dropoutRate ? 1 : 0).toArray();
    }

    @Override
    public double[] forward(double[] in) {
        prepareDropout();
        result = new double[out];
        IntStream.range(0, out).parallel().filter((j) -> dropout[j] == 1).forEach((j) -> {
            for (int i = 0; i < in.length; ++i) {
                result[j] += in[i] * weight[i][j];
            }
            result[j] += bias[j];
        });
        activation.applyAfter(result);
        if (!Arrays.stream(result).allMatch(Double::isFinite)) {
            System.out.println("there is infinite value");
            System.out.println(Arrays.toString(result));
        }
        return result;
    }

    @Override
    public double[] backward(double[] in, double[] delta) {
        /*
        double[][] oldweight = Arrays.stream(weight).parallel()
        .map(row -> Arrays.copyOf(row, row.length))
        .toArray(double[][]::new);*/
        double[] newDelta = new double[in.length];
        double[] diffed = Arrays.stream(result).map(activation::diff).toArray();
        IntStream.range(0, in.length).parallel().forEach((i) -> {
            for (int j = 0; j < out; ++j) {
                if (dropout[j] != 1) {
                    continue;
                }
                double d = diffed[j] * delta[j];
                newDelta[i] += d * in[i] * weight[i][j];
                weightDelta[i][j] += d * in[i] * localEp;
            }
        });
        IntStream.range(0, out).parallel().filter((j) -> dropout[j] == 1).forEach((j) -> {
            biasDelta[j] += diffed[j] * delta[j] * localEp;
        });
        return newDelta;
    }

    @Override
    public void prepareBatch(double momentam) {
        Arrays.stream(weightDelta).parallel().forEach(row -> {
            IntStream.range(0, row.length).forEach(i -> row[i] = row[i] * momentam);
        });
        IntStream.range(0, biasDelta.length).parallel().forEach(i -> biasDelta[i] = biasDelta[i] * momentam);
    }

    @Override
    public void joinBatch(int count) {
        IntStream.range(0, weight.length).parallel().forEach(i -> {
            for(int j = 0; j < weight[i].length; ++j){
                weight[i][j] += weightDelta[i][j] / count;
            }
        });
        IntStream.range(0, bias.length).parallel().forEach(i -> {
            bias[i] += biasDelta[i] / count;
        });
    }

    @Override
    public int getOutputSize() {
        return out;
    }

    public double[] getBias() {
        return bias;
    }

    @Override
    public String toString() {
        return String.format("Fully connect:%s %d->%d dropout:%.2f", name, in, out, dropoutRate);
    }

}
