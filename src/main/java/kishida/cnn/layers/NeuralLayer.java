/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.layers;

import java.util.DoubleSummaryStatistics;
import java.util.Objects;
import kishida.cnn.ConvolutionalNet;
import kishida.cnn.activation.ActivationFunction;

/**
 *
 * @author naoki
 */
public abstract class NeuralLayer {
    String name;
    float[] result;
    NeuralLayer preLayer;
    ActivationFunction activation;

    public NeuralLayer(String name, ActivationFunction activation) {
        this.name = name;
        this.activation = activation;
    }

    public float[] forward() {
        Objects.requireNonNull(preLayer, "preLayer is null on " + name);
        return forward(preLayer.result);
    }

    public float[] backward(float[] delta) {
        return backward(preLayer.result, delta);
    }

    public abstract float[] forward(float[] in);

    public abstract float[] backward(float[] in, float[] delta);

    public void prepareBatch(float momentam){
        // do nothing as default
    }
    public void joinBatch(int count, float weightDecay, float learningRate){
        // do nothing as default
    }

    public String getName() {
        return name;
    }

    public float[] getResult() {
        return result;
    }

    public abstract int getOutputSize();
    public DoubleSummaryStatistics getResultStatistics(){
        return ConvolutionalNet.summary(result);
    }

}
