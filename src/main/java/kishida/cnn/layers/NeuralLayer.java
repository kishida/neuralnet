/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.layers;

import java.util.Objects;
import kishida.cnn.activation.ActivationFunction;

/**
 *
 * @author naoki
 */
public abstract class NeuralLayer {
    String name;
    double[] result;
    NeuralLayer preLayer;
    ActivationFunction activation;

    public NeuralLayer(String name, ActivationFunction activation) {
        this.name = name;
        this.activation = activation;
    }

    public double[] forward() {
        Objects.requireNonNull(preLayer, "preLayer is null on " + name);
        return forward(preLayer.result);
    }

    public double[] backward(double[] delta) {
        return backward(preLayer.result, delta);
    }

    public abstract double[] forward(double[] in);

    public abstract double[] backward(double[] in, double[] delta);

    public void prepareBatch(){
        // do nothing as default
    }
    public void joinBatch(int count){
        // do nothing as default
    }

    public String getName() {
        return name;
    }

    public double[] getResult() {
        return result;
    }

    public abstract int getOutputSize();

}
