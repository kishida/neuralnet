/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.layers;

import kishida.cnn.activation.LinearFunction;

/**
 *
 * @author naoki
 */
public class InputLayer extends ImageNeuralLayer {

    public InputLayer(int width, int height) {
        super("入力", new LinearFunction(), 0, 0, 0, 3, width, height);
    }

    @Override
    public double[] forward(double[] in) {
        this.result = in;
        return result;
    }

    @Override
    public double[] backward(double[] in, double[] delta) {
        // do nothing
        return null;
    }

    public void setInput(double[] input){
        result = input;
    }
}
