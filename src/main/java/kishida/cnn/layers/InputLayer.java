/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.layers;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 *
 * @author naoki
 */
public class InputLayer extends ImageNeuralLayer {

    @JsonCreator
    public InputLayer(
            @JsonProperty("width") int width,
            @JsonProperty("height") int height) {
        super("input", 0, 0, 0, 3, width, height);
    }

    @Override
    public void setPreLayer(NeuralLayer preLayer) {
        // do nothing
    }

    public int getWidth() {
        return super.outputWidth;
    }

    public int getHeight() {
        return super.outputHeight;
    }

    @Override
    public float[] forward(float[] in) {
        this.result = in;
        return result;
    }

    @Override
    public float[] backward(float[] in, float[] delta) {
        // do nothing
        return null;
    }

    public void setInput(float[] input){
        result = input;
    }

    @Override
    public String toString() {
        return String.format("%s:Input size:%dx%d",name, this.outputWidth, this.outputHeight);
    }
}
