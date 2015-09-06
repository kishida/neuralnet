/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.layers;

import com.fasterxml.jackson.annotation.JsonIgnore;
import java.util.Objects;

/**
 *
 * @author naoki
 */
public abstract class ImageNeuralLayer extends NeuralLayer {
    int inputChannels;
    int inputWidth;
    int inputHeight;
    int outputChannels;
    int outputWidth;
    int outputHeight;

    public ImageNeuralLayer(String name) {
        this(name, 0, 0, 0, 0, 0, 0);
    }

    public ImageNeuralLayer(String name,
            int inputChannels, int inputWidth, int inputHeight,
            int outputChannels, int outputWidth, int outputHeight) {
        super(name);
        this.inputChannels = inputChannels;
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.outputChannels = outputChannels;
        this.outputWidth = outputWidth;
        this.outputHeight = outputHeight;
    }

    public void setPreLayer(NeuralLayer preLayer) {
        Objects.requireNonNull(preLayer, "need preLayer");
        if(!(preLayer instanceof ImageNeuralLayer)){
            throw new IllegalArgumentException("Need ImageNeuralLayer instead of " +
                    preLayer.getClass());
        }
        this.preLayer = preLayer;
        ImageNeuralLayer imageLayer = (ImageNeuralLayer) preLayer;
        this.inputChannels = imageLayer.outputChannels;
        this.inputWidth = imageLayer.outputWidth;
        this.inputHeight = imageLayer.outputHeight;
    }

    @JsonIgnore
    public int getOutputChannels() {
        return outputChannels;
    }

    @JsonIgnore
    public int getOutputWidth() {
        return outputWidth;
    }

    @JsonIgnore
    public int getOutputHeight() {
        return outputHeight;
    }

    @Override
    public int getOutputSize() {
        return outputChannels * outputWidth * outputHeight;
    }

}
