/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.layers;

import kishida.cnn.activation.ActivationFunction;

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

    public ImageNeuralLayer(String name, ActivationFunction activation,
            int inputChannels, int inputWidth, int inputHeight,
            int outputChannels, int outputWidth, int outputHeight) {
        super(name, activation);
        this.inputChannels = inputChannels;
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.outputChannels = outputChannels;
        this.outputWidth = outputWidth;
        this.outputHeight = outputHeight;
    }

    public int getOutputChannels() {
        return outputChannels;
    }

    public int getOutputWidth() {
        return outputWidth;
    }

    public int getOutputHeight() {
        return outputHeight;
    }

    @Override
    public int getOutputSize() {
        return outputChannels * outputWidth * outputHeight;
    }

}
