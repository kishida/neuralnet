/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.layers;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.jogamp.opencl.CLBuffer;
import java.nio.FloatBuffer;
import kishida.cnn.opencl.OpenCL;
import lombok.Getter;

/**
 *
 * @author naoki
 */
public class InputLayer extends ImageNeuralLayer implements FullGpuEnabled {
    @JsonIgnore
    @Getter
    CLBuffer<FloatBuffer> bufResult;

    public InputLayer(int width, int height) {
        this("input", width, height);
    }

    @JsonCreator
    public InputLayer(
            @JsonProperty("input") String input,
            @JsonProperty("width") int width,
            @JsonProperty("height") int height) {
        super("input", 0, 0, 0, 3, width, height);
        bufResult = OpenCL.createWriteBuffer(outputChannels * outputWidth * outputHeight);
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
    public boolean isUseGpu() {
        return false;
    }

    @Override
    public float[] forward(float[] in) {
        this.result = in;
        bufResult.getBuffer().put(result);
        OpenCL.getQueue()
                .putWriteBuffer(bufResult, false);
        return result;
    }

    @Override
    public void forward(CLBuffer<FloatBuffer> input) {
        // do nothing
    }

    @Override
    public float[] backward(float[] in, float[] delta) {
        // do nothing
        return null;
    }

    @Override
    public CLBuffer<FloatBuffer> backwardBuf(CLBuffer<FloatBuffer> bufInput, CLBuffer<FloatBuffer> bufDelta) {
        // do nothing
        return null;
    }

    @Override
    public CLBuffer<FloatBuffer> backwardBuf(CLBuffer<FloatBuffer> bufInput, float[] delta) {
        // do nothing
        return null;
    }

    public void setInput(float[] input){
        result = input;
        bufResult.getBuffer().put(result).rewind();
        OpenCL.getQueue()
                .putWriteBuffer(bufResult, false);
    }

    @Override
    public String toString() {
        return String.format("%s:Input size:%dx%d",name, this.outputWidth, this.outputHeight);
    }
}
