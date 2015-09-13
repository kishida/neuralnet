/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.layers;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.Arrays;
import java.util.stream.IntStream;
import lombok.Getter;

/**
 *
 * @author naoki
 */
public class MaxPoolingLayer extends ImageNeuralLayer {
    @Getter
    int size;
    @Getter
    int stride;
    float[] newDelta;

    @JsonCreator
    public MaxPoolingLayer(
            @JsonProperty("name") String name,
            @JsonProperty("size") int size,
            @JsonProperty("stride") int stride) {
        super(name);
        this.size = size;
        this.stride = stride;
    }

    @Override
    public final void setPreLayer(NeuralLayer preLayer) {
        super.setPreLayer(preLayer);
        outputChannels = inputChannels;
        outputWidth = inputWidth / stride;
        outputHeight = inputHeight / stride;
        result = new float[outputChannels * outputWidth * outputHeight];
        newDelta = new float[inputChannels * inputWidth * inputHeight];
    }

    /** プーリング(max) */
    @Override
    public float[] forward(float[] data) {
        IntStream.range(0, inputChannels).parallel().forEach(ch -> {
            for (int x = 0; x < outputWidth; ++x) {
                for (int y = 0; y < outputHeight; ++y) {
                    float max = Float.NEGATIVE_INFINITY;
                    for (int i = 0; i < size; ++i) {
                        int xx = x * stride + i - size / 2;
                        if (xx < 0 || xx >= inputWidth) {
                            continue;
                        }
                        for (int j = 0; j < size; ++j) {
                            int yy = y * stride + j - size / 2;
                            if (yy < 0 || yy >= inputHeight) {
                                continue;
                            }
                            float d = data[ch * inputWidth * inputHeight + xx * inputHeight + yy];
                            if (max < d) {
                                max = d;
                            }
                        }
                    }
                    result[ch * outputWidth * outputHeight + x * outputHeight + y] = max;
                }
            }
        });
        return result;
    }

    @Override
    public float[] backward(float[] in, float[] delta) {
        Arrays.fill(newDelta, 0);
        IntStream.range(0, inputChannels).parallel().forEach(ch -> {
            for (int x = 0; x < outputWidth; ++x) {
                for (int y = 0; y < outputHeight; ++y) {
                    float max = Float.NEGATIVE_INFINITY;
                    int maxX = 0;
                    int maxY = 0;
                    for (int i = 0; i < size; ++i) {
                        int xx = x * stride + i - size / 2;
                        if (xx < 0 || xx >= inputWidth) {
                            continue;
                        }
                        for (int j = 0; j < size; ++j) {
                            int yy = y * stride + j - size / 2;
                            if (yy < 0 || yy >= inputHeight) {
                                continue;
                            }
                            float d = in[ch * inputWidth * inputHeight + xx * inputWidth + yy];
                            if (max < d) {
                                max = d;
                                maxX = xx;
                                maxY = yy;
                            }
                        }
                    }
                    int chxy = ch * outputWidth * outputHeight + x * outputHeight + y;
                    newDelta[ch * inputWidth * inputHeight + maxX * inputHeight + maxY] +=
                             delta[chxy];
                }
            }
        });
        return newDelta;
    }

    @Override
    public String toString() {
        return String.format("%s:Max pooling size:%dx%d stride:%d in:%dx%dx%d out %dx%dx%d",
                name, this.size, this.size, this.stride,
                this.inputWidth, this.inputHeight, this.inputChannels,
                this.outputWidth, this.outputHeight, this.outputChannels);
    }

}
