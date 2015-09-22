/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.layers;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.Arrays;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import kishida.cnn.opencl.MaxPoolingCL;
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
        if(false){
            MaxPoolingCL.INSTANCE.forward(inputChannels, inputWidth, inputHeight,
                    outputWidth, outputHeight, size, stride, data, result);
        }else{
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
        }
        return result;
    }

    @Override
    public float[] backward(float[] in, float[] delta) {
        return backward(in, delta, false);
    }
    public float[] backward(float[] in, float[] delta, boolean gpu) {
        if(gpu){
            MaxPoolingCL.INSTANCE.backword(inputChannels, inputWidth, inputHeight,
                    outputWidth, outputHeight, size, stride, in, delta, newDelta);
        }else{
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
        }
        return newDelta;
    }

    public static void main(String[] args) {
        InputLayer input = new InputLayer(6, 6);
        MaxPoolingLayer pool = new MaxPoolingLayer("test_pool", 3, 2);
        pool.setPreLayer(input);
        for(int i = 0; i < pool.newDelta.length; ++i){
            pool.newDelta[i] = 3;
        }
        float[] in = new float[6 * 6 * 3];
        for(int i = 0; i < in.length; ++i){
            in[i] = i;
        }
        float[] delta = {
            0.01f, 0.02f, 0.03f, 0.05f, 0.07f, 0.11f, 0.13f, 0.17f, 0.19f,
            1, 2, 3, 5, 7, 11, 13, 17, 19,
            1, 2, 3, 5, 7, 11, 13, 17, 19};
        float[] newDeltaGpu = pool.backward(in, delta, true);
        float[] newDeltaCpu = pool.backward(in, delta, false);
        System.out.println(Arrays.equals(newDeltaCpu, newDeltaGpu));
        IntStream.range(0, newDeltaGpu.length / 6 / 3).forEach(i -> {
            System.out.println(IntStream.range(0, 6)
                    .map(n -> n + i * 6)
                    .mapToObj(n -> "" + (int)(newDeltaGpu[n]*100))
                    .collect(Collectors.joining(",")));
        });

        Random r = new Random();
        for(int t = 0; t < 1000; ++t){
        for(int i = 0; i < in.length; ++i){
            in[i] = r.nextFloat();
        }
        for(int i = 0; i < delta.length; ++i){
            delta[i] = r.nextFloat();
        }
        float[] newDeltaGpu2 = pool.backward(in, delta, true);
        float[] newDeltaCpu2 = pool.backward(in, delta, false);
        if(!Arrays.equals(newDeltaCpu2, newDeltaGpu2)){
            System.out.println("wrong");
        };
        }
    }

    @Override
    public String toString() {
        return String.format("%s:Max pooling size:%dx%d stride:%d in:%dx%dx%d out %dx%dx%d",
                name, this.size, this.size, this.stride,
                this.inputWidth, this.inputHeight, this.inputChannels,
                this.outputWidth, this.outputHeight, this.outputChannels);
    }

}
