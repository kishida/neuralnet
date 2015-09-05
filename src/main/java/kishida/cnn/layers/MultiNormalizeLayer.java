/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.layers;

import java.util.stream.IntStream;
import kishida.cnn.activation.LinearFunction;

/**
 *
 * @author naoki
 */
public class MultiNormalizeLayer extends ImageNeuralLayer{
    public MultiNormalizeLayer(String name, int size, float threshold, ImageNeuralLayer preLayer, boolean useGpu) {
        this(name, preLayer.outputChannels, preLayer.outputWidth, preLayer.outputHeight, preLayer, size, threshold, useGpu);
    }


    public MultiNormalizeLayer(String name, int inputChannels, int inputWidth, int inputHeight, ImageNeuralLayer preLayer, int size, float threshold, boolean useGpu) {
        super(name, new LinearFunction(), inputChannels, inputWidth, inputHeight, inputChannels, inputWidth, inputHeight);
        this.preLayer = preLayer;
        this.size = size;
        this.threshold = threshold;
        this.useGpu = useGpu;
        averages = new float[inputWidth * inputHeight];
        rates = new float[inputWidth * inputHeight];
        result = new float[inputChannels * inputHeight * inputWidth];
    }

    float[] averages;
    float[] rates;
    int size;
    float threshold;
    boolean useGpu;

    @Override
    public float[] forward(float[] in) {

        IntStream.range(0, inputWidth).parallel().forEach(x -> {
            for(int y = 0; y < inputHeight; ++y){
                float total = 0;
                int count = 0;
                for(int i = 0; i < size; ++i){
                    int xx = x + i - size / 2;
                    if(xx < 0 || xx >= inputWidth){
                        continue;
                    }
                    for(int j = 0; j < size; ++j){
                        int yy = y + j - size / 2;
                        if(yy < 0 || yy >= inputHeight){
                            continue;
                        }
                        for(int ch = 0; ch < inputChannels; ++ch){
                            total += in[ch * inputHeight * inputWidth + xx * inputHeight + yy];
                            ++count;
                        }
                    }
                }
                float average = total / count;
                float variance = 0;
                for(int i = 0; i < size; ++i){
                    int xx = x + i - size / 2;
                    if(xx < 0 || xx >= inputWidth){
                        continue;
                    }
                    for(int j = 0; j < size; ++j){
                        int yy = y + j - size / 2;
                        if(yy < 0 || yy >= inputHeight){
                            continue;
                        }
                        for(int ch = 0; ch < inputChannels; ++ch){
                            float data = in[ch * inputHeight * inputWidth + xx * inputHeight + yy];
                            variance += (data - average) * (data - average);
                        }
                    }
                }
                float std = Math.max(threshold, (float)Math.sqrt(variance / count));
                averages[x * inputHeight + y] = average;
                rates[x * inputHeight + y] = std;
                for(int ch = 0; ch < inputChannels; ++ch){
                    int pos = ch * inputHeight * inputWidth + x * inputHeight + y;
                    result[pos] = (in[pos] - average) / std;
                }
            }
        });

        return result;
    }

    @Override
    public float[] backward(float[] in, float[] delta) {
        return delta;
    }

    @Override
    public String toString() {
        return String.format("%s:Multi channel normalize size:%dx%d in:%dx%dx%d out %dx%dx%d",
                name, this.size, this.size,
                this.inputWidth, this.inputHeight, this.inputChannels,
                this.outputWidth, this.outputHeight, this.outputChannels);
    }

}
