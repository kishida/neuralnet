/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.layers;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.stream.IntStream;
import lombok.Getter;

/**
 *
 * @author naoki
 */
public class MultiNormalizeLayer extends ImageNeuralLayer{
    @Getter
    int size;
    @Getter
    float threshold;
    @Getter
    boolean useGpu;

    @JsonCreator
    public MultiNormalizeLayer(
            @JsonProperty("name") String name,
            @JsonProperty("size") int size,
            @JsonProperty("threshold") float threshold,
            @JsonProperty("useGpu") boolean useGpu) {
        super(name);
        this.size = size;
        this.threshold = threshold;
        this.useGpu = useGpu;
    }

    @Override
    public final void setPreLayer(NeuralLayer preLayer) {
        super.setPreLayer(preLayer);
        outputChannels = inputChannels;
        outputWidth = inputWidth;
        outputHeight = inputHeight;
        result = new float[inputChannels * inputHeight * inputWidth];
    }

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
