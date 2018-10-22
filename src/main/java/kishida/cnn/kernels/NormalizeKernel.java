/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.kernels;

import com.aparapi.Kernel;
import java.util.stream.IntStream;

/**
 *
 * @author naoki
 */
public class NormalizeKernel extends Kernel {
    public static NormalizeKernel INSTANCE = new NormalizeKernel();

    private NormalizeKernel() {
        setExplicit(true);
    }

    @Override
    public void run() {
        int chxy = getGlobalId();
        proc(chxy);
    }

    private void proc(int chxy) {
        int ch = chxy / (inputWidth * inputHeight);
        int x = (chxy % (inputWidth * inputHeight)) / inputHeight;
        int y = chxy % inputHeight;
        //平均
        int count = 0;
        float total = 0;
        for (int i = 0; i < size; ++i) {
            int xx = x + i - size / 2;
            if (xx >= 0 && xx < inputWidth) {
                for (int j = 0; j < size; ++j) {
                    int yy = y + j - size / 2;
                    if (yy >= 0 && yy < inputHeight) {
                        total += input[ch * inputWidth * inputHeight + xx * inputHeight + yy];
                        ++count;
                    }
                }
            }
        }
        float average = total / count;
        //分散
        float variance = 0;
        for (int i = 0; i < size; ++i) {
            int xx = x + i - size / 2;
            if (xx >= 0 && xx < inputWidth) {
                for (int j = 0; j < size; ++j) {
                    int yy = y + j - size / 2;
                    if (yy >= 0 && yy < inputHeight) {
                        float d = input[ch * inputWidth * inputHeight + xx * inputHeight + yy];
                        variance += (d - average) * (d - average);
                    }
                }
            }
        }
        float std = max(threshold, sqrt(variance / count));
        result[chxy] = (input[chxy] - average) / std;
    }
    float[] result;
    float[] input;
    int inputChannels;
    int inputWidth;
    int inputHeight;
    int size;
    float threshold;

    public float[] normalize(float[] input, int inputChannels, int inputWidth, int inputHeight,
            int size, float threshold, float[] result, boolean useGpu) {
        this.input = input;
        this.result = result;
        this.inputChannels = inputChannels;
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.size = size;
        this.threshold = threshold;
        if (useGpu) {
            put(input);
            execute(inputChannels * inputWidth * inputHeight);
            get(result);
        } else {
            IntStream.range(0, inputChannels).parallel().forEach(ch -> {
                for (int x = 0; x < inputWidth; ++x) {
                    for (int y = 0; y < inputHeight; ++y) {
                        proc(ch * inputWidth * inputHeight + x * inputHeight + y);
                    }
                }
            });
        }
        return result;
    }

}
