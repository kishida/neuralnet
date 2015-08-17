package kishida.imagefiltering;

import com.amd.aparapi.Kernel;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.GridLayout;
import java.awt.Image;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.DoubleSummaryStatistics;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.function.DoubleToIntFunction;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTabbedPane;

/**
 *
 * @author naoki
 */
public class ConvolutionalNet {
    static final double ep = 0.00001;
    static Random random = new Random();
    static final boolean USE_GPU1 = true;
    static final boolean USE_GPU2 = false;
    static final int FILTER_1ST = 16;
    static final int FILTER_2ND = 12;
    //static final int FILTER_1ST = 48;
    //static final int FILTER_2ND = 96;

    static class Img{

        public Img(Path filename, boolean inverse, int x, int y) {
            this.filename = filename;
            this.inverse = inverse;
            this.x = x;
            this.y = y;
        }
        Path filename;
        boolean inverse;
        int x;
        int y;
    }
    
    /** 活性化関数 */
    interface ActivationFunction{
        double apply(double value);
        /** 微分 */
        double diff(double value);
    }
    static class LinearFunction implements ActivationFunction{

        @Override
        public double apply(double value) {
            return value;
        }

        @Override
        public double diff(double value) {
            return 1;
        }
        
    }
    /** 正規化線形関数 */
    static class RetifierdLinear implements ActivationFunction{

        @Override
        public double apply(double value) {
            return value > 0 ? value : 0;
        }

        @Override
        public double diff(double value) {
            return value > 0 ? 1 : 0;
        }
        
    }
    /** ロジスティックシグモイド関数 */
    static class LogisticFunction implements ActivationFunction{

        @Override
        public double apply(double value) {
            return 1 / (1 + Math.exp(-value));
        }

        @Override
        public double diff(double value) {
            return value * (1 - value);
        }
        
    }
    
    /** ソフトマックス */
    static class SoftMaxFunction implements ActivationFunction{

        @Override
        public double apply(double value) {
            return value;//結果をフラット化してから実装する
        }

        @Override
        public double diff(double value) {
            return value;// * (1 - value);
        }
        
    }
    
    static abstract class ImageNeuralLayer{
        String name;
        double[] result;
        ImageNeuralLayer preLayer;
        ActivationFunction activation;
        int inputChannels;
        int inputWidth;
        int inputHeight;
        int outputChannels;
        int outputWidth;
        int outputHeight;

        public ImageNeuralLayer(String name, ActivationFunction activation,
                int inputChannels, int inputWidth, int inputHeight, 
                int outputChannels, int outputWidth, int outputHeight) {
            this.name = name;
            this.activation = activation;
            this.inputChannels = inputChannels;
            this.inputWidth = inputWidth;
            this.inputHeight = inputHeight;
            this.outputChannels = outputChannels;
            this.outputWidth = outputWidth;
            this.outputHeight = outputHeight;
        }

        
        double[] forward(){
            return forward(preLayer.result);
        }
        double[] backword(double[] delta){
            return backword(preLayer.result, delta, activation);
        }
        
        abstract double[] forward(double[] in);
        abstract double[] backword(double[] in, double[] delta, ActivationFunction activation);

        public String getName() {
            return name;
        }

        public double[] getResult() {
            return result;
        }
        
    }
    
    static class InputFilter extends ImageNeuralLayer{

        public InputFilter(int width, int height) {
            super("入力", new LinearFunction(), 0, 0, 0, 3, width, height);
        }

        @Override
        double[] forward(double[] in) {
            this.result = in;
            return result;
        }

        @Override
        double[] backword(double[] in, double[] delta, ActivationFunction act) {
            // do nothing
            return null;
        }
        
    }
    static class ConvolutionForwardKernel extends Kernel{

        public ConvolutionForwardKernel() {
            setExplicit(true);
        }
        
        @Override
        public void run() {
            int fixy = getGlobalId();
            proc(fixy);
        }
        
        private void proc(int fixy){
            int fi = fixy / (outputHeight * outputWidth);
            int x = (fixy % (outputHeight * outputWidth)) / outputHeight;
            int y = fixy % outputHeight;
            double r = 0; // 毎回resultを足すよりもまとめて足したほうがGPUの場合に速くなる。
            for(int ch = 0; ch < inputChannels; ++ch){
                for(int i = 0; i < filterSize; ++i){
                    int xx = x * stride + i - filterSize / 2;
                    if(xx >= 0 && xx < inputWidth){
                        for(int j = 0; j < filterSize; ++j){
                            int yy = y * stride + j - filterSize / 2;
                            if(yy >= 0 && yy < inputHeight){
                                r += input[ch * inputWidth * inputHeight + xx * inputHeight + yy] * 
                                        filter[fi * inputChannels * filterSize * filterSize + 
                                            ch * filterSize * filterSize + i * filterSize + j];
                            }
                        }
                    }
                }
            }
            result[fixy] += r + bias[fi];
        }
        
        double[] result;
        double[] input;
        int inputChannels;
        int inputWidth;
        int inputHeight;
        double[] filter;
        int outputChannels;
        int outputWidth;
        int outputHeight;
        int filterSize;
        int stride;
        double[] bias;
        public double[] forward(double[] input, int inputChannels, int inputWidth, int inputHeight, 
                double[] filter, int outputChannels, int outputWidth, int outputHeight, int filterSize, int stride,
                double[] bias, ActivationFunction activation, boolean useGpu){
            this.input = input;
            this.inputChannels = inputChannels;
            this.inputWidth = inputWidth;
            this.inputHeight = inputHeight;
            this.filter = filter;
            this.outputChannels = outputChannels;
            this.outputWidth = outputWidth;
            this.outputHeight = outputHeight;
            this.filterSize = filterSize;
            this.stride = stride;
            this.bias = bias;
            
            result = new double[outputChannels * outputWidth * outputHeight];
            if(useGpu){
                put(input);
                put(filter);
                put(bias);
                execute( outputChannels * outputWidth * outputHeight);
                get(result);
            }else{
                IntStream.range(0, outputChannels * outputWidth * outputHeight).parallel().forEach(fxy -> {
                    proc(fxy);
                });
            }
            IntStream.range(0, result.length).parallel().forEach(fi ->{
                result[fi] = activation.apply(result[fi]);
            });
            return result;            
        }
    }
    static ConvolutionForwardKernel convolutionForwardKernel = new ConvolutionForwardKernel();
    static ConvolutionBackwordKernel convolutionBackwordKernel = new ConvolutionBackwordKernel();
    static ConvolutionBackwordDeltaKernel convolutionBackwordDeltaKernel = new ConvolutionBackwordDeltaKernel();
    static ConvolutionBackwordBiasKernel convolutionBackwordBiasKernel = new ConvolutionBackwordBiasKernel();
    static ConvolutionBackwordFilterKernel convolutionBackwordFilterKernel = new ConvolutionBackwordFilterKernel();
    
    static class ConvolutionBackwordBiasKernel extends Kernel{

        public ConvolutionBackwordBiasKernel() {
            setExplicit(true);
        }
        
        @Override
        public void run() {
            int fxy = getGlobalId();
            proc(fxy);
        }
                double[] result;
        double[] delta;
        double localEp;
        double[] biasDelta;

        private void proc(int fxy){
            double d = (result[fxy] > 0 ? 1 : 0) * delta[fxy];
            biasDelta[fxy] = localEp * d;
        }

        void backwordBias(double[] delta, double[] result, 
                 int outputChannels, int outputWidth, int outputHeight, 
                double[] bias, boolean useGpu){
            this.delta = delta;
            this.result = result;
            this.localEp = ep / (outputWidth * outputHeight);
            this.biasDelta = Arrays.copyOf(result, result.length);
            
            if(useGpu){
                put(delta);
                put(result);
                execute( outputChannels * outputWidth * outputHeight);
                get(biasDelta);
                IntStream.range(0, outputChannels).parallel().forEach(f -> {
                    for(int xy = 0; xy < outputWidth * outputHeight; ++xy){
                        bias[f] += biasDelta[f * outputWidth * outputHeight + xy];
                    }
                });
            }else{
                IntStream.range(0, outputChannels).parallel().forEach(f -> {
                    for(int xy = 0; xy <  outputWidth * outputHeight; ++xy){
                        proc(f * outputWidth * outputHeight + xy);
                        bias[f] += biasDelta[f * outputWidth * outputHeight + xy];
                    }
                });
            }
        }
            
    }


    static class ConvolutionBackwordDeltaKernel extends Kernel{

        public ConvolutionBackwordDeltaKernel() {
            setExplicit(true);
        }

        @Override
        public void run() {
            int fxy = getGlobalId();
            proc(fxy);
        }
        private void proc(int chxxyy){
            int ch = chxxyy / (inputWidth * inputHeight);
            int xx = (chxxyy % (inputWidth * inputHeight)) / inputHeight;
            int yy = chxxyy % inputHeight;
            int sizeHalf = filterSize / 2;
            double tempDelta = 0;
            for(int f = 0; f < outputChannels; ++f){
                for(int i = 0; i < filterSize; ++i){
                    int x = (xx  - i + sizeHalf) / stride;
                    if(xx == x * stride + i - sizeHalf
                            && x >= 0 && x < outputWidth){
                        for(int j = 0; j < filterSize; ++j){
                            int y = (yy - j + sizeHalf) / stride;
                            if(yy == y * stride + j -sizeHalf
                                    && y >= 0 && y < outputHeight){
                                int fxy = f * outputWidth * outputHeight + x * outputHeight + y;
                                double d = (result[fxy] > 0 ? 1 : 0) * delta[fxy];
                                tempDelta += d * filter[f * inputChannels * filterSize * filterSize + 
                                            ch * filterSize * filterSize + i * filterSize + j];

                            }
                        }
                    }
                }
            }
            newDelta[chxxyy] = tempDelta;
        }
        double[] result;
        int inputChannels;
        int inputWidth;
        int inputHeight;
        double[] filter;
        int outputChannels;
        int outputWidth;
        int outputHeight;
        int filterSize;
        int stride;
        double[] delta;
        double[] newDelta;
        double[] backword(double[] delta, double[] result, int inputChannels, int inputWidth, int inputHeight, 
                double[] filter, int outputChannels, int outputWidth, int outputHeight, int filterSize, int stride,
                boolean useGpu){
            this.delta = delta;
            this.inputChannels = inputChannels;
            this.inputWidth = inputWidth;
            this.inputHeight = inputHeight;
            this.filter = filter;
            this.outputChannels = outputChannels;
            this.outputWidth = outputWidth;
            this.outputHeight = outputHeight;
            this.filterSize = filterSize;
            this.stride = stride;
            this.result = result;
            this.newDelta = new double[inputChannels * inputWidth * inputHeight];
            
            if(useGpu){
                put(filter);
                put(delta);
                put(result);
                execute( inputChannels * inputWidth * inputHeight);
                get(newDelta);
            }else{
                IntStream.range(0, inputChannels * inputWidth).parallel().forEach(chx -> {
                    for(int y = 0; y < inputHeight; ++y){
                        proc(chx * inputHeight + y);
                    }
                });
            }
            return newDelta;
        }
        

    }    


    static class ConvolutionBackwordFilterKernel extends Kernel{

        public ConvolutionBackwordFilterKernel() {
            setExplicit(true);
        }

        @Override
        public void run() {
            int fchij = getGlobalId();
            proc(fchij);
        }
        private void proc(int fchij){
            int f = fchij / (inputChannels * filterSize * filterSize);
            int ch = (fchij % (inputChannels * filterSize * filterSize)) / (filterSize * filterSize);
            int i = (fchij % (filterSize * filterSize)) / filterSize;
            int j = fchij % filterSize;
            
            double df = 0;
            for(int x = 0; x < outputWidth; ++x){
                for(int y = 0; y < outputHeight; ++y){
                    int fxy = f * outputWidth * outputHeight + x * outputHeight + y;
                    double d = (result[fxy] > 0 ? 1 : 0) * delta[fxy];
                    
                    int xx = x * stride + i - filterSize / 2;
                    if(xx >= 0 && xx < inputWidth){
                            int yy = y * stride + j - filterSize / 2;
                            if(yy >= 0 && yy < inputHeight){
                                df += d * localEp * input[ch * inputWidth * inputHeight + xx * inputHeight + yy];
                            }
                    }
                }
            
            }
            filter[fchij] += df;
        }
        double[] input;
        double[] result;
        double[] delta;
        int inputChannels;
        int inputWidth;
        int inputHeight;
        double[] filter;
        int outputChannels;
        int outputWidth;
        int outputHeight;
        int filterSize;
        int stride;
        double localEp;

        void backword(double[] delta, double[] result, double[] input, int inputChannels, int inputWidth, int inputHeight, 
                double[] filter, int outputChannels, int outputWidth, int outputHeight, int filterSize, int stride,
                boolean useGpu){
            this.input = input;
            this.delta = delta;
            this.inputChannels = inputChannels;
            this.inputWidth = inputWidth;
            this.inputHeight = inputHeight;
            this.filter = filter;
            this.outputChannels = outputChannels;
            this.outputWidth = outputWidth;
            this.outputHeight = outputHeight;
            this.filterSize = filterSize;
            this.stride = stride;
            this.result = result;
            this.localEp = ep / (outputWidth * outputHeight);
            
            if(useGpu){
                put(delta);
                put(filter);
                put(input);
                put(result);
                execute( outputChannels * inputChannels * filterSize * filterSize);
                get(filter);
            }else{
                IntStream.range(0, outputChannels).parallel().forEach(f -> {
                    for(int chij = 0; chij <  inputChannels * filterSize * filterSize; ++chij){
                        proc(f * inputChannels * filterSize * filterSize + chij);
                    }
                });
            }
        }
        

    }
        
    static class ConvolutionBackwordKernel extends Kernel{

        public ConvolutionBackwordKernel() {
            setExplicit(true);
        }

        @Override
        public void run() {
            int fxy = getGlobalId();
            proc(fxy);
        }
        private void proc(int fxy){
            double d = (result[fxy] > 0 ? 1 : 0) * delta[fxy];
            int f = fxy / (outputWidth * outputHeight);
            int x = (fxy % (outputWidth * outputHeight)) / outputHeight;
            int y = fxy % outputHeight;
            for(int ch = 0; ch < inputChannels; ++ch){
                for(int i = 0; i < filterSize; ++i){
                    int xx = x * stride + i - filterSize / 2;
                    if(xx >= 0 && xx < inputWidth){
                        for(int j = 0; j < filterSize; ++j){
                            int yy = y * stride + j - filterSize / 2;
                            if(yy >= 0 && yy < inputHeight){
                                tempDelta[f *  inputChannels * inputWidth * inputHeight +
                                            ch * inputWidth * inputHeight + xx * inputHeight + yy] += 
                                        d * oldfilter[f * inputChannels * filterSize * filterSize + 
                                            ch * filterSize * filterSize + i * filterSize + j];
                                filter[f * inputChannels * filterSize * filterSize + 
                                                ch * filterSize * filterSize + i * filterSize + j] += 
                                        d * localEp * input[ch * inputWidth * inputHeight + xx * inputHeight + yy];
                            }
                        }
                    }
                }
            }
            //bias[f] += localEp * d;
            biasDelta[fxy] = localEp * d;
        }
        double[] input;
        double[] result;
        int inputChannels;
        int inputWidth;
        int inputHeight;
        double[] filter;
        int outputChannels;
        int outputWidth;
        int outputHeight;
        int filterSize;
        int stride;
        double[] bias;
        double[] delta;
        double[] oldfilter;
        double localEp;
        double[] tempDelta;
        double[] biasDelta;
        double[] backword(double[] delta, double[] result, double[] input, int inputChannels, int inputWidth, int inputHeight, 
                double[] filter, int outputChannels, int outputWidth, int outputHeight, int filterSize, int stride,
                double[] bias, ActivationFunction act, boolean useGpu){
            this.delta = delta;
            this.input = input;
            this.inputChannels = inputChannels;
            this.inputWidth = inputWidth;
            this.inputHeight = inputHeight;
            this.filter = filter;
            this.outputChannels = outputChannels;
            this.outputWidth = outputWidth;
            this.outputHeight = outputHeight;
            this.filterSize = filterSize;
            this.stride = stride;
            this.bias = bias;
            this.result = result;
            this.oldfilter = Arrays.copyOf(filter, filter.length);
            this.tempDelta = new double[outputChannels * inputChannels * inputWidth * inputHeight];
            this.localEp = ep / (outputWidth * outputHeight);
            this.biasDelta = Arrays.copyOf(result, result.length);
            
            if(useGpu){
                put(filter);
                put(delta);
                put(oldfilter);
                put(input);
                put(result);
                put(tempDelta);
                execute( outputChannels * outputWidth * outputHeight);
                get(filter);
                get(tempDelta);
                get(biasDelta);
            }else{
                IntStream.range(0, outputChannels).parallel().forEach(f -> {
                    for(int xy = 0; xy <  outputWidth * outputHeight; ++xy){
                        proc(f * outputWidth * outputHeight + xy);
                    }
                });
            }
            double[] newDelta = new double[input.length];
            IntStream.range(0, outputChannels).parallel().forEach(f -> {
                for(int chxy = 0; chxy < inputChannels * inputWidth * inputHeight; ++chxy) {
                        newDelta[chxy] += tempDelta[f * inputChannels * inputWidth * inputHeight + chxy];
                }
                for(int xy = 0; xy < outputWidth * outputHeight; ++xy){
                    bias[f] += biasDelta[f * outputWidth * outputHeight + xy];
                }
            });
            return newDelta;
        }
        

    }
    
    /** 畳み込み層 */
    static class ConvolutionLayer extends ImageNeuralLayer{
        double[] filter;
        double[] bias;
        int stride;
        int filterSize;
        boolean useGpu;
        public ConvolutionLayer(String name, int channel, int width, int height, int filterCount,  int size, int stride, boolean useGpu) {
            super(name, new RetifierdLinear(), channel, width, height, filterCount, width / stride, height / stride);
            this.filter = random.doubles(size * size * channel * filterCount)
                    .map(d -> (d * 2 - 0.5) / size / size / channel)
                    .toArray();
            this.bias = DoubleStream.generate(() -> .1).limit(filterCount).toArray();
            this.stride = stride;
            this.filterSize = size;
            this.useGpu = useGpu;
        }
        /** 畳み込みフィルタを適用する */
        @Override
        double[] forward(double[] img) {
            result = convolutionForwardKernel.forward(img, inputChannels, inputWidth, inputHeight, 
                filter, outputChannels, outputWidth, outputHeight, filterSize, stride,
                bias, activation, useGpu);
            return result;
        }

        /** 畳み込み層の学習 */
        @Override
        double[] backword(double[] input, double[] delta, ActivationFunction act){
            if(useGpu){
                // GPUバージョン
                double[] newDelta = convolutionBackwordDeltaKernel.backword(delta, result, 
                        inputChannels, inputWidth, inputHeight, filter,
                        outputChannels, outputWidth, outputHeight, filterSize, stride, true);
                convolutionBackwordFilterKernel.backword(delta, result, input, 
                        inputChannels, inputWidth, inputHeight, filter, 
                        outputChannels, outputWidth, outputHeight, filterSize, stride, useGpu);
                convolutionBackwordBiasKernel.backwordBias(delta, result, 
                        outputChannels, outputWidth, outputHeight, bias, useGpu);
                return newDelta;
                /*
                return convolutionBackwordKernel.backword(delta, result,
                        input, inputChannels, inputWidth, inputHeight, 
                        filter, outputChannels, outputWidth, outputHeight,
                        filterSize, stride, bias, act, useGpu);
                */
            }else{
                // CPUバージョン
                return convolutionBackwordKernel.backword(delta, result,
                        input, inputChannels, inputWidth, inputHeight, 
                        filter, outputChannels, outputWidth, outputHeight,
                        filterSize, stride, bias, act, false);
            }
        }
    }
    
    static class MaxPoolingLayer extends ImageNeuralLayer{
        int size;
        int stride;

        public MaxPoolingLayer(String name, int size, int stride, int channels, int inputWidth, int inputHeight) {
            super(name, new LinearFunction(), channels, inputWidth, inputHeight, channels,
                    inputWidth / stride, inputHeight / stride);
            this.size = size;
            this.stride = stride;
        }
        /** プーリング(max) */
        @Override
        double[] forward(double[] data){
            result = new double[outputChannels * outputWidth * outputHeight];
            IntStream.range(0, inputChannels).parallel().forEach(ch -> {
                for(int x = 0; x < outputWidth; ++x){
                    for(int y = 0; y < outputHeight; ++y){
                        double max = Double.NEGATIVE_INFINITY;
                        for(int i = 0; i < size; ++i){
                            int xx = x * stride + i - size / 2;
                            if(xx < 0 || xx >= inputWidth){
                                continue;
                            }
                            for(int j = 0; j < size; ++j){
                                int yy = y * stride + j - size / 2;
                                if(yy < 0 || yy >= inputHeight){
                                    continue;
                                }
                                double d = data[ch * inputWidth * inputHeight + xx * inputHeight + yy];
                                if(max < d){
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
        double[] backword(double[] in, double[] delta, ActivationFunction act){
            double[] newDelta = new double[in.length];
            IntStream.range(0, inputChannels).parallel().forEach(ch -> {
                for(int x = 0; x < outputWidth; ++x){
                    for(int y = 0; y < outputHeight; ++y){
                        double max = Double.NEGATIVE_INFINITY;
                        int maxX = 0;
                        int maxY = 0;
                        for(int i = 0; i < size; ++i){
                            int xx = x * stride + i - size / 2;
                            if(xx < 0 || xx >= inputWidth){
                                continue;
                            }
                            for(int j = 0; j < size; ++j){
                                int yy = y * stride + j - size / 2;
                                if(yy < 0 || yy >= inputHeight){
                                    continue;
                                }
                                double d = in[ch * inputWidth * inputHeight + xx * inputWidth + yy];
                                if(max < d){
                                    max = d;
                                    maxX = xx;
                                    maxY = yy;
                                }
                            }
                        }
                        int chxy = ch * outputWidth * outputHeight + x * outputHeight + y;
                        newDelta[ch * inputWidth * inputHeight + maxX * inputHeight + maxY] += 
                                act.diff(result[chxy]) * delta[chxy];
                    }
                }
            });


            return newDelta;
        }
    }
    
    static class NormalizeLayer extends ImageNeuralLayer{
        double[] averages;
        double[] rates;
        int size;
        double threshold;
        public NormalizeLayer(String name, int size, double threshold, int channels, int width, int height) {
            super(name, new LinearFunction(), channels, width, height, channels, width, height);
            this.size = size;
            this.threshold = threshold;
        }

        
        
        @Override
        double[] forward(double[] in) {
            averages = new double[in.length];
            rates = new double[in.length];
            result = new double[in.length];
            
            IntStream.range(0, inputChannels).parallel().forEach(ch -> {
                for(int lx = 0; lx < inputWidth; ++lx){
                    int x = lx;
                    for(int ly = 0; ly < inputHeight; ++ly){
                        int y = ly;
                        //平均
                        DoubleSummaryStatistics summary = 
                                IntStream.range(0, size)
                                .map(i -> x + i - size / 2)
                                .filter(xx -> xx >= 0 && xx < inputWidth)
                                .mapToObj(xx -> 
                                        IntStream.range(0, size)
                                        .map(j -> y + j - size / 2)
                                        .filter(yy -> yy >= 0 && yy < inputHeight)
                                        .mapToDouble(yy -> in[ch * inputWidth * inputHeight + xx * inputHeight + yy]))
                                .flatMapToDouble(s -> s).summaryStatistics();
                        //分散
                        double variance = 
                                IntStream.range(0, size)
                                .map(i -> x + i - size / 2)
                                .filter(xx -> xx >= 0 && xx < inputWidth)
                                .mapToObj(xx -> 
                                        IntStream.range(0, size)
                                        .map(j -> y + j - size / 2)
                                        .filter(yy -> yy >= 0 && yy < inputHeight)
                                        .mapToDouble(yy -> 
                                                (in[ch * inputWidth * inputHeight + xx * inputHeight + yy] 
                                                        - summary.getAverage()) * 
                                                (in[ch * inputWidth * inputHeight + xx * inputHeight + yy] 
                                                        - summary.getAverage())))
                                .flatMapToDouble(s -> s).sum() / summary.getCount();
                        double std = Math.max(threshold, Math.sqrt(variance));
                        int chxy = ch * outputWidth * outputHeight + x * outputHeight + y;
                        result[chxy] = (in[chxy] - summary.getAverage()) / std;
                        averages[chxy] = summary.getAverage();
                        rates[chxy] = std;
                    }
                }
            });
            
            return result;
        }

        @Override
        double[] backword(double[] in, double[] delta, ActivationFunction act) {
            return IntStream.range(0, delta.length).parallel()
                .mapToDouble(ch -> delta[ch] * rates[ch] + averages[ch])
                .toArray();
        }
        
    }
    
    static class FullyConnect{
        double[][] weight;
        double[] bias;
        int out;
        double[] result;
        int[] dropout;
        String name;
        double dropoutRate = 1;
        public FullyConnect(String name, int in, int out, double dropoutRate) {
            this.name = name;
            this.out = out;
            weight = Stream.generate(() -> 
                    DoubleStream.generate(() -> (random.nextDouble() * 1.5 - .5) / in).limit(out).toArray()
            ).limit(in).toArray(double[][]::new);
            bias = DoubleStream.generate(() -> .1).limit(out).toArray();
            dropout = IntStream.generate(() -> 1).limit(out).toArray();
            this.dropoutRate = dropoutRate;
        }
        
        public void prepareDropout(){
            dropout = random.doubles(out).mapToInt(d -> d < dropoutRate ? 1 : 0).toArray();
        }
        
        public double[] forward(double[] in){
            result = new double[out];
            IntStream.range(0, out).parallel().filter(j -> dropout[j] == 1).forEach(j -> {
                for(int i = 0; i < in.length; ++i){
                    result[j] += in[i] * weight[i][j];
                }
                result[j] += bias[j];
            });
            
            return result;
        }
        public double[] backward(double[] in, double[] delta, ActivationFunction act){
            double[][] oldweight = Arrays.stream(weight).parallel()
                    .map(row -> Arrays.copyOf(row, row.length))
                    .toArray(double[][]::new);
            double[] newDelta = new double[in.length];
            
            IntStream.range(0, in.length).parallel().forEach(i -> {
                for(int j = 0; j < out; ++j){
                    if(dropout[j] != 1){
                        continue;
                    }
                    double d = act.diff(result[j]) * delta[j];
                    newDelta[i] += d * oldweight[i][j];
                    weight[i][j] += d * ep * in[i];
                    
                }
            });
            IntStream.range(0, out).parallel().forEach(j -> {
                bias[j] += act.diff(result[j]) * delta[j] * ep;
            });
            return newDelta;
        }
    }
    
    static List<Double> historyData = new ArrayList<>();
    static LinkedList<Integer> rateData = new LinkedList<>();
    
    public static void main(String[] args) throws IOException {
        JFrame f = createFrame();
        f.setVisible(true);
        
        Path dir = Paths.get("C:\\Users\\naoki\\Desktop\\sampleimg288");
        List<String> categories = Files.list(dir)
                .filter(p -> Files.isDirectory(p))
                .map(p -> p.getFileName().toString())
                .filter(n -> !n.startsWith("_"))
                .collect(Collectors.toList());
        List<ImageNeuralLayer> layers = new ArrayList<>();
        InputFilter input = new InputFilter(256, 256);
        layers.add(input);
        
        //一段目
        layers.add(new ConvolutionLayer("conv1", 3, 256, 256, FILTER_1ST, 11, 4, USE_GPU1));
        //一段目のプーリング
        layers.add(new MaxPoolingLayer("pool1", 3, 2, FILTER_1ST, 256 / 4, 256 / 4));
        //一段目の正規化
        layers.add(new NormalizeLayer("norm1", 5, .1, FILTER_1ST, 256 / 8, 256 / 8));
        //二段目
        layers.add(new ConvolutionLayer("conv2", FILTER_1ST, 256 / 8, 256 / 8, FILTER_2ND, 5, 2, USE_GPU2));
        //二段目のプーリング
        layers.add(new MaxPoolingLayer("pool2", 3, 2, FILTER_2ND, 256 / 16, 256 / 16));
        
        NormalizeLayer norm2 = new NormalizeLayer("norm2", 5, .1, FILTER_2ND, 256 / 32, 256 / 32);
        layers.add(norm2);
        
        //全結合1
        FullyConnect fc1 = new FullyConnect("fc1", FILTER_2ND * 256 / 32 * 256 / 32, 32, 0.5);
        //全結合2
        FullyConnect fc2 = new FullyConnect("fc2", 32, categories.size(), 1);
        
        //Path p = dir.resolve("cat\\DSC00800.JPG");
        List<Img> files = Files.walk(dir)
                .filter(p -> !Files.isDirectory(p))
                .filter(p -> !p.getParent().getFileName().toString().startsWith("_"))
                .flatMap(p -> IntStream.range(0, 3).mapToObj(i -> 
                        IntStream.range(0, 3).mapToObj(j -> 
                                Stream.of(new Img(p, true, i, j), new Img(p, false, i, j)))
                                .flatMap(s -> s)).flatMap(s -> s))
                .collect(Collectors.toList());
        int[] count = {0};
        for(int loop = 0; loop < 30; ++loop){
        Collections.shuffle(files, random);
        long start = System.currentTimeMillis();
        long[] pStart = {start};
        files.stream().forEach(img -> {
            Path p = img.filename;
            String catName = p.getParent().getFileName().toString();
            double[] correctData = categories.stream()
                    .mapToDouble(name -> name.equals(catName) ? 1 : 0)
                    .toArray();

            BufferedImage readImg;
            try {
                readImg = ImageIO.read(p.toFile());
            } catch (IOException ex) {
                throw new UncheckedIOException(ex);
            }
            BufferedImage resized = resize(readImg, 256 + 32, 256 + 32, true, img.inverse);
            BufferedImage moved = move(resized, 256, 256, img.x * 16, img.y * 16);
            double[] readData = imageToArray(moved);


            double[] output = forward(layers, fc1, fc2, readData, correctData);
            //元画像の表示
            
            org.setIcon(new ImageIcon(resized));
            
            //判定結果
            double max = Double.NEGATIVE_INFINITY;
            int maxIndex = -1;
            for(int i = 0; i < output.length; ++i) {
                if (output[i] > max){
                    max = output[i];
                    maxIndex = i;
                }
            }
            if(maxIndex < 0){
                org.setText("no data");
                rateData.add(0);
            }else{
                org.setText(categories.get(maxIndex));
                rateData.add((int)correctData[maxIndex] );
            }
            //正答率
            while(rateData.size() > 20){
                rateData.removeFirst();
            }
            historyData.add(rateData.stream().mapToDouble(d -> d).sum() 
                    / rateData.size());
            Image lineGraph = createLineGraph(500, 200, 
                    historyData.stream().mapToDouble(d -> d).toArray(), 1, 0);
            historyLabel.setIcon(new ImageIcon(lineGraph));
            //一段目のフィルタの表示
            ConvolutionLayer conv1 = (ConvolutionLayer) layers.get(1);
            for(int i = 0; i < conv1.outputChannels; ++i){
                filtersLabel[i].setIcon(new ImageIcon(resize(arrayToImage(conv1.filter, i, 11, 11), 44, 44, false, false)));
            }
            //フィルタ後の表示
            for(int i = 0; i < conv1.outputChannels; ++i){
                filteredLabel[i].setIcon(new ImageIcon(arrayToImageMono(conv1.result, i, conv1.outputWidth, conv1.outputHeight)));
            }
            for(int i = 0; i < layers.get(2).outputChannels; ++i){
                pooledLabel[i].setIcon(new ImageIcon(resize(arrayToImageMono(layers.get(2).result, i, layers.get(2).outputWidth, layers.get(2).outputHeight), 48, 48)));
            }
            for(int i = 0; i < layers.get(3).outputChannels; ++i){
                normedLabel[i].setIcon(new ImageIcon(resize(arrayToImageMono(layers.get(3).result, i, layers.get(3).outputWidth, layers.get(3).outputHeight), 48, 48)));
            }
            //全結合一段の表示
            firstFc.setIcon(new ImageIcon(createGraph(256, 128, fc1.result)));
            //全結合二段の表示
            lastResult.setIcon(new ImageIcon(createGraph(256, 128, output)));
            
            firstBias.setIcon(new ImageIcon(createGraph(500, 128, conv1.bias)));
            secondBias.setIcon(new ImageIcon(createGraph(500, 128, ((ConvolutionLayer)layers.get(4)).bias)));
            fc1Bias.setIcon(new ImageIcon(createGraph(500, 128, fc1.bias)));
            fc2Bias.setIcon(new ImageIcon(createGraph(500, 128, fc2.bias)));
            
            count[0]++;
            if(count[0] >= 10){
                System.out.printf("%4d %.2f %s %s%n", count[0], 10 * 60 * 1000. / (System.currentTimeMillis() - pStart[0]),
                        convolutionForwardKernel.getExecutionMode(),
                        convolutionBackwordKernel.getExecutionMode());
                count[0] = 0;
                pStart[0] = System.currentTimeMillis();
            }
        });
        long end = System.currentTimeMillis();
        System.out.println(end - start);
        System.out.printf("%.2fm%n", (end - start) / 1000. / 60);
        }
    }
    
    static Image createGraph(int width, int height, double[] data){
        BufferedImage result = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = (Graphics2D) result.getGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, width, height);
        DoubleSummaryStatistics summary = Arrays.stream(data).summaryStatistics();
        DoubleToIntFunction f = d -> (int)(height - 
                (d - summary.getMin()) / (summary.getMax() - summary.getMin()) * height);
        g.setColor(Color.BLACK);
        g.drawLine(0, f.applyAsInt(0), width, f.applyAsInt(0));
        int bottom = f.applyAsInt(0);
        for(int i = 0; i < data.length; ++i){
            int left = i * width / data.length;
            int right = (i + 1) * width / data.length;
            int top = f.applyAsInt(data[i]);
            g.fillRect(left, Math.min(top, bottom), right - left - 1, Math.abs(bottom - top));
        }
        g.dispose();
        return result;
    }
    static Image createLineGraph(int width, int height, double[] data, double max, double min){
        BufferedImage result = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = (Graphics2D) result.getGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, width, height);
        g.setColor(Color.BLACK);
        for(int i = 1; i < data.length; ++i){
            g.drawLine((i - 1) * width / data.length, (int)((data[i - 1] - max) * (height - 10) / (min - max) - 5)
                    , i * width / data.length, (int)((data[i] - max) * (height - 10)/ (min - max)) - 5);
        }
        g.dispose();
        return result;
    }
    static JLabel org = new JLabel();
    static JLabel firstFc = new JLabel();
    static JLabel lastResult = new JLabel();
    static JLabel[] filtersLabel = Stream.generate(() -> new JLabel()).limit(FILTER_1ST)
            .toArray(JLabel[]::new);
    static JLabel firstBias = new JLabel();
    static JLabel secondBias = new JLabel();
    static JLabel fc1Bias = new JLabel();
    static JLabel fc2Bias = new JLabel();
    static JLabel[] filteredLabel = Stream.generate(() -> new JLabel()).limit(FILTER_1ST)
            .toArray(JLabel[]::new);
    static JLabel[] pooledLabel = Stream.generate(() -> new JLabel()).limit(FILTER_1ST)
            .toArray(JLabel[]::new);
    static JLabel[] normedLabel = Stream.generate(() -> new JLabel()).limit(FILTER_1ST)
            .toArray(JLabel[]::new);
    static JLabel historyLabel = new JLabel();
    
    static JFrame createFrame(){
        JFrame f = new JFrame("畳み込みニューラルネット");
        f.setLayout(new GridLayout(3, 1));
        
        JPanel north = new JPanel();
        // 上段
        f.add(north);
        north.setLayout(new GridLayout(1, 2));
        north.add(org);
        // 上段右
        JPanel northRight = new JPanel();
        north.add(northRight);
        northRight.setLayout(new GridLayout(2, 1));
        northRight.add(firstFc);
        northRight.add(lastResult);
        
        //中段
        int h = 3;//Math.max((int)Math.sqrt(FILTER_1ST), 1);
        int w = 5;//FILTER_1ST / h;
        JTabbedPane tab = new JTabbedPane(JTabbedPane.RIGHT);
        f.add(tab);
        JPanel middle = new JPanel();
        tab.add("filter", middle);
        middle.setLayout(new GridLayout(h, w));
        Arrays.stream(filtersLabel).forEach(middle::add);
        JPanel filtered = new JPanel();
        tab.add("filtered", filtered);
        filtered.setLayout(new GridLayout(h, w));
        Arrays.stream(filteredLabel).forEach(filtered::add);
        JPanel pooled = new JPanel();
        tab.add("pooled", pooled);
        pooled.setLayout(new GridLayout(h, w));
        Arrays.stream(pooledLabel).forEach(pooled::add);
        JPanel normed = new JPanel();
        tab.add("normed", normed);
        normed.setLayout(new GridLayout(h, w));
        Arrays.stream(normedLabel).forEach(normed::add);
        
        //下段
        JTabbedPane bottomTab = new JTabbedPane(JTabbedPane.TOP);
        f.add(bottomTab);
        JPanel bottom = new JPanel();
        bottomTab.add("bias", bottom);
        bottom.setLayout(new GridLayout(4, 1));
        bottom.add(firstBias);
        bottom.add(secondBias);
        bottom.add(fc1Bias);
        bottom.add(fc2Bias);
        bottomTab.add("history", historyLabel);
        
        f.setSize(540, 580);
        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        return f;
    }
    
    static double[] forward(List<ImageNeuralLayer> layers, FullyConnect fc1, FullyConnect fc2,
            double[] readData, double[] correctData){
        ImageNeuralLayer norm2 = layers.get(layers.size() - 1);
        layers.get(0).result = readData;
        for(int i = 1; i < layers.size(); ++i){
            layers.get(i).preLayer = layers.get(i - 1);
            layers.get(i).forward();
        }
        /*
        //一段目のフィルタをかける
        double[][][] filtered1 = conv1.forward(readData);
        //プーリング
        double[][][] pooled1 = pool1.forward(filtered1);
        double[][][] pooled1norm = norm1.forward(pooled1);
        //二段目のフィルタをかける
        double[][][] filtered2 = conv2.forward(pooled1norm);
        //プーリング
        double[][][] pooled2 = pool2.forward(filtered2);
        double[][][] pooled2norm = norm2.forward(pooled2);
        */
        double[] flattenPooled2 = norm2.getResult();
        //全結合一段
        fc1.prepareDropout();
        double[] fc1out = fc1.forward(flattenPooled2);
        double[] fc1outRe = Arrays.stream(fc1out).map(d -> d > 0 ? d : 0).toArray();
        //全結合二段
        double[] fc2out = fc2.forward(fc1outRe);
        //System.out.println(Arrays.stream(fc2out).mapToObj(d -> String.format("%.3f", d)).collect(Collectors.joining(",")));
        //ソフトマックス
        double[] output = softMax(fc2out);
        //結果を書き戻しておく
        for(int i = 0; i < fc2.result.length; ++i){
            fc2.result[i] = output[i];
        }
        //System.out.println(Arrays.stream(output).mapToObj(d -> String.format("%.3f", d)).collect(Collectors.joining(",")));
        //全結合二段の逆伝播
        double[] delta = IntStream.range(0, output.length)
                .mapToDouble(idx -> correctData[idx] - output[idx])
                //.map(d -> -d)
                .toArray();
        double[] deltaFc2 = fc2.backward(fc1out, delta, new SoftMaxFunction());
        //全結合一段の逆伝播
        double[] deltaFc1 = fc1.backward(flattenPooled2, deltaFc2, new RetifierdLinear());
        
        //プーリングの逆伝播
        for(int i = layers.size() - 1; i >= 1; --i){
            deltaFc1 = layers.get(i).backword(deltaFc1);
        }

        return output;
    }
    static void printDim(String name, double[][][] data){
        System.out.printf("%s:%dx%dx%d%n", name,
                data[0].length, data[0][0].length, data.length);
    }
    static void printData(double[][] data){
        System.out.println(Arrays.stream(data)
                .map(da -> Arrays.stream(da)
                        .mapToObj(d -> String.format("%.3f", d))
                        .collect(Collectors.joining(",")))
                .collect(Collectors.joining("\n")));
    }
    static double[] softMax(double[] output){
        double total = Arrays.stream(output).parallel()
                .map(d -> Math.exp(d))
                .sum();
        return Arrays.stream(output).parallel()
                .map(d -> Math.exp(d) / total)
                .toArray();
    }
    
    /** 値のクリッピング */
    static int clip(double c){
        if(c < 0) return 0;
        if(c > 255) return 255;
        return (int)c;
    }

    private static BufferedImage move(BufferedImage imgRead, int width, int height, int ox, int oy){
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics g = img.getGraphics();
        g.drawImage(imgRead, -ox, -oy, null);
        g.dispose();
        return img;
    }
    
    static BufferedImage resize(BufferedImage imgRead, int width, int height) {
        return resize(imgRead, width, height, true, false);
    }
    private static BufferedImage resize(BufferedImage imgRead, int width, int height, boolean bicubic, boolean inverse) {
        /*
        if(imgRead.getWidth() * height > imgRead.getHeight() * width){
            height = imgRead.getHeight() * width / imgRead.getWidth();
        }else{
            width = imgRead.getWidth() * height / imgRead.getHeight();
        }*/
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics g = img.getGraphics();
        if(bicubic){
            ((Graphics2D)g).setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC);
        }
        if(inverse){
            g.drawImage(imgRead, width, 0, -width, height, null);
        }else{
            g.drawImage(imgRead, 0, 0, width, height, null);
            
        }
        g.dispose();
        return img;
    }    

    static BufferedImage arrayToImage(double[] filteredData, int idx, int width, int height) {
        return arrayToImage(filteredData, idx, width, height, 1);
    }

    static BufferedImage arrayToImage(double[] filteredData, int idx, int width, int height, double rate) {
        DoubleSummaryStatistics summary = Arrays.stream(filteredData,
                idx * 3 * width * height, (idx + 1) * 3 * width * height).parallel()
                .summaryStatistics();
        double[] normed = Arrays.stream(filteredData, idx * width * height, (idx + 3) * width * height).parallel()
                        .map(d -> (d - summary.getMin()) 
                                / (summary.getMax() - summary.getMin()))
                        .toArray();
        //rate = 1;
        BufferedImage filtered = new BufferedImage(
                width, height,
                BufferedImage.TYPE_INT_RGB);
        for(int x = 0; x < width; ++x){
            for(int y = 0; y < height; ++y){
                filtered.setRGB(x, y,
                        ((int)clip(normed[x * height + y] * rate * 255) << 16) +
                        ((int)clip(normed[x * height + y + width * height] * rate * 255) << 8) +
                         (int)clip(normed[x * height + y + 2 * width * height] * rate * 255));
            }
        }
        return filtered;
    }
    static BufferedImage arrayToImageMono(double[] filteredData, int idx, int width, int height){
        double[] colorData = new double[width * height * 3];
        for(int i = 0; i < width * height; ++i){
            colorData[i] = filteredData[idx * width * height + i];
            colorData[i + width * height] = filteredData[idx * width * height + i];
            colorData[i + width * height * 2] = filteredData[idx * width * height + i];
        }
        return arrayToImage(colorData, 0, width, height);
    }
    /** 画像から配列へ変換 */
    private static double[] imageToArray(BufferedImage img) {
        int width = img.getWidth();
        int height = img.getHeight();
        double[] imageData = new double[3 * width * height];
        for(int x = 0; x < width; ++x){
            for(int y = 0; y < height; ++y){
                int rgb = img.getRGB(x, y);
                int pos = x * height + y;
                imageData[pos] = (rgb >> 16 & 0xff) / 255.;
                imageData[pos + width * height] = (rgb >> 8 & 0xff) / 255.;
                imageData[pos + 2 * width * height] = (rgb & 0xff) / 255.;
            }
        }
        
        DoubleSummaryStatistics summaryStatistics = Arrays.stream(imageData)
                .summaryStatistics();
        for(int ch = 0; ch < imageData.length; ++ch){
            imageData[ch] -= summaryStatistics.getAverage();
        }
        return imageData;
    }    
    
}
