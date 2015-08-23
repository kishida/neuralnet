package kishida.cnn;

import kishida.cnn.layers.ImageNeuralLayer;
import kishida.cnn.layers.MaxPoolingLayer;
import kishida.cnn.layers.ConvolutionLayer;
import kishida.cnn.layers.FullyConnect;
import kishida.cnn.layers.NormalizeLayer;
import kishida.cnn.layers.InputLayer;
import kishida.cnn.layers.NeuralLayer;
import kishida.cnn.activation.ActivationFunction;
import kishida.cnn.activation.SoftMaxFunction;
import kishida.cnn.activation.RetifierdLinear;
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
    private static final double ep = 0.001;
    public static Random random = new Random(1234);
    private static final boolean USE_GPU1 = true;
    private static final boolean USE_GPU2 = true;
    private static final int FILTER_1ST = 48;
    private static final int FILTER_2ND = 96;
    private static final int FULL_1ST = 4096;
    private static final int FILTER_1ST_SIZE = 11;
    //static final int FILTER_1ST = 48;
    //static final int FILTER_2ND = 96;
    private static final int FILTER_ROWS = 4;//Math.max((int)Math.sqrt(FILTER_1ST), 1);
    private static final int FILTER_COLS = 6;//FILTER_1ST / h;

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

    public static class ConvolutionForwardKernel extends Kernel{

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
    public static ConvolutionForwardKernel convolutionForwardKernel = new ConvolutionForwardKernel();
    public static ConvolutionBackwordKernel convolutionBackwordKernel = new ConvolutionBackwordKernel();
    public static ConvolutionBackwordDeltaKernel convolutionBackwordDeltaKernel = new ConvolutionBackwordDeltaKernel();
    public static ConvolutionBackwordBiasKernel convolutionBackwordBiasKernel = new ConvolutionBackwordBiasKernel();
    public static ConvolutionBackwordFilterKernel convolutionBackwordFilterKernel = new ConvolutionBackwordFilterKernel();

    public static class ConvolutionBackwordBiasKernel extends Kernel{

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

        public void backwordBias(double[] delta, double[] result,
                 int outputChannels, int outputWidth, int outputHeight,
                double[] bias, boolean useGpu){
            this.delta = delta;
            this.result = result;
            this.localEp = ep / (outputWidth * outputHeight);
            this.biasDelta = new double[result.length];

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


    public static class ConvolutionBackwordDeltaKernel extends Kernel{

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
                    if((xx - i + sizeHalf) % stride == 0 && // yy == y * stride + j -sizeHalf だとなぜかGPUで動かない
                             x >= 0 && x < outputWidth){
                        for(int j = 0; j < filterSize; ++j){
                            int y = (yy - j + sizeHalf) / stride;
                            if((yy - j + sizeHalf) % stride == 0 &&
                                    y >= 0 && y < outputHeight){
                                int fxy = f * outputWidth * outputHeight + x * outputHeight + y;
                                double d = (result[fxy] > 0 ? 1 : 0) * delta[fxy];
                                tempDelta += d * input[chxxyy] * filter[f * inputChannels * filterSize * filterSize +
                                            ch * filterSize * filterSize + i * filterSize + j];
                            }
                        }
                    }
                }
            }
            newDelta[chxxyy] = tempDelta;
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
        double[] delta;
        double[] newDelta;
        public double[] backword(double[] input, double[] delta, double[] result, int inputChannels, int inputWidth, int inputHeight,
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
            this.newDelta = new double[inputChannels * inputWidth * inputHeight];

            if(useGpu){
                put(filter);
                put(delta);
                put(result);
                put(input);
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


    public static class ConvolutionBackwordFilterKernel extends Kernel{

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

        public void backword(double[] delta, double[] result, double[] input, int inputChannels, int inputWidth, int inputHeight,
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

    public static class ConvolutionBackwordKernel extends Kernel{

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
                                        d * input[ch * inputWidth * inputHeight + xx * inputHeight + yy]
                                         * oldfilter[f * inputChannels * filterSize * filterSize +
                                            ch * filterSize * filterSize + i * filterSize + j];
/*
                                        d * oldfilter[f * inputChannels * filterSize * filterSize +
                                            ch * filterSize * filterSize + i * filterSize + j];
                                        */
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
        public double[] backward(double[] delta, double[] result, double[] input, int inputChannels, int inputWidth, int inputHeight,
                double[] filter, int outputChannels, int outputWidth, int outputHeight, int filterSize, int stride,
                double[] bias, boolean useGpu){
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



    public static NormalizeKernel normalizeKernel = new NormalizeKernel();
    public static class NormalizeKernel extends Kernel{

        public NormalizeKernel(){
            setExplicit(true);
        }

        @Override
        public void run() {
            int chxy = getGlobalId();
            proc(chxy);
        }

        private void proc(int chxy){
            int ch = chxy / (inputWidth * inputHeight);
            int x = (chxy % (inputWidth * inputHeight)) / inputHeight;
            int y = chxy % inputHeight;
            //平均
            int count = 0;
            double total = 0;
            for(int i = 0; i < size; ++i){
                int xx = x + i - size / 2;
                if(xx >= 0 && xx < inputWidth){
                    for(int j = 0; j < size; ++j){
                        int yy = y + j - size / 2;
                        if(yy >= 0 && yy < inputHeight){
                            total += input[ch * inputWidth * inputHeight + xx * inputHeight + yy];
                            ++count;
                        }
                    }
                }
            }
            double average = total / count;
            //分散
            double variance = 0;
            for(int i = 0; i < size; ++i){
                int xx = x + i - size / 2;
                if(xx >= 0 && xx < inputWidth){
                    for(int j = 0; j < size; ++j){
                        int yy = y + j - size / 2;
                        if(yy >= 0 && yy < inputHeight){
                            double d = input[ch * inputWidth * inputHeight + xx * inputHeight + yy];
                            variance += (d - average) * (d - average);
                        }
                    }
                }
            }
            double std = max(threshold, sqrt(variance / count));
            result[chxy] = (input[chxy] - average) / std;
            averages[chxy] = average;
            rates[chxy] = std;
        }
        double[] averages;
        double[] rates;
        double[] result;
        double[] input;
        int inputChannels;
        int inputWidth;
        int inputHeight;
        int size;
        double threshold;

        public double[] normalize(double[] input, int inputChannels, int inputWidth, int inputHeight, int size, double[] averages, double[] rates, double threshold, boolean useGpu){
            this.input = input;
            this.rates = rates;
            this.result = new double[inputChannels * inputWidth * inputHeight];
            this.averages = averages;
            this.inputChannels = inputChannels;
            this.inputWidth = inputWidth;
            this.inputHeight = inputHeight;
            this.size = size;
            this.threshold = threshold;

            if(useGpu){
                put(input);
                execute(inputChannels * inputWidth * inputHeight);
                get(averages);
                get(rates);
                get(result);
            }else{
                IntStream.range(0, inputChannels).parallel().forEach(ch -> {
                    for(int x = 0; x < inputWidth; ++x){
                        for(int y = 0; y < inputHeight; ++y){
                            proc(ch * inputWidth * inputHeight + x * inputHeight + y);
                        }
                    }
                });
            }

            return result;
        }
    }



    static List<Double> historyData = new ArrayList<>();
    static LinkedList<Integer> rateData = new LinkedList<>();

    public static void main(String[] args) throws IOException {
        String def = "C:\\Users\\naoki\\Desktop\\sampleimg288";
        Path dir = Paths.get(args.length > 0 ? args[0] : def);
        List<String> categories = Files.list(dir)
                .filter(p -> Files.isDirectory(p))
                .map(p -> p.getFileName().toString())
                .filter(n -> !n.startsWith("_"))
                .collect(Collectors.toList());

        JFrame f = createFrame();
        f.setVisible(true);

        List<NeuralLayer> layers = new ArrayList<>();
        InputLayer input = new InputLayer(227, 227);
        layers.add(input);

        ImageNeuralLayer pre = input;
        //一段目
        layers.add(pre = new ConvolutionLayer("conv1", pre, FILTER_1ST, FILTER_1ST_SIZE, 4, true));
        //一段目のプーリング
        layers.add(pre = new MaxPoolingLayer("pool1", 3, 2, pre));
        //一段目の正規化
        layers.add(pre = new NormalizeLayer("norm1", 5, .01, pre, true));
        //二段目
        layers.add(pre = new ConvolutionLayer("conv2", pre, FILTER_2ND, 5, 1, true));
        //二段目のプーリング
        layers.add(pre = new MaxPoolingLayer("pool2", 3, 2, pre));

        layers.add(pre = new NormalizeLayer("norm2", 5, .01, pre, true));

        //layers.add(pre = new ConvolutionLayer("conv3", pre, 384, 3, 1, true));
        //layers.add(pre = new ConvolutionLayer("conv4", pre, 384, 3, 1, true));
        //layers.add(pre = new ConvolutionLayer("conv5", pre, 256, 3, 1, true));
        //layers.add(pre = new MaxPoolingLayer("pool5", 3, 2, pre));

        NeuralLayer npre = pre;

        //FullyConnect fc0 = new FullyConnect("fc0", npre, 4096, .5, new RetifierdLinear());
        //layers.add(npre = fc0);

        //全結合1
        FullyConnect fc1 = new FullyConnect("fc1", npre, 2048, 0.5, new RetifierdLinear(), ep);
        layers.add(npre = fc1);
        //全結合2
        FullyConnect fc2 = new FullyConnect("fc2", npre, 300, 1, new SoftMaxFunction(), ep);
        layers.add(npre = fc2);

        layers.forEach(System.out::println);

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
            double[] correctData = DoubleStream.concat(categories.stream()
                    .mapToDouble(name -> name.equals(catName) ? 1 : 0),
                    DoubleStream.generate(() -> 0)).limit(1000)
                    .toArray();

            BufferedImage readImg;
            try {
                readImg = ImageIO.read(p.toFile());
            } catch (IOException ex) {
                throw new UncheckedIOException(ex);
            }
            BufferedImage resized = resize(readImg, 256 + 32, 256 + 32, true, img.inverse);
            BufferedImage moved = resize(move(resized, 256, 256, img.x * 16, img.y * 16), 227, 227);
            double[] readData = normalize(imageToArray(moved));

            double[] output = forward(layers, readData, correctData);
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
            }else if(maxIndex >= categories.size()){
                org.setText("out of data");
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
            for(int i = 0; i < conv1.getOutputChannels(); ++i){
                filtersLabel[i].setIcon(new ImageIcon(resize(arrayToImage(conv1.getFilter(), i, FILTER_1ST_SIZE, FILTER_1ST_SIZE), 44, 44, false, false)));
            }
            //フィルタ後の表示
            for(int i = 0; i < conv1.getOutputChannels(); ++i){
                filteredLabel[i].setIcon(new ImageIcon(arrayToImageMono(conv1.getResult(), i, conv1.getOutputWidth(), conv1.getOutputHeight())));
            }
            ImageNeuralLayer pool1 = (ImageNeuralLayer) layers.get(2);
            for(int i = 0; i < pool1.getOutputChannels(); ++i){
                pooledLabel[i].setIcon(new ImageIcon(resize(arrayToImageMono(pool1.getResult(), i, pool1.getOutputWidth(), pool1.getOutputHeight()), 48, 48)));
            }
            ImageNeuralLayer norm1 = (ImageNeuralLayer) layers.get(3);
            for(int i = 0; i < Math.min(normedLabel.length, norm1.getOutputChannels()); ++i){
                normedLabel[i].setIcon(new ImageIcon(resize(arrayToImageMono(norm1.getResult(), i, norm1.getOutputWidth(), norm1.getOutputHeight()), 48, 48)));
            }
            //全結合一段の表示
            firstFc.setIcon(new ImageIcon(createGraph(256, 128, fc1.getResult())));
            //全結合二段の表示
            lastResult.setIcon(new ImageIcon(createGraph(256, 128, output)));

            firstBias.setIcon(new ImageIcon(createGraph(500, 128, conv1.getBias())));
            secondBias.setIcon(new ImageIcon(createGraph(500, 128, ((ConvolutionLayer)layers.get(4)).getBias())));
            fc1Bias.setIcon(new ImageIcon(createGraph(500, 128, fc1.getBias())));
            fc2Bias.setIcon(new ImageIcon(createGraph(500, 128, fc2.getBias())));

            //System.out.println(Arrays.stream(output).mapToObj(d -> String.format("%.2f", d)).collect(Collectors.joining(",")));

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
            g.fillRect(left, Math.min(top, bottom), Math.max(1, right - left - 1), Math.abs(bottom - top));
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
        JTabbedPane tab = new JTabbedPane(JTabbedPane.RIGHT);
        f.add(tab);
        JPanel middle = new JPanel();
        tab.add("Filter", middle);
        middle.setLayout(new GridLayout(FILTER_ROWS, FILTER_COLS));
        Arrays.stream(filtersLabel).forEach(middle::add);
        JPanel filtered = new JPanel();
        tab.add("filtered", filtered);
        filtered.setLayout(new GridLayout(FILTER_ROWS, FILTER_COLS));
        Arrays.stream(filteredLabel).forEach(filtered::add);
        JPanel pooled = new JPanel();
        tab.add("pooled", pooled);
        pooled.setLayout(new GridLayout(FILTER_ROWS, FILTER_COLS));
        Arrays.stream(pooledLabel).forEach(pooled::add);
        JPanel normed = new JPanel();
        tab.add("normed", normed);
        normed.setLayout(new GridLayout(FILTER_ROWS, FILTER_COLS));
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

    static double[] forward(List<NeuralLayer> layers, double[] readData, double[] correctData){
        ((InputLayer)layers.get(0)).setInput(readData);
        for(int i = 1; i < layers.size(); ++i){
            //layers.get(i).preLayer = layers.get(i - 1);
            layers.get(i).forward();
        }
        double[] output = layers.get(layers.size() - 1).getResult();
        //誤差を求める
        double[] delta = IntStream.range(0, output.length)
                .mapToDouble(idx -> correctData[idx] - output[idx])
                //.map(d -> -d)
                .toArray();
        //逆伝播
        for(int i = layers.size() - 1; i >= 1; --i){
            delta = layers.get(i).backward(delta);
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
        double[] normed = Arrays.stream(filteredData, idx * 3 * width * height, (idx + 1) * 3 * width * height).parallel()
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
        IntStream.range(0, width).parallel().forEach(x -> {
            for(int y = 0; y < height; ++y){
                int i = x * height + y;
                double c = filteredData[idx * width * height + i];
                colorData[i] = c;
                colorData[i + width * height] = c;
                colorData[i + width * height * 2] = c;
            }
        });
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
        return imageData;
    }
    static double[] normalize(double[] data){
        int size = data.length / 3;
        double[] result = new double[data.length];
        for(int i = 0; i < 3; ++i){
            int start = i * size;
            int end = start + size;
            // 平均
            DoubleSummaryStatistics sum = Arrays.stream(data, start, end).parallel().summaryStatistics();
            // 分散
            double dist = Math.sqrt(Arrays.stream(data, start, end).parallel()
                    .map(d -> (d - sum.getAverage()) * (d - sum.getAverage()))
                    .sum() / sum.getCount());
            // 正規化
            for(int j = start; j < end; ++j){
                result[j] = (data[j] - sum.getAverage()) / dist;
            }
        }
        return result;
    }
}
