package kishida.imagefiltering;

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
    static final double ep = 0.00000000001;

    static class Img{

        public Img(Path filename, boolean inverse) {
            this.filename = filename;
            this.inverse = inverse;
        }
        Path filename;
        boolean inverse;
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
            double ap = apply(value);
            return ap * (1 - ap);
        }
        
    }
    
    static abstract class ImageNouralLayer{
        String name;
        double[][][] result;
        ImageNouralLayer preLayer;
        ActivationFunction activation;

        public ImageNouralLayer(String name, ActivationFunction activation) {
            this.name = name;
            this.activation = activation;
        }
        
        double[][][] forward(){
            return forward(preLayer.result);
        }
        double[][][] backword(double[][][] delta){
            return backword(preLayer.result, delta, preLayer.activation);
        }
        
        abstract double[][][] forward(double[][][] in);
        abstract double[][][] backword(double[][][] in, double[][][] delta, ActivationFunction activation);

        public String getName() {
            return name;
        }

        public double[][][] getResult() {
            return result;
        }
        
    }
    
    static class InputFilter extends ImageNouralLayer{

        public InputFilter() {
            super("入力", new LinearFunction());
        }

        @Override
        double[][][] forward(double[][][] in) {
            this.result = in;
            return result;
        }

        @Override
        double[][][] backword(double[][][] in, double[][][] delta, ActivationFunction act) {
            // do nothing
            return null;
        }
        
    }
    
    /** 畳み込み層 */
    static class ConvolutionLayer extends ImageNouralLayer{
        double[][][][] filter;
        double[] bias;
        int stride;
        public ConvolutionLayer(String name, int filterCount, int channel, int size, int stride) {
            super(name, new RetifierdLinear());
            this.filter = Stream.generate(() -> Stream.generate(() -> createRandomFilter(size))
                            .limit(channel).toArray(double[][][]::new))
                        .limit(filterCount).toArray(double[][][][]::new);
            this.bias = DoubleStream.generate(() -> r.nextDouble()).limit(filterCount).toArray();
            this.stride = stride;
        }
        /** 畳み込みフィルタを適用する */
        @Override
        double[][][] forward(double[][][] img) {
            int width = img[0].length;
            int height = img[0][0].length;
            int filterSize = filter[0][0].length;
            result = new double[filter.length][width / stride][height / stride];
            IntStream.range(0, filter.length).parallel().forEach(fi ->{
                for(int x = 0; x < width / stride; ++x){
                    for(int y = 0; y < height / stride; ++y){
                        for(int ch = 0; ch < filter[fi].length; ++ch){
                            for(int i = 0; i < filter[0][0].length; ++i){
                                int xx = x * stride + i - filterSize / 2;
                                if(xx < 0 || xx >= width){
                                    continue;
                                }
                                for(int j = 0; j < filter[0][0][0].length; ++j){
                                    int yy = y * stride + j - filterSize / 2;
                                    if(yy < 0 || yy >= height){
                                        continue;
                                    }
                                    result[fi][x][y] += img[ch][xx][yy] * 
                                            filter[fi][ch][i][j];
                                }
                            }
                        }
                        result[fi][x][y] += bias[fi];
                    }
                }
                for(int x = 0; x < width / stride; ++x){
                    for(int y = 0; y < height / stride; ++y){
                        result[fi][x][y] = activation.apply(result[fi][x][y]);
                    }
                }
            });
            return result;
        }

        /** 畳み込み層の学習 */
        @Override
        double[][][] backword(double[][][] input, double[][][] delta, ActivationFunction act){
            double[][][] newDelta = new double[input.length][input[0].length][input[0][0].length];
            double[][][][] oldfilter = Arrays.stream(filter)
                    .map(f -> Arrays.stream(f)
                            .map(ch -> Arrays.stream(ch)
                                    .map(row -> Arrays.copyOf(row, row.length))
                                    .toArray(double[][]::new)
                            ).toArray(double[][][]::new))
                    .toArray(double[][][][]::new);
            
            for(int lf = 0; lf < filter.length; ++lf) {
                int f = lf;
                IntStream.range(0, filter[f].length).parallel().forEach(ch -> {
                    for(int x = 0; x < input[0].length / stride; ++x){
                        for(int y = 0; y < input[0][0].length / stride; ++y){
                            for(int i = 0; i < filter[0][0].length; ++i){
                                int xx = x * stride + i - filter[0][0].length / 2;
                                if(xx < 0 || xx >= input[0].length){
                                    continue;
                                }
                                for(int j = 0; j < filter[0][0][0].length; ++j){
                                    int yy = y * stride + j - filter[0][0][0].length / 2;
                                    if(yy < 0 || yy >= input[0][0].length){
                                        continue;
                                    }
                                    double d = act.diff(input[ch][xx][yy]) * delta[f][x][y];
                                    newDelta[ch][i][j] += d * oldfilter[f][ch][i][j];
                                    filter[f][ch][i][j] += d * ep;
                                }
                            }
                        }
                    }
                });
                // chでの並列ができなくなるので抜き出しておく
                for(int x = 0; x < input[0].length / stride; ++x){
                    for(int y = 0; y < input[0][0].length / stride; ++y){
                        bias[f] += ep * delta[f][x][y];
                    }
                }
            };
            return newDelta;
        }
    }
    
    static class Normalize extends ImageNouralLayer{
        double range;
        double average;

        public Normalize(String name) {
            super(name, new LinearFunction());
        }
        
        @Override
        double[][][] forward(double[][][] data){
            result = new double[data.length][data[0].length][data[0][0].length];
            for(int i = 0; i < data.length; ++i){
                DoubleSummaryStatistics st = Arrays.stream(data[i])
                        .flatMapToDouble(Arrays::stream)
                        .summaryStatistics();
                average = st.getAverage();
                range = st.getMax() - average;
                if(range == 0){
                    // rangeが0になるようであれば、割らないようにする
                    range = 1;
                }
                for(int j = 0; j < data[i].length; ++j){
                    for(int k = 0; k < data[i][j].length; ++k){
                        result[i][j][k] = (data[i][j][k] - average) / range;
                    }
                }

            }
            return result;
        }
        @Override
        double[][][] backword(double[][][] in, double[][][] data, ActivationFunction act){
            double[][][] newDelta = new double[data.length][data[0].length][data[0][0].length];
            for(int i = 0; i < data.length; ++i){
                for(int j = 0; j < data[i].length; ++j){
                    for(int k = 0; k < data[i][j].length; ++k){
                        newDelta[i][j][k] = data[i][j][k] * range + average;
                    }
                }
            }
            return newDelta;
        }
    }
    
    
    static class MaxPoolingLayer extends ImageNouralLayer{
        int size;
        int stride;

        public MaxPoolingLayer(String name, int size, int stride) {
            super(name, new LinearFunction());
            this.size = size;
            this.stride = stride;
        }
        /** プーリング(max) */
        @Override
        double[][][] forward(double[][][] data){
            result = new double[data.length][data[0].length / stride][data[0][0].length / stride];
            IntStream.range(0, data.length).parallel().forEach(ch -> {
                for(int x = 0; x < data[0].length / stride; ++x){
                    for(int y = 0; y < data[0][0].length / stride; ++y){
                        double max = Double.NEGATIVE_INFINITY;
                        for(int i = 0; i < size; ++i){
                            int xx = x * stride + i - size / 2;
                            if(xx < 0 || xx >= data[0].length){
                                continue;
                            }
                            for(int j = 0; j < size; ++j){
                                int yy = y * stride + j - size / 2;
                                if(yy < 0 || yy >= data[0][0].length){
                                    continue;
                                }
                                if(max < data[ch][xx][yy]){
                                    max = data[ch][xx][yy];
                                }
                            }
                        }
                        result[ch][x][y] = max;
                    }
                }
            });
            return result;
        }

        @Override
        double[][][] backword(double[][][] in, double[][][] delta, ActivationFunction act){
            double[][][] newDelta = new double[in.length][in[0].length][in[0][0].length];
            IntStream.range(0, in.length).parallel().forEach(ch -> {
                for(int x = 0; x < in[0].length / stride; ++x){
                    for(int y = 0; y < in[0][0].length / stride; ++y){
                        double max = Double.NEGATIVE_INFINITY;
                        int maxX = 0;
                        int maxY = 0;
                        for(int i = 0; i < size; ++i){
                            int xx = x * stride + i - size / 2;
                            if(xx < 0 || xx >= in[0].length){
                                continue;
                            }
                            for(int j = 0; j < size; ++j){
                                int yy = y * stride + j - size / 2;
                                if(yy < 0 || yy >= in[0][0].length){
                                    continue;
                                }
                                if(max < in[ch][xx][yy]){
                                    max = in[ch][xx][yy];
                                    maxX = xx;
                                    maxY = yy;
                                }
                            }
                        }
                        newDelta[ch][maxX][maxY] = delta[ch][x][y];
                    }
                }
            });


            return newDelta;
        }
    }
    
    static class NormalizeLayer extends ImageNouralLayer{
        double[][][] averages;
        double[][][] rates;
        int size;
        double threshold;
        public NormalizeLayer(String name, int size, double threshold) {
            super(name, new LinearFunction());
            this.size = size;
            this.threshold = threshold;
        }

        
        
        @Override
        double[][][] forward(double[][][] in) {
            averages = new double[in.length][in[0].length][in[0][0].length];
            rates = new double[in.length][in[0].length][in[0][0].length];
            result = new double[in.length][in[0].length][in[0][0].length];
            
            IntStream.range(0, in.length).parallel().forEach(ch -> {
                for(int lx = 0; lx < in[ch].length; ++lx){
                    int x = lx;
                    for(int ly = 0; ly < in[ch][x].length; ++ly){
                        int y = ly;
                        //平均
                        DoubleSummaryStatistics summary = 
                                IntStream.range(0, size)
                                .map(i -> x + i - size / 2)
                                .filter(xx -> xx >= 0 && xx < in[ch].length)
                                .mapToObj(xx -> 
                                        IntStream.range(0, size)
                                        .map(j -> y + j - size / 2)
                                        .filter(yy -> yy >= 0 && yy < in[ch][x].length)
                                        .mapToDouble(yy -> in[ch][xx][yy]))
                                .flatMapToDouble(s -> s).summaryStatistics();
                        //分散
                        double variance = 
                                IntStream.range(0, size)
                                .map(i -> x + i - size / 2)
                                .filter(xx -> xx >= 0 && xx < in[ch].length)
                                .mapToObj(xx -> 
                                        IntStream.range(0, size)
                                        .map(j -> y + j - size / 2)
                                        .filter(yy -> yy >= 0 && yy < in[ch][x].length)
                                        .mapToDouble(yy -> 
                                                (in[ch][xx][yy] - summary.getAverage()) * 
                                                (in[ch][xx][yy] - summary.getAverage())))
                                .flatMapToDouble(s -> s).sum();
                        double std = Math.max(threshold, Math.sqrt(variance));
                        result[ch][x][y] = (in[ch][x][y] - summary.getAverage()) / std;
                        averages[ch][x][y] = summary.getAverage();
                        rates[ch][x][y] = std;
                    }
                }
            });
            
            return result;
        }

        @Override
        double[][][] backword(double[][][] in, double[][][] delta, ActivationFunction act) {
            return IntStream.range(0, delta.length)
                    .mapToObj(ch -> IntStream.range(0, delta[ch].length)
                        .mapToObj(x -> IntStream.range(0, delta[ch][x].length)
                            .mapToDouble(y -> delta[ch][x][y] * rates[ch][x][y] + averages[ch][x][y])
                                .toArray())
                        .toArray(double[][]::new))
                    .toArray(double[][][]::new);
        }
        
    }
    
    static class FullyConnect{
        double[][] weight;
        double[] bias;
        int out;
        double[] result;
        String name;
        public FullyConnect(String name, int in, int out) {
            this.name = name;
            this.out = out;
            weight = Stream.generate(() -> 
                    DoubleStream.generate(() -> r.nextDouble() * 2 - 1).limit(out).toArray()
            ).limit(in).toArray(double[][]::new);
            bias = DoubleStream.generate(() -> r.nextDouble()).limit(out).toArray();
        }
        
        public double[] forward(double[] in){
            result = new double[out];
            for(int j = 0; j < out; ++j){
                for(int i = 0; i < in.length; ++i){
                    result[j] += in[i] * weight[i][j];
                }
                result[j] += bias[j];
            }
            
            return result;
        }
        public double[] backward(double[] in, double[] delta, ActivationFunction act){
            double[][] oldweight = Arrays.stream(weight)
                    .map(row -> Arrays.copyOf(row, row.length))
                    .toArray(double[][]::new);
            double[] newDelta = new double[in.length];
            
            for(int j = 0; j < out; ++j){
                for(int i = 0; i < in.length; ++i){
                    double d = act.diff(in[i]) * delta[j];
                    newDelta[i] += d * oldweight[i][j];
                    weight[i][j] += d * ep;
                    
                }
                bias[j] += delta[j] * ep;
            }
            return newDelta;
        }
    }
    static double[][][] norm(double[][][] data){
        DoubleSummaryStatistics summary = Arrays.stream(data)
                .flatMap(Arrays::stream)
                .flatMapToDouble(Arrays::stream)
                .summaryStatistics();
        
        return Arrays.stream(data)
                .map(ch -> Arrays.stream(ch)
                        .map(row -> Arrays.stream(row)
                                .map(d -> d - summary.getAverage())
                                .toArray())
                        .toArray(double[][]::new))
                .toArray(double[][][]::new);
    }
    static double[][][] norm0(double[][][] data){
        return data;
    }
    public static void main(String[] args) throws IOException {
        JFrame f = createFrame();
        f.setVisible(true);
        
        Path dir = Paths.get("C:\\Users\\naoki\\Desktop\\sampleimg");
        List<String> categories = Files.list(dir)
                .filter(p -> Files.isDirectory(p))
                .map(p -> p.getFileName().toString())
                .collect(Collectors.toList());
        List<ImageNouralLayer> layers = new ArrayList<>();
        InputFilter input = new InputFilter();
        layers.add(input);
        //一段目
        layers.add(new ConvolutionLayer("conv1", 48, 3, 11, 4));
        //一段目のプーリング
        layers.add(new MaxPoolingLayer("pool1", 3, 2));
        //一段目の正規化
        layers.add(new NormalizeLayer("norm1", 5, .1));
        //二段目
        layers.add(new ConvolutionLayer("conv2", 96, 48, 5, 2));
        //二段目のプーリング
        layers.add(new MaxPoolingLayer("pool2", 3, 2));
        
        NormalizeLayer norm2 = new NormalizeLayer("norm2", 5, .1);
        layers.add(norm2);
        
        //全結合1
        FullyConnect fc1 = new FullyConnect("fc1", 6144, 32);
        //全結合2
        FullyConnect fc2 = new FullyConnect("fc2", 32, categories.size());
        
        //Path p = dir.resolve("cat\\DSC00800.JPG");
        List<Path> files = Files.walk(dir)
                .filter(p -> !Files.isDirectory(p))
                .collect(Collectors.toList());
        Collections.shuffle(files);
        files.stream().forEach(p -> {
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
            BufferedImage resized = resize(readImg, 256, 256);
            double[][][] readData = norm0(imageToArray(resized));

            //元画像の表示
            org.setIcon(new ImageIcon(resized));

            double[] output = forward(layers, fc1, fc2, readData, correctData);
            //一段目のフィルタの表示
            ConvolutionLayer conv1 = (ConvolutionLayer) layers.get(1);
            for(int i = 0; i < conv1.filter.length; ++i){
                filtersLabel[i].setIcon(new ImageIcon(resize(arrayToImage(conv1.filter[i]), 44, 44, false, false)));
            }
            //フィルタ後の表示
            for(int i = 0; i < conv1.result.length; ++i){
                filteredLabel[i].setIcon(new ImageIcon(arrayToImage(conv1.result[i])));
            }
            for(int i = 0; i < layers.get(2).result.length; ++i){
                pooledLabel[i].setIcon(new ImageIcon(arrayToImage(layers.get(2).result[i])));
            }
            for(int i = 0; i < layers.get(3).result.length; ++i){
                normedLabel[i].setIcon(new ImageIcon(arrayToImage(layers.get(3).result[i])));
            }
            //全結合一段の表示
            firstFc.setIcon(new ImageIcon(createGraph(256, 128, fc1.result)));
            //全結合二段の表示
            lastResult.setIcon(new ImageIcon(createGraph(256, 128, output)));
            
            firstBias.setIcon(new ImageIcon(createGraph(500, 128, conv1.bias)));
            secondBias.setIcon(new ImageIcon(createGraph(500, 128, ((ConvolutionLayer)layers.get(4)).bias)));
            fc1Bias.setIcon(new ImageIcon(createGraph(500, 128, fc1.bias)));
            fc2Bias.setIcon(new ImageIcon(createGraph(500, 128, fc2.bias)));
            
        });
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
        for(int i = 0; i < data.length; ++i){
            int left = i * width / data.length;
            int bottom = f.applyAsInt(0);
            int right = (i + 1) * width / data.length;
            int top = f.applyAsInt(data[i]);
            g.fillRect(left, Math.min(top, bottom), right - left - 1, Math.abs(bottom - top));
        }
        
        return result;
    }
    
    static JLabel org = new JLabel();
    static JLabel firstFc = new JLabel();
    static JLabel lastResult = new JLabel();
    static JLabel[] filtersLabel = Stream.generate(() -> new JLabel()).limit(48)
            .toArray(JLabel[]::new);
    static JLabel firstBias = new JLabel();
    static JLabel secondBias = new JLabel();
    static JLabel fc1Bias = new JLabel();
    static JLabel fc2Bias = new JLabel();
    static JLabel[] filteredLabel = Stream.generate(() -> new JLabel()).limit(48)
            .toArray(JLabel[]::new);
    static JLabel[] pooledLabel = Stream.generate(() -> new JLabel()).limit(48)
            .toArray(JLabel[]::new);
    static JLabel[] normedLabel = Stream.generate(() -> new JLabel()).limit(48)
            .toArray(JLabel[]::new);
    
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
        JTabbedPane tab = new JTabbedPane();
        f.add(tab);
        JPanel middle = new JPanel();
        tab.add("filter", middle);
        middle.setLayout(new GridLayout(6, 8));
        Arrays.stream(filtersLabel).forEach(middle::add);
        JPanel filtered = new JPanel();
        tab.add("filtered", filtered);
        filtered.setLayout(new GridLayout(6, 8));
        Arrays.stream(filteredLabel).forEach(filtered::add);
        JPanel pooled = new JPanel();
        tab.add("pooled", pooled);
        pooled.setLayout(new GridLayout(6, 8));
        Arrays.stream(pooledLabel).forEach(pooled::add);
        JPanel normed = new JPanel();
        tab.add("normed", normed);
        normed.setLayout(new GridLayout(6, 8));
        Arrays.stream(normedLabel).forEach(normed::add);
        
        //下段
        JPanel bottom = new JPanel();
        f.add(bottom);
        bottom.setLayout(new GridLayout(4, 1));
        bottom.add(firstBias);
        bottom.add(secondBias);
        bottom.add(fc1Bias);
        bottom.add(fc2Bias);
        
        f.setSize(540, 580);
        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        return f;
    }
    
    static double[] forward(List<ImageNouralLayer> layers, FullyConnect fc1, FullyConnect fc2,
            double[][][] readData, double[] correctData){
        ImageNouralLayer norm2 = layers.get(layers.size() - 1);
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
        double[] flattenPooled2 = flatten(norm2.getResult());
        //全結合一段
        double[] fc1out = fc1.forward(flattenPooled2);
        double[] re = Arrays.stream(fc1out).map(d -> d > 0 ? d : 0).toArray();
        //全結合二段
        double[] fc2out = fc2.forward(re);
        System.out.println(Arrays.stream(fc2out).mapToObj(d -> String.format("%.3f", d)).collect(Collectors.joining(",")));
        //ソフトマックス
        double[] output = softMax(fc2out);
        System.out.println(Arrays.stream(output).mapToObj(d -> String.format("%.3f", d)).collect(Collectors.joining(",")));
        //全結合二段の逆伝播
        double[] delta = IntStream.range(0, output.length)
                .mapToDouble(idx -> -(correctData[idx] - output[idx]))
                .toArray();
        double[] deltaFc2 = fc2.backward(re, delta, norm2.activation);
        //全結合一段の逆伝播
        double[] deltaFc1 = fc1.backward(flattenPooled2, deltaFc2, new RetifierdLinear());
        
        double[][][] deltaFc1Dim3 = divide3dim(deltaFc1, norm2.result[0].length, norm2.result[0][0].length);
        //プーリングの逆伝播
        for(int i = layers.size() - 1; i >= 1; --i){
            deltaFc1Dim3 = layers.get(i).backword(deltaFc1Dim3);
        }
        /*
        double[][][] deltaNorm2 = norm2.backword(null, deltaFc1Dim3);
        printDim("deltaNorm2", deltaNorm2);
        double[][][] deltaPool2 = pool2.backword(filtered2, deltaNorm2);
        //二段目のフィルタの逆伝播
        double[][][] deltaConv2 = conv2.backword(pooled1norm, deltaPool2);
        //プーリングの逆伝播
        double[][][] deltaPool1 = pool1.backword(filtered1, norm1.backword(null, deltaConv2));
        
        //一段目のフィルタの逆伝播
        double[][][] all1 = Arrays.stream(readData).map(ch -> 
                Arrays.stream(ch).map(row -> 
                        Arrays.stream(row).map(d -> 1)
                                .toArray())
                        .toArray(size -> new double[size][]))
                .toArray(size -> new double[size][][]);
        conv1.backword(all1, deltaPool1);
        */
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
    
    static Random r = new Random();
    static double[][] createRandomFilter(int size){
        double [][] result = new double[size][size];
        double total = 0;
        for(int i = 0; i < size; ++i){
            for(int j = 0; j < size; ++j){
                result[i][j] = r.nextDouble();
                total += result[i][j];
            }
        }
        //total = (total + 1) / (size * size);
        for(int i = 0; i < size; ++i){
            for(int j = 0; j < size; ++j){
                //result[i][j] /= total;
            }
        }
        
        return result;
    }
    
    static double[][][] divide3dim(double[] data, int sec, int third){
        return IntStream.range(0, data.length / sec / third).mapToObj(i -> 
            IntStream.range(0, sec).mapToObj(j -> 
                Arrays.copyOfRange(data, 
                        i * sec * third +j * third, 
                        i * sec * third + j * third + third)
            ).toArray(double[][]::new)
        ).toArray(double[][][]::new);
    }
    
    static double[] flatten(double[][][] data){
        return Arrays.stream(data)
                .flatMap(Arrays::stream)
                .flatMapToDouble(Arrays::stream)
                .toArray();
    }
    
    /** 値のクリッピング */
    static int clip(double c){
        if(c < 0) return 0;
        if(c > 255) return 255;
        return (int)c;
    }

    
    private static BufferedImage resize(BufferedImage imgRead, int width, int height) {
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

    static BufferedImage arrayToImage(double[][][] filteredData) {
        return arrayToImage(filteredData, 1);
    }

    static BufferedImage arrayToImage(double[][][] filteredData, double rate) {
        //rate = 1;
        BufferedImage filtered = new BufferedImage(
                filteredData[0].length, filteredData[0][0].length,
                BufferedImage.TYPE_INT_RGB);
        for(int x = 0; x < filteredData[0].length; ++x){
            for(int y = 0; y < filteredData[0][0].length; ++y){
                filtered.setRGB(x, y,
                        ((int)clip(filteredData[0][x][y] * rate * 255) << 16) +
                        ((int)clip(filteredData[1][x][y] * rate * 255) << 8) +
                         (int)clip(filteredData[2][x][y] * rate * 255));
            }
        }
        return filtered;
    }
    static BufferedImage arrayToImage(double[][] filteredData){
        DoubleSummaryStatistics summary = Arrays.stream(filteredData)
                .flatMapToDouble(Arrays::stream)
                .summaryStatistics();
        double[][] normed = Arrays.stream(filteredData)
                .map(row -> Arrays.stream(row)
                        .map(d -> (d - summary.getMin()) 
                                / (summary.getMax() - summary.getMin()))
                        .toArray())
                .toArray(double[][]::new);
        
        return arrayToImage(new double[][][]{normed, normed, normed});
    }
    /** 画像から配列へ変換 */
    private static double[][][] imageToArray(BufferedImage img) {
        int width = img.getWidth();
        int height = img.getHeight();
        double[][][] imageData = new double[3][width][height];
        for(int x = 0; x < width; ++x){
            for(int y = 0; y < height; ++y){
                int rgb = img.getRGB(x, y);
                imageData[0][x][y] = (rgb >> 16 & 0xff) / 255.;
                imageData[1][x][y] = (rgb >> 8 & 0xff) / 255.;
                imageData[2][x][y] = (rgb & 0xff) / 255.;
            }
        }
        return imageData;
    }    
    
}
