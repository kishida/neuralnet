/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.imagefiltering;

import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import javax.imageio.ImageIO;

/**
 *
 * @author naoki
 */
public class ConvolutionalNet {
    static final double ep = 0.000000001;

    static class Img{

        public Img(Path filename, boolean inverse) {
            this.filename = filename;
            this.inverse = inverse;
        }
        Path filename;
        boolean inverse;
    }
    
    /** 畳み込み層 */
    static class ConvolutionLayer{
        double[][][][] filter;
        double[] bias;
        int stride;

        public ConvolutionLayer(int filterCount, int channel, int size, int stride) {
            this.filter = Stream.generate(() -> Stream.generate(() -> createRandomFilter(size))
                            .limit(channel).toArray(len -> new double[len][][]))
                        .limit(filterCount).toArray(len -> new double[len][][][]);
            this.bias = DoubleStream.generate(() -> r.nextDouble()).limit(filterCount).toArray();
            this.stride = stride;
        }
        /** 畳み込みフィルタを適用する */
        double[][][] applyFilter(double[][][] img) {
            int width = img[0].length;
            int height = img[0][0].length;
            int filterSize = filter[0][0].length;
            double[][][] result = new double[filter.length][width / stride][height / stride];
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
            });
            return result;
        }

        /** 畳み込み層の学習 */
        void learn(double[][][] input, double[][][] delta){
            IntStream.range(0, filter.length).parallel().forEach(f -> {
                for(int ch = 0; ch < filter[0].length; ++ch){
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
                                    double d = (input[ch][xx][yy] > 0 ? 1 : 0) * delta[f][x][y];
                                    filter[f][ch][i][j] += d * ep;
                                }
                            }
                            bias[f] += ep * delta[f][x][y];
                        }
                    }
                }
            });
        }
    }
    
    static class MaxPoolingLayer{
        int size;
        int stride;

        public MaxPoolingLayer(int size, int stride) {
            this.size = size;
            this.stride = stride;
        }
        /** プーリング(max) */
        double[][][] pooling(double[][][] data){
            double[][][] result = new double[data.length][data[0].length / stride][data[0][0].length / stride];
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

        double[][][] backwordPooling(double[][][] in, double[][][] delta){
            double[][][] result = new double[in.length][in[0].length / stride][in[0][0].length / stride];
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
                        result[ch][x][y] = delta[ch][maxX][maxY];
                    }
                }
            });


            return result;
        }
    }
    
    static class FullyConnect{
        double[][] weight;
        double bias;
        int out;
        public FullyConnect(int in, int out) {
            this.out = out;
            weight = Stream.generate(() -> 
                    DoubleStream.generate(r::nextDouble).limit(out).toArray()
            ).limit(in).toArray(len -> new double[len][]);
            bias = r.nextDouble();
        }
        
        public double[] forward(double[] in){
            double[] result = new double[out];
            for(int j = 0; j < out; ++j){
                for(int i = 0; i < in.length; ++i){
                    result[j] += in[i] * weight[i][j];
                }
                result[j] += bias;
            }
            
            return result;
        }
    }
    
    public static void main(String[] args) throws IOException {
        //一段目
        ConvolutionLayer conv1 = new ConvolutionLayer(48, 3, 11, 4);
        //一段目のプーリング
        MaxPoolingLayer pool1 = new MaxPoolingLayer(3, 2);
        //二段目
        ConvolutionLayer conv2 = new ConvolutionLayer(96, 48, 5, 2);
        //二段目のプーリング
        MaxPoolingLayer pool2 = new MaxPoolingLayer(3, 2);
        
        //全結合1
        FullyConnect fc1 = new FullyConnect(6144, 32);
        //全結合2
        FullyConnect fc2 = new FullyConnect(32, 8);
        
        Path p = Paths.get("C:\\Users\\naoki\\Desktop\\sampleimg\\DSC06399.JPG");
        BufferedImage readImg = ImageIO.read(p.toFile());
        BufferedImage resized = resize(readImg, 256, 256);
        double[][][] readData = imageToArray(resized);
        //一段目のフィルタをかける
        double[][][] filterd1 = conv1.applyFilter(readData);
        //プーリング
        double[][][] pooled1 = pool1.pooling(filterd1);
        //二段目のフィルタをかける
        double[][][] filtered2 = conv2.applyFilter(pooled1);
        //プーリング
        double[][][] pooled2 = pool2.pooling(filtered2);
        double[] flattenPooled2 = flatten(pooled2);
        System.out.println(flattenPooled2.length);
        //全結合一段
        double[] fc1out = fc1.forward(flattenPooled2);
        //全結合二段
        double[] fc2out = fc2.forward(fc1out);
        //ソフトマックス
        double[] output = softMax(fc2out);
        System.out.println(Arrays.stream(output).mapToObj(d -> String.format("%.3f", d)).collect(Collectors.joining(",")));
        //全結合二段の逆伝播
        //全結合一段の逆伝播
        //プーリングの逆伝播
        //二段目のフィルタの逆伝播
        //プーリングの逆伝播
        //一段目のフィルタの逆伝播
        //一段目のフィルタの表示
        //フィルタ後の表示
        //全結合一段の表示
        //全結合二段の表示
        
    }
    static void printDim(String name, double[][][] data){
        System.out.printf("%s:%dx%dx%d%n", name,
                data[0].length, data[0][0].length, data.length);
    }

    static double[] softMax(double[] output){
        double total = Arrays.stream(output).parallel()
                .map(d -> Math.exp(d))
                .sum();
        System.out.println(total);
        return Arrays.stream(output).parallel()
                .map(d -> Math.exp(d) / total)
                .toArray();
    }
    
    static Random r = new Random();
    static double[][] createRandomFilter(int size){
        double [][] result = new double[size][size];
        for(int i = 0; i < size; ++i){
            for(int j = 0; j < size; ++j){
                result[i][j] = r.nextDouble();
            }
        }

        
        return result;
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
        return resize(imgRead, width, height, false);
    }
    private static BufferedImage resize(BufferedImage imgRead, int width, int height, boolean inverse) {
        /*
        if(imgRead.getWidth() * height > imgRead.getHeight() * width){
            height = imgRead.getHeight() * width / imgRead.getWidth();
        }else{
            width = imgRead.getWidth() * height / imgRead.getHeight();
        }*/
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics g = img.getGraphics();
        ((Graphics2D)g).setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC);
        if(inverse){
            g.drawImage(imgRead, width, 0, -width, height, null);
        }else{
            g.drawImage(imgRead, 0, 0, width, height, null);
            
        }
        g.dispose();
        return img;
    }    
        
    static BufferedImage arrayToImage(double[][][] filteredData) {
        BufferedImage filtered = new BufferedImage(
                filteredData[0].length, filteredData[0][0].length,
                BufferedImage.TYPE_INT_RGB);
        for(int x = 0; x < filteredData[0].length; ++x){
            for(int y = 0; y < filteredData[0][0].length; ++y){
                filtered.setRGB(x, y,
                        ((int)clip(filteredData[0][x][y] * 255) << 16) +
                        ((int)clip(filteredData[1][x][y] * 255) << 8) +
                         (int)clip(filteredData[2][x][y] * 255));
            }
        }
        return filtered;
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
