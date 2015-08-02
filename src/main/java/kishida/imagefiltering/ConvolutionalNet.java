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
import java.nio.file.Path;
import java.util.Random;
import java.util.stream.IntStream;

/**
 *
 * @author naoki
 */
public class ConvolutionalNet {
    static final int stride = 4;
    static final double ep = 0.000000001;

    static class Img{

        public Img(Path filename, boolean inverse) {
            this.filename = filename;
            this.inverse = inverse;
        }
        Path filename;
        boolean inverse;
    }
    
    public static void main(String[] args) {
        
    }
    /** 畳み込み層の学習 */
    static void convolutionalLearn(double[][][] delta, double[][][][] filters, double[] bias, double[][][] input, int step ){
        IntStream.range(0, filters.length).parallel().forEach(f -> {
            for(int ch = 0; ch < filters[0].length; ++ch){
                for(int x = 0; x < input[0].length / step; ++x){
                    for(int y = 0; y < input[0][0].length / step; ++y){
                        for(int i = 0; i < filters[0][0].length; ++i){
                            int xx = x * step + i - filters[0][0].length / 2;
                            if(xx < 0 || xx >= input[0].length){
                                continue;
                            }
                            for(int j = 0; j < filters[0][0][0].length; ++j){
                                int yy = y * step + j - filters[0][0][0].length / 2;
                                if(yy < 0 || yy >= input[0][0].length){
                                    continue;
                                }
                                double d = (input[ch][xx][yy] > 0 ? 1 : 0) * delta[f][x][y];
                                filters[f][ch][i][j] += d * ep;
                            }
                        }
                        bias[f] += ep * delta[f][x][y];
                    }
                }
            }
        });
    }

    static Random r = new Random();
    static double[][] createRandomFilter(int size){
        double [][] result = new double[size][size];
        for(int i = 0; i < size; ++i){
            for(int j = 0; j < size; ++j){
                result[i][j] = (r.nextDouble() - 0.5) * 2;
            }
        }

        
        return result;
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
        if(imgRead.getWidth() * height > imgRead.getHeight() * width){
            height = imgRead.getHeight() * width / imgRead.getWidth();
        }else{
            width = imgRead.getWidth() * height / imgRead.getHeight();
        }
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
    
    /** 畳み込みフィルタを適用する */
    static double[][][] applyFilter(double[][][] img, double[][][][] filter, double[] bias, int inStride) {
        int width = img[0].length;
        int height = img[0][0].length;
        int filterSize = filter[0][0].length;
        double[][][] result = new double[filter.length][width / inStride][height / inStride];
        IntStream.range(0, filter.length).parallel().forEach(fi ->{
            for(int x = 0; x < width / inStride; ++x){
                for(int y = 0; y < height / inStride; ++y){
                    for(int ch = 0; ch < filter[fi].length; ++ch){
                        for(int i = 0; i < filter[0][0].length; ++i){
                            int xx = x * inStride + i - filterSize / 2;
                            if(xx < 0 || xx >= width){
                                continue;
                            }
                            for(int j = 0; j < filter[0][0][0].length; ++j){
                                int yy = y * inStride + j - filterSize / 2;
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
