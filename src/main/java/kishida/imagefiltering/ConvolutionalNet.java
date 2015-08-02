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
import java.util.Random;
import java.util.stream.IntStream;

/**
 *
 * @author naoki
 */
public class ConvolutionalNet {
    public static void main(String[] args) {
        
    }

    private static BufferedImage resize(BufferedImage imgRead, int width, int height) {
        if(imgRead.getWidth() * height > imgRead.getHeight() * width){
            height = imgRead.getHeight() * width / imgRead.getWidth();
        }else{
            width = imgRead.getWidth() * height / imgRead.getHeight();
        }
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics g = img.getGraphics();
        ((Graphics2D)g).setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC);
        g.drawImage(imgRead, 0, 0, width, height, null);
        g.dispose();
        return img;
    }    
    /** 画像から配列へ変換 */
    private static double[][][] imageToArray(BufferedImage img) {
        int width = img.getWidth();
        int height = img.getHeight();
        double[][][] imageData = new double[3][width][height];
        for(int x = 0; x < width; ++x){
            for(int y = 0; y < height; ++y){
                int rgb = img.getRGB(x, y);
                imageData[0][x][y] = (rgb >> 16 & 0xff) / 128. - 1;
                imageData[1][x][y] = (rgb >> 8 & 0xff) / 128. - 1;
                imageData[2][x][y] = (rgb & 0xff) / 128. - 1;
            }
        }
        return imageData;
    }    
    
    static Random r = new Random();
    static double[][] createRandomFilter(int size){
        double [][] result = new double[size][size];
        double total = 0;
        for(int i = 0; i < size; ++i){
            for(int j = 0; j < size; ++j){
                result[i][j] = (r.nextDouble() - 0.5) * 2;
                total += result[i][j];
            }
        }

        double ave = (total - 1) / (size * size);
        for(int i = 0; i < size; ++i){
            for(int j = 0; j < size; ++j){
                result[i][j] -= ave;
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

    /** フィルタを適用する */
    static double[][][] applyFilter(double[][][] img, double[][][][] filter, int inStride) {
        int width = img[0].length;
        int height = img[0][0].length;
        int filterSize = filter[0][0].length;
        double[][][] result = new double[filter.length][width / inStride][height / inStride];
        for(int lx = 0; lx < width / inStride; ++lx){
            int x = lx;
            for(int ly = 0; ly < height / inStride; ++ly){
                int y = ly;
                for(int li = 0; li < filter[0][0].length; ++li){
                    int i = li;
                    int xx = x * inStride + i - filterSize / 2;
                    if(xx < 0 || xx >= width){
                        continue;
                    }
                    IntStream.range(0, filter[0][0][0].length).parallel().forEach(j -> {
                    //for(int j = 0; j < filter[0][0][0].length; ++j){
                        int yy = y * inStride + j - filterSize / 2;
                        if(yy < 0 || yy >= height){
                            return;//ループを続ける
                        }
                        for(int fi = 0; fi < filter.length; ++fi){
                            for(int fj = 0; fj < filter[fi].length; ++fj){
                                try{
                                result[fi][x][y] += img[fj][xx][yy] * 
                                        filter[fi][fj][i][j];
                                }catch(ArrayIndexOutOfBoundsException ex){
                                    System.out.println(ex);
                                }
                            }
                        }
                    //}
                    });
                }
            }
        }
        return result;
    }
    /** フィルタを適用する */
    static double[][][] applyInverseFilter(double[][][] img, double[][][][] filter, int outStride) {
        int width = img[0].length;
        int height = img[0][0].length;
        double[][][] result = new double[filter[0].length][width * outStride][height * outStride];
        int filterSize = filter[0][0].length;
        for(int lx = 0; lx < width; ++lx){
            int x = lx;
            for(int ly = 0; ly < height; ++ly){
                int y = ly;
                for(int li = 0; li < filter[0][0].length; ++li){
                    int i = li;
                    int xx = x * outStride + i - filterSize / 2;
                    if(xx < 0 || xx >= width * outStride){
                        continue;
                    }
                        
                    //for(int j = 0; j < filter[0][0][0].length; ++j){
                    IntStream.range(0, filter[0][0][0].length).parallel().forEach(j -> {
                        int yy = y * outStride + j - filterSize / 2;
                        if(yy < 0 || yy >= height * outStride){
                            return; // ループを続ける
                        }
                        for(int fi = 0; fi < filter.length; ++fi){
                            for(int fj = 0; fj < filter[fi].length; ++fj){
                                result[fj][xx][yy] += img[fj][x][y] * 
                                        filter[fi][fj][i][j];
                            }
                        }
                    });
                }
            }
        }
        return result;
    }
        
    
    static BufferedImage arrayToImage(double[][][] filteredData) {
        BufferedImage filtered = new BufferedImage(
                filteredData[0].length, filteredData[0][0].length,
                BufferedImage.TYPE_INT_RGB);
        for(int x = 0; x < filteredData[0].length; ++x){
            for(int y = 0; y < filteredData[0][0].length; ++y){
                filtered.setRGB(x, y,
                        ((int)clip(filteredData[0][x][y] * 255 + 128) << 16) +
                        ((int)clip(filteredData[1][x][y] * 255 + 128) << 8) +
                         (int)clip(filteredData[2][x][y] * 255 + 128));
            }
        }
        return filtered;
    }        
}
