/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.imagefiltering;

import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.GridLayout;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Random;
import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

/**
 *
 * @author naoki
 */
public class RandomFilter {
    
    public static void main(String[] args) throws IOException {
        JFrame f = new JFrame("ランダムフィルタ");
        f.setLayout(new GridLayout(2, 2));
        File imageFile = new File("C:\\Users\\naoki\\Desktop\\1353908241630o.jpg");
        BufferedImage readImg = ImageIO.read(imageFile);
        BufferedImage resized = resize(readImg, 600, 400);
        JLabel lbl = new JLabel(new ImageIcon(resized));
        f.add(lbl);
        
        int size = 11;
        double [][][][] filter = new double[3][][][];
        for(int i = 0; i < filter.length; ++i){
            filter[i] = new double[][][]{
                createRandomFilter(size), createRandomFilter(size), createRandomFilter(size)};
        }
        double[][][] img = imageToArray(resized);
        double[][][] filtered = applyFilter(img, filter, 4);
        BufferedImage filteredImg = arrayToImage(filtered);
        f.add(new JLabel(new ImageIcon(filteredImg)));
        
        double[][][] inverse = applyInverseFilter(filtered, filter, 4);
        BufferedImage inverseImg = arrayToImage(inverse);
        f.add(new JLabel(new ImageIcon(inverseImg)));
        
        double[][][] refiltered = applyFilter(inverse, filter, 4);
        f.add(new JLabel(new ImageIcon(arrayToImage(refiltered))));
        
        f.setSize(500, 400);
        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        f.setVisible(true);
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
                imageData[0][x][y] = (rgb >> 16 & 0xff) / 255.;
                imageData[1][x][y] = (rgb >> 8 & 0xff) / 255.;
                imageData[2][x][y] = (rgb & 0xff) / 255.;
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
        if(false){
            for(int i = 0; i < size; ++i){
                for(int j = 0; j < size; ++j){
                    result[i][j] -= total / size / size;
                }
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
        for(int x = 0; x < width / inStride; ++x){
            for(int y = 0; y < height / inStride; ++y){
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
                    }
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
        for(int x = 0; x < width; ++x){
            for(int y = 0; y < height; ++y){
                for(int i = 0; i < filter[0][0].length; ++i){
                    int xx = x * outStride + i - filterSize / 2;
                    if(xx < 0 || xx >= width * outStride){
                        continue;
                    }
                        
                    for(int j = 0; j < filter[0][0][0].length; ++j){
                        int yy = y * outStride + j - filterSize / 2;
                        if(yy < 0 || yy >= height * outStride){
                            continue;
                        }
                        for(int fi = 0; fi < filter.length; ++fi){
                            for(int fj = 0; fj < filter[fi].length; ++fj){
                                result[fj][xx][yy] += img[fj][x][y] * 
                                        filter[fi][fj][i][j];
                            }
                        }
                    }
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
                        ((int)clip(filteredData[0][x][y] * 255) << 16) +
                        ((int)clip(filteredData[1][x][y] * 255) << 8) +
                         (int)clip(filteredData[2][x][y] * 255));
            }
        }
        return filtered;
    }    
}
