/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.imagefiltering;

import java.awt.Graphics;
import java.awt.GridLayout;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.SwingConstants;

/**
 *
 * @author naoki
 */
public class InverseFilter {
    public static void main(String... args) throws IOException {
        
        JFileChooser fc = new JFileChooser();
        fc.setDialogTitle("フィルタする画像");
        int dialogResult = fc.showOpenDialog(null);
        if(dialogResult != JFileChooser.APPROVE_OPTION){
            return;
        }
        File imageFile = fc.getSelectedFile();

        JFrame f = new JFrame("Gabor逆フィルタ");
        f.setLayout(new GridLayout(3, 2));

        double gamma = 0.7;
        double sigma = 0.3;
        double[][] viewFilter = createGabor(300, Math.PI / 4, gamma, sigma);
        BufferedImage filterImage = new BufferedImage(300, 300, BufferedImage.TYPE_INT_RGB);
        for(int x = 0; x < 300; ++x){
            for(int y = 0; y < 300; ++y){
                int c = (int)((viewFilter[x][y] + 1) / 2 * 255);
                filterImage.setRGB(x, y, (c << 16) + (c << 8) + c);
            }
        }
        f.add(createLabel("フィルタ", filterImage));

        BufferedImage imgRead = ImageIO.read(imageFile); // 適当な画像を指定
        int width = 400, height = 300;
        if(imgRead.getWidth() * height > imgRead.getHeight() * width){
            height = imgRead.getHeight() * width / imgRead.getWidth();
        }else{
            width = imgRead.getWidth() * height / imgRead.getHeight();
        }
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics g = img.getGraphics();
        g.drawImage(imgRead, 0, 0, width, height, null);
        g.dispose();

        double[][][] imageData = new double[3][width][height];
        for(int x = 0; x < width; ++x){
            for(int y = 0; y < height; ++y){
                int rgb = img.getRGB(x, y);
                imageData[0][x][y] = (rgb >> 16 & 0xff) / 255.;
                imageData[1][x][y] = (rgb >> 8 & 0xff) / 255.;
                imageData[2][x][y] = (rgb & 0xff) / 255.;
            }
        }


        f.add(createLabel("オリジナル", img));

        String[] names = {"縦", "ななめ", "横", "ななめ"};
        
        int filterSize = 3;
        double[][] blankFilter = new double[filterSize][filterSize];
        
        for(int i = 0; i < 4; ++i){
            double[][] baseFilter = createGabor(filterSize, Math.PI / 4 * i, gamma, sigma);
            double[][][][] filters = new double[][][][]{
                {baseFilter, blankFilter, blankFilter},
                {blankFilter, baseFilter, blankFilter},
                {blankFilter, blankFilter, baseFilter}
            };
            double[][][] filteredData = applyFilter(imageData, filters);
            //BufferedImage filtered = arrayToImage(filteredData);
            double[][][][] inverse = invertFilter(filters);
            double[][][] inverseData = applyFilter(filteredData, inverse);
            BufferedImage filtered = arrayToImage(inverseData);
            f.add(createLabel(String.format("フィルター%d(%s)",i , names[i]), filtered));
        }

        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        f.setSize(800, 1200);
        f.setVisible(true);
    }

    static double[][][][] invertFilter(double[][][][] filter){
        double[][][][] result = new double[filter.length][filter[0].length]
                [filter[0][0].length][filter[0][0][0].length];
        for(int fi = 0; fi < filter.length; ++fi){
            for(int fj = 0; fj < filter[fi].length; ++fj){
                for(int i = 0; i < filter[fi][fj].length; ++i){
                    for(int j = 0; j < filter[fi][fj][i].length; ++j){
                        result[fi][fj][i][j] = filter[fi][fj]
                                [filter[fi][fj].length - i - 1]
                                [filter[fi][fj][0].length - j - 1];
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

    /** ラベル生成 */
    static JLabel createLabel(String label, BufferedImage img){
        JLabel l = new JLabel(label, SwingConstants.CENTER);
        l.setVerticalTextPosition(SwingConstants.TOP);
        l.setHorizontalTextPosition(SwingConstants.CENTER);
        if(img != null){
            l.setIcon(new ImageIcon(img));
        }
        return l;
    }

    /**
     * ガボールフィルターの作成
     * @param h サイズ
     * @param theta 傾き
     * @param gamma
     * @param sigma
     * @return
     */
    static double[][] createGabor(int h, double theta, double gamma, double sigma) {
        double[][] filter = new double[h][h];
        double total = 0;
        for(int x = 0; x < h; ++x){
            for(int y = 0; y < h; ++y){
                double nx = x * 2 / (double)h - 1;
                double ny = y * 2 / (double)h - 1;
                double xx = nx * Math.cos(theta) + ny * Math.sin(theta);
                double yy = - nx * Math.sin(theta) + ny * Math.cos(theta);
                double w = Math.exp( - (xx * xx + gamma * gamma * yy * yy ) / (2 * sigma * sigma));
                double phai = 0;
                filter[x][y] = w * Math.cos(xx * Math.PI * 2.5  + phai);
                total += filter[x][y];
            }
        }
        total /= h * h;
        for(int x = 0; x < h; ++x){
            for(int y = 0; y < h; ++y){
                filter[x][y] -= total;
            }
        }
        return filter;
    }

    /** 値のクリッピング */
    static int clip(double c){
        if(c < 0) return 0;
        if(c > 255) return 255;
        return (int)c;
    }

    /** フィルタを適用する */
    static double[][][] applyFilter(double[][][] img, double[][][][] filter) {
        int width = img[0].length - filter[0][0].length + 1;
        int height = img[0][0].length - filter[0][0].length + 1;
        double[][][] result = new double[filter.length][width][height];
        for(int x = 0; x < width; ++x){
            for(int y = 0; y < height; ++y){
                for(int i = 0; i < filter.length; ++i){
                    result[i][x][y] = 0;
                }
                for(int i = 0; i < filter[0][0].length; ++i){
                    for(int j = 0; j < filter[0][0][0].length; ++j){
                        for(int fi = 0; fi < filter.length; ++fi){
                            for(int fj = 0; fj < filter[fi].length; ++fj){
                                result[fi][x][y] += img[fj][x + i][y + j] * 
                                        filter[fi][fj][i][j];
                            }
                        }
                    }
                }
            }
        }
        return result;
    }
}
