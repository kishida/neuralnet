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
 * @author kishida
 */
public class GaborFilter {
    public static void main(String... args) throws IOException {
		JFileChooser fc = new JFileChooser();
		fc.setDialogTitle("フィルタする画像");
		int dialogResult = fc.showOpenDialog(null);
		if(dialogResult != JFileChooser.APPROVE_OPTION){
			return;
		}
		File imageFile = fc.getSelectedFile();

        JFrame f = new JFrame("Gaborフィルタ");
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
			height = imgRead.getHeight() * height / imgRead.getWidth();
		}else{
			width = imgRead.getWidth() * width / imgRead.getHeight();
		}
		BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		Graphics g = img.getGraphics();
		g.drawImage(imgRead, 0, 0, width, height, null);
		g.dispose();

        f.add(createLabel("オリジナル", img));

        String[] names = {"縦", "ななめ", "横", "ななめ"};
        for(int i = 0; i < 4; ++i){
            double[][] filter = createGabor(9, Math.PI / 4 * i, gamma, sigma);
            BufferedImage filtered = applyFilter(img, filter);
            f.add(createLabel(String.format("フィルター%d(%s)",i , names[i]), filtered));
        }

        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        f.setSize(800, 1200);
        f.setVisible(true);
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
    static BufferedImage applyFilter(BufferedImage img, double[][] filter) {
        int width = img.getWidth() - filter.length + 1;
        int height = img.getHeight() - filter.length + 1;
        BufferedImage result = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for(int x = 0; x < width; ++x){
            for(int y = 0; y < height; ++y){
                double r = 0;
                double g = 0;
                double b = 0;
                for(int i = 0; i < filter.length; ++i){
                    for(int j = 0; j < filter.length; ++j){
                        int rgb = img.getRGB(x + i, y + j);
                        double f = filter[i][j];
                        r += ((rgb >> 16) & 255) * f;
                        g += ((rgb >> 8) & 255) * f;
                        b += (rgb & 255) * f;
                    }
                }
                result.setRGB(x, y, (clip(r) << 16) + (clip(g) << 8) + clip(b));
            }
        }
        return result;
    }
}
