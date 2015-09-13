/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.util;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.util.DoubleSummaryStatistics;
import java.util.List;
import java.util.function.DoubleToIntFunction;
import java.util.stream.IntStream;

/**
 *
 * @author naoki
 */
public class ImageUtil {

    public static Image createGraph(int width, int height, float[] data){
        BufferedImage result = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = (Graphics2D) result.getGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, width, height);
        DoubleSummaryStatistics summary = FloatUtil.summary(data);
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
    public static Image createLineGraph(int width, int height, List<Double> data, float max, float min){
        BufferedImage result = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = (Graphics2D) result.getGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, width, height);
        g.setColor(Color.BLACK);
        int step = data.size() / (width * 3) + 1;
        for(int i = step; i < data.size(); i += step){
            g.drawLine((i - 1) * width / data.size(), (int)((data.get(i - step) - max) * (height - 10) / (min - max) - 5)
                    , i * width / data.size(), (int)((data.get(i) - max) * (height - 10)/ (min - max)) - 5);
        }
        g.dispose();
        return result;
    }
    /** 値のクリッピング */
    private static int clip(double c){
        if(c < 0) return 0;
        if(c > 255) return 255;
        return (int)c;
    }

    public static BufferedImage move(BufferedImage imgRead, int width, int height, int ox, int oy){
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics g = img.getGraphics();
        g.drawImage(imgRead, -ox, -oy, null);
        g.dispose();
        return img;
    }

    public static BufferedImage resize(BufferedImage imgRead, int width, int height) {
        return resize(imgRead, width, height, true, false);
    }
    public static BufferedImage resize(BufferedImage imgRead, int width, int height, boolean bicubic, boolean inverse) {
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

    public static BufferedImage arrayToImage(float[] filteredData, int idx, int width, int height) {
        return arrayToImage(filteredData, idx, width, height, 1);
    }

    public static BufferedImage arrayToImage(float[] filteredData, int idx, int width, int height, float rate) {
        DoubleSummaryStatistics summary = FloatUtil.summary(filteredData,
                idx * 3 * width * height, (idx + 1) * 3 * width * height);
        double[] normed = IntStream.range(idx * 3 * width * height, (idx + 1) * 3 * width * height)
                .parallel()
                .mapToDouble(i -> (filteredData[i] - summary.getMin())
                        / (summary.getMax() - summary.getMin()))
                .toArray();
        //rate = 1;
        BufferedImage filtered = new BufferedImage(
                width, height,
                BufferedImage.TYPE_INT_RGB);
        for(int x = 0; x < width; ++x){
            for(int y = 0; y < height; ++y){
                filtered.setRGB(x, y,
                        (clip(normed[x * height + y] * rate * 255) << 16) +
                        (clip(normed[x * height + y + width * height] * rate * 255) << 8) +
                         clip(normed[x * height + y + 2 * width * height] * rate * 255));
            }
        }
        return filtered;
    }
    public static BufferedImage arrayToImageStraight(float[] filteredData, int width, int height){
        BufferedImage filtered = new BufferedImage(
                width, height,
                BufferedImage.TYPE_INT_RGB);
        for(int x = 0; x < width; ++x){
            for(int y = 0; y < height; ++y){
                filtered.setRGB(x, y,
                        ((int)(filteredData[x * height + y] * 255) << 16) +
                        ((int)(filteredData[x * height + y + width * height] * 255) << 8) +
                         (int)(filteredData[x * height + y + 2 * width * height] * 255));
            }
        }
        return filtered;

    }
    public static BufferedImage arrayToImageMono(float[] filteredData, int idx, int width, int height){
        float[] colorData = new float[width * height * 3];
        IntStream.range(0, width).parallel().forEach(x -> {
            for(int y = 0; y < height; ++y){
                int i = x * height + y;
                float c = filteredData[idx * width * height + i];
                colorData[i] = c;
                colorData[i + width * height] = c;
                colorData[i + width * height * 2] = c;
            }
        });
        return arrayToImage(colorData, 0, width, height);
    }
    /** 画像から配列へ変換 */
    public static float[] imageToArray(BufferedImage img) {
        float[] imageData = new float[3 * img.getWidth() * img.getHeight()];
        imageToArray(img, imageData);
        return imageData;
    }
    /** 画像から配列へ変換 */
    public static void imageToArray(BufferedImage img, float[] imageData) {
        int width = img.getWidth();
        int height = img.getHeight();
        for(int x = 0; x < width; ++x){
            for(int y = 0; y < height; ++y){
                int rgb = img.getRGB(x, y);
                int pos = x * height + y;
                imageData[pos] = (rgb >> 16 & 0xff) / 255.f;
                imageData[pos + width * height] = (rgb >> 8 & 0xff) / 255.f;
                imageData[pos + 2 * width * height] = (rgb & 0xff) / 255.f;
            }
        }
    }
    private static float[] normalizeImage(float[] data){
        int size = data.length / 3;
        float[] result = new float[data.length];
        for(int i = 0; i < 3; ++i){
            int start = i * size;
            int end = start + size;
            // 平均
            DoubleSummaryStatistics sum = FloatUtil.summary(data, start, end);
            // 分散
            float dist = (float)Math.sqrt(IntStream.range(start, end).parallel().mapToDouble(idx -> data[idx])
                    .map(d -> (d - sum.getAverage()) * (d - sum.getAverage()))
                    .sum() / sum.getCount());
            // 正規化
            for(int j = start; j < end; ++j){
                result[j] = (float)(data[j] - sum.getAverage()) / dist;
            }
        }
        return result;
    }

}
