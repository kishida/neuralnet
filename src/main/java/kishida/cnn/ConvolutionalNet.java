package kishida.cnn;

import kishida.cnn.layers.ImageNeuralLayer;
import kishida.cnn.layers.MaxPoolingLayer;
import kishida.cnn.layers.ConvolutionLayer;
import kishida.cnn.layers.FullyConnect;
import kishida.cnn.layers.InputLayer;
import kishida.cnn.layers.NeuralLayer;
import kishida.cnn.activation.SoftMaxFunction;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.GridLayout;
import java.awt.Image;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.DoubleSummaryStatistics;
import java.util.LinkedList;
import java.util.List;
import java.util.function.DoubleToIntFunction;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTabbedPane;
import kishida.cnn.activation.RectifiedLinear;
import kishida.cnn.kernels.ConvolutionBackwordKernel;
import kishida.cnn.kernels.ConvolutionForwardKernel;
import kishida.cnn.layers.LerningLayer;
import kishida.cnn.layers.MultiNormalizeLayer;
import kishida.cnn.util.FloatUtil;

/**
 *
 * @author naoki
 */
public class ConvolutionalNet {
    private static final float learningRate = 0.01f;
    private static final float weightDecay = 0.0005f;
    private static final boolean USE_GPU1 = true;
    private static final boolean USE_GPU2 = true;
    private static final int FILTER_1ST = 96;
    private static final int FILTER_2ND = 256;
    private static final int FULL_1ST = 4096;
    private static final int FILTER_1ST_SIZE = 11;
    //static final int FILTER_1ST = 48;
    //static final int FILTER_2ND = 96;
    private static final int FILTER_ROWS = 8;//Math.max((int)Math.sqrt(FILTER_1ST), 1);
    private static final int FILTER_COLS = 12;//FILTER_1ST / h;
    private static final int IMAGE_SIZE = 227;
    private static final int MINI_BATCH = 128;
    private static final float MOMENTAM = 0.9f;
    public static final String AVERAGE_PNG = "average.png";

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

        BufferedImage readImage(){
            BufferedImage readImg;
            try {
                readImg = ImageIO.read(filename.toFile());
            } catch (IOException ex) {
                throw new UncheckedIOException(ex);
            }
            BufferedImage resized = resize(readImg, 256 + 32, 256 + 32, true, inverse);
            BufferedImage moved = resize(move(resized, 256, 256, x * 16, y * 16), IMAGE_SIZE, IMAGE_SIZE);
            return moved;
        }
    }

    static List<Double> historyData = new ArrayList<>();
    static LinkedList<Integer> rateData = new LinkedList<>();

    public static void main(String[] args) throws IOException {
        System.setProperty("com.amd.aparapi.enableShowGeneratedOpenCL", "false");
        String def = "C:\\Users\\naoki\\Desktop\\sampleimg288";
        Path dir = Paths.get(args.length > 0 ? args[0] : def);
        List<String> categories = Files.list(dir)
                .filter(p -> Files.isDirectory(p))
                .map(p -> p.getFileName().toString())
                .filter(n -> !n.startsWith("_"))
                .collect(Collectors.toList());

        JFrame f = createFrame();
        f.setVisible(true);


        List<Img> files = Files.walk(dir)
                .filter(p -> !Files.isDirectory(p))
                .filter(p -> !AVERAGE_PNG.equals(p.getFileName().toString()))
                .filter(p -> !p.getParent().getFileName().toString().startsWith("_"))
                /*
                .flatMap(p -> IntStream.range(0, 3).mapToObj(i ->
                        IntStream.range(0, 3).mapToObj(j ->
                                Stream.of(new Img(p, true, i, j), new Img(p, false, i, j)))
                                .flatMap(Function.identity())).flatMap(Function.identity()))
                */
                .map(p -> new Img(p, true, 0, 0))
                .collect(Collectors.toList());


        // 画素ごとの平均をとる
        Path avePath = dir.resolve(AVERAGE_PNG);
        BufferedImage aveImage;
        if(!Files.exists(avePath)){
            float[] aveData = new float[IMAGE_SIZE * IMAGE_SIZE * 3];
            int imgCount = 0;
            for(Img img : files){
                float[] imageArray = imageToArray(img.readImage());
                IntStream.range(0, imageArray.length).parallel().forEach( i -> {
                    aveData[i] += imageArray[i];
                });
                ++imgCount;
                if(imgCount % 100 == 0){
                    System.out.println(imgCount);
                }
            }
            for(int i = 0; i < aveData.length; ++i){
                aveData[i] /= files.size();
            }
            aveImage = arrayToImageStraight(aveData, IMAGE_SIZE, IMAGE_SIZE);
            ImageIO.write(aveImage, "png", avePath.toFile());
        }else{
            aveImage = ImageIO.read(avePath.toFile());
        }
        org.setIcon(new ImageIcon(aveImage));
        float[] aveData = imageToArray(aveImage);

        List<NeuralLayer> layers = new ArrayList<>();
        InputLayer input = new InputLayer(227, 227);
        layers.add(input);

        //一段目
        layers.add(new ConvolutionLayer("conv1", FILTER_1ST, FILTER_1ST_SIZE, 4, 0, USE_GPU1));
        //一段目のプーリング
        layers.add(new MaxPoolingLayer("pool1", 3, 2));
        //一段目の正規化
        //layers.add(pre = new NormalizeLayer("norm1", 5, .01, pre, USE_GPU1));
        layers.add(new MultiNormalizeLayer("norm1", 5, .000001f, USE_GPU1));
        //二段目
        layers.add(new ConvolutionLayer("conv2", FILTER_2ND, 5, 1, 1, USE_GPU2));
        //二段目のプーリング
        layers.add(new MaxPoolingLayer("pool2", 3, 2));

        //layers.add(pre = new NormalizeLayer("norm2", 5, .01, pre, USE_GPU2));
        layers.add(new MultiNormalizeLayer("norm2", 5, .000001f, USE_GPU2));

        layers.add(new ConvolutionLayer("conv3", 384, 3, 1, 0, USE_GPU1));
        layers.add(new ConvolutionLayer("conv4", 384, 3, 1, 1, USE_GPU1));
        layers.add(new ConvolutionLayer("conv5", 256, 3, 1, 1, USE_GPU1));
        layers.add(new MaxPoolingLayer("pool5", 3, 2));
        layers.add(new FullyConnect("fc0", 4096, 1, .5f, new RectifiedLinear(), USE_GPU1));

        //全結合1
        FullyConnect fc1 = new FullyConnect("fc1", FULL_1ST, 1, 0.5f, new RectifiedLinear(), USE_GPU1);
        layers.add(fc1);
        //全結合2
        FullyConnect fc2 = new FullyConnect("fc2", categories.size(), 1, 1, new SoftMaxFunction(), false);
        layers.add(fc2);

        NeuralNetwork nn = new NeuralNetwork(learningRate, weightDecay, MINI_BATCH, MOMENTAM,
                1234, 2345, layers);
        nn.init();
        layers.forEach(System.out::println);

        int[] count = {0};
        for(int loop = 0; loop < 30; ++loop){
            Collections.shuffle(files, nn.imageRandom);
            long start = System.currentTimeMillis();
            long[] pStart = {start};
            float[] readData = new float[3 * IMAGE_SIZE * IMAGE_SIZE];
            for(Img img : files) {
                Path p = img.filename;
                String catName = p.getParent().getFileName().toString();
                float[] correctData = new float[categories.size()];
                for(int i = 0; i < categories.size(); ++i){
                    correctData[i] = categories.get(i).equals(catName) ? 1 : 0;
                }

                BufferedImage resized = img.readImage();
                //float[] readData = normalizeImage(imageToArray(resized));
                imageToArray(resized, readData);
                for(int i = 0; i < readData.length; ++i){
                    readData[i] -= aveData[i];
                }

                float[] output = nn.forward(readData, correctData);
                //元画像の表示

                org.setIcon(new ImageIcon(resized));

                //判定結果
                float max = Float.NEGATIVE_INFINITY;
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
                if(count[0] >= MINI_BATCH){
                    layers.forEach(layer -> layer.joinBatch());
                    for(int i = 0; i < conv1.getOutputChannels(); ++i){
                        filtersLabel[i].setIcon(new ImageIcon(resize(arrayToImage(
                                conv1.getFilter(), i, FILTER_1ST_SIZE, FILTER_1ST_SIZE), 44, 44, false, false)));
                    }
                    //フィルタ後の表示
                    for(int i = 0; i < conv1.getOutputChannels(); ++i){
                        filteredLabel[i].setIcon(new ImageIcon(arrayToImageMono(
                                conv1.getResult(), i, conv1.getOutputWidth(), conv1.getOutputHeight())));
                    }
                    ImageNeuralLayer pool1 = (ImageNeuralLayer) layers.get(2);
                    for(int i = 0; i < pool1.getOutputChannels(); ++i){
                        pooledLabel[i].setIcon(new ImageIcon(resize(arrayToImageMono(
                                pool1.getResult(), i, pool1.getOutputWidth(), pool1.getOutputHeight()), 48, 48)));
                    }
                    ImageNeuralLayer norm1 = (ImageNeuralLayer) layers.get(3);
                    for(int i = 0; i < Math.min(normedLabel.length, norm1.getOutputChannels()); ++i){
                        normedLabel[i].setIcon(new ImageIcon(resize(arrayToImageMono(
                                norm1.getResult(), i, norm1.getOutputWidth(), norm1.getOutputHeight()), 48, 48)));
                    }

                    System.out.printf("%4d %.2f %s %s%n",
                            count[0], MINI_BATCH * 60 * 1000. / (System.currentTimeMillis() - pStart[0]),
                            ConvolutionForwardKernel.INSTANCE.getExecutionMode(),
                            ConvolutionBackwordKernel.INSTANCE.getExecutionMode());

                    for(NeuralLayer layer : layers){
                        System.out.printf("%s result: %.2f～%.2f average %.2f ", layer.getName(),
                                layer.getResultStatistics().getMin(),
                                layer.getResultStatistics().getMax(),
                                layer.getResultStatistics().getAverage());
                        if(layer instanceof LerningLayer){
                            DoubleSummaryStatistics ws = ((LerningLayer)layer).getWeightStatistics();
                            System.out.printf("weight: %.2f～%.2f average %.2f ",
                                    ws.getMin(), ws.getMax(), ws.getAverage());
                            DoubleSummaryStatistics bs = ((LerningLayer)layer).getBiasStatistics();
                            System.out.printf("bias: %.2f～%.2f average %.2f ",
                                    bs.getMin(), bs.getMax(), bs.getAverage());
                        }
                        System.out.println();
                    }

                    count[0] = 0;
                    pStart[0] = System.currentTimeMillis();
                    layers.forEach(layer -> layer.prepareBatch());

                    try(Writer w = Files.newBufferedWriter(Paths.get("C:\\Users\\naoki\\Desktop\\alexnet.json.txt"))){
                        nn.writeAsJson(w);
                    }
                }
            }
            long end = System.currentTimeMillis();
            System.out.println(end - start);
            System.out.printf("%.2fm%n", (end - start) / 1000. / 60);
        }
    }

    static Image createGraph(int width, int height, float[] data){
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
    static Image createLineGraph(int width, int height, double[] data, float max, float min){
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

    static void printDim(String name, float[][][] data){
        System.out.printf("%s:%dx%dx%d%n", name,
                data[0].length, data[0][0].length, data.length);
    }
    static void printData(float[][] data){
        System.out.println(Arrays.stream(data)
                .map(da -> IntStream.range(0, da.length)
                        .mapToObj(d -> String.format("%.3f", da[d]))
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

    static BufferedImage arrayToImage(float[] filteredData, int idx, int width, int height) {
        return arrayToImage(filteredData, idx, width, height, 1);
    }

    static BufferedImage arrayToImage(float[] filteredData, int idx, int width, int height, float rate) {
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
    static BufferedImage arrayToImageStraight(float[] filteredData, int width, int height){
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
    static BufferedImage arrayToImageMono(float[] filteredData, int idx, int width, int height){
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
    private static float[] imageToArray(BufferedImage img) {
        float[] imageData = new float[3 * img.getWidth() * img.getHeight()];
        imageToArray(img, imageData);
        return imageData;
    }
    /** 画像から配列へ変換 */
    private static void imageToArray(BufferedImage img, float[] imageData) {
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
    static float[] normalizeImage(float[] data){
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
