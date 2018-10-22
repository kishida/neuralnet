package kishida.cnn;

import kishida.cnn.layers.ImageNeuralLayer;
import kishida.cnn.layers.ConvolutionLayer;
import kishida.cnn.layers.FullyConnect;
import kishida.cnn.layers.NeuralLayer;
import java.awt.GridLayout;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.io.UncheckedIOException;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.DoubleSummaryStatistics;
import java.util.LinkedList;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTabbedPane;
import kishida.cnn.kernels.ConvolutionBackwordKernel;
import kishida.cnn.kernels.ConvolutionForwardKernel;
import kishida.cnn.layers.LerningLayer;
import static kishida.cnn.util.ImageUtil.*;

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
    private static final String FILENAME = "C:\\Users\\naoki\\Desktop\\alexnet.json.txt";
    private static final String RESOURCE_NAME = "/alexnet_def.json";
    //private static final String FILENAME = "C:\\Users\\naoki\\Desktop\\tinynet.json.txt";
    //private static final String RESOURCE_NAME = "/tinynet_def.json";

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

    @SuppressWarnings({"ThrowableInstanceNotThrown", "ThrowableInstanceNeverThrown"})
    public static void main(String[] args) throws IOException {
        System.setProperty("com.aparapi.enableShowGeneratedOpenCL", "false");
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

                .flatMap(p -> IntStream.range(0, 3).mapToObj(i ->
                        IntStream.range(0, 3).mapToObj(j ->
                                Stream.of(new Img(p, true, i, j), new Img(p, false, i, j)))
                                .flatMap(Function.identity())).flatMap(Function.identity()))

                //.map(p -> new Img(p, false, 0, 0))
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

        /*
        List<NeuralLayer> layers = new ArrayList<>();
        layers.addAll(Arrays.asList(
            new InputLayer("input", 227, 227),
            //一段目
            new ConvolutionLayer("conv1", FILTER_1ST, FILTER_1ST_SIZE, 4, 0, USE_GPU1),
            //一段目のプーリング
            new MaxPoolingLayer("pool1", 3, 2),
            //一段目の正規化
            //layers.add(pre = new NormalizeLayer("norm1", 5, .01, pre, USE_GPU1));
            new MultiNormalizeLayer("norm1", 5, .000001f, USE_GPU1),
            //二段目
            new ConvolutionLayer("conv2", FILTER_2ND, 5, 1, 1, USE_GPU2),
            //二段目のプーリング
            new MaxPoolingLayer("pool2", 3, 2),
            //二段目の正規化
            new MultiNormalizeLayer("norm2", 5, .000001f, USE_GPU2),

            new ConvolutionLayer("conv3", 384, 3, 1, 0, USE_GPU1),
            new ConvolutionLayer("conv4", 384, 3, 1, 1, USE_GPU1),
            new ConvolutionLayer("conv5", 256, 3, 1, 1, USE_GPU1),
            new MaxPoolingLayer("pool5", 3, 2),
            new FullyConnect("fc0", 4096, 1, .5f, new RectifiedLinear(), false),

            //全結合1
            new FullyConnect("fc1", FULL_1ST, 1, 0.5f, new RectifiedLinear(), USE_GPU1),
            //全結合2
            new FullyConnect("fc2", categories.size(), 1, 1, new SoftMaxFunction(), false)
        ));

        NeuralNetwork nn = new NeuralNetwork(learningRate, weightDecay, MINI_BATCH, MOMENTAM,
                1234, 2345, layers);
        */
        /*
        try(Writer w = Files.newBufferedWriter(Paths.get("C:\\Users\\naoki\\Desktop\\tinynet_def.json.txt"))){
            nn.writeAsJson(w);
        }*/

        NeuralNetwork nn;

        if(true){
            try(InputStream is = ConvolutionalNet.class.getResourceAsStream(RESOURCE_NAME);
                    InputStreamReader isr = new InputStreamReader(is)){
                nn = NeuralNetwork.readFromJson(isr);
            }
        }else{
            try(Reader r = Files.newBufferedReader(Paths.get(FILENAME))){
                nn = NeuralNetwork.readFromJson(r);
            }
        }

        nn.init();
        nn.getLayers().forEach(System.out::println);
        FullyConnect fc1 = (FullyConnect)nn.findLayerByName("fc1")
                .orElseThrow(() -> new IllegalArgumentException("fc1 not found"));
        FullyConnect fc2 = (FullyConnect)nn.findLayerByName("fc2")
                .orElseThrow(() -> new IllegalArgumentException("fc2 not found"));
        ConvolutionLayer conv1 = (ConvolutionLayer)nn.findLayerByName("conv1")
            .orElseThrow(() -> new IllegalArgumentException("conv1 not found"));
        ConvolutionLayer conv2 = (ConvolutionLayer)nn.findLayerByName("conv2")
                .orElseThrow(() -> new IllegalArgumentException("conv2 not found"));
        ImageNeuralLayer pool1 = (ImageNeuralLayer)nn.findLayerByName("pool1")
                .orElseThrow(() -> new IllegalArgumentException("pool1 not found"));
        ImageNeuralLayer norm1 = (ImageNeuralLayer)nn.findLayerByName("norm1")
                .orElseThrow(() -> new IllegalArgumentException("norm1 not found"));
        int[] lastHour = {LocalTime.now().getHour()};
        int[] count = {0};
        int[] batchCount = {0};
        for(; nn.getLoop() < 30; nn.setLoop(nn.getLoop() + 1)){
            nn.saveImageRandomState();
            Collections.shuffle(files, nn.getImageRandom());
            long start = System.currentTimeMillis();
            long[] pStart = {start};
            float[] readData = new float[3 * IMAGE_SIZE * IMAGE_SIZE];
            files.stream().skip(nn.getImageIndex()).forEach(img -> {
                if(count[0] == 0){
                    for(int i = 0; i < conv1.getOutputChannels(); ++i){
                        filtersLabel[i].setIcon(new ImageIcon(resize(arrayToImage(
                                conv1.getFilter(), i, FILTER_1ST_SIZE, FILTER_1ST_SIZE), 44, 44, false, false)));
                    }
                    //フィルタ後の表示
                    for(int i = 0; i < conv1.getOutputChannels(); ++i){
                        filteredLabel[i].setIcon(new ImageIcon(arrayToImageMono(
                                conv1.getResult(), i, conv1.getOutputWidth(), conv1.getOutputHeight())));
                    }
                    for(int i = 0; i < pool1.getOutputChannels(); ++i){
                        pooledLabel[i].setIcon(new ImageIcon(resize(arrayToImageMono(
                                pool1.getResult(), i, pool1.getOutputWidth(), pool1.getOutputHeight()), 48, 48)));
                    }
                    for(int i = 0; i < Math.min(normedLabel.length, norm1.getOutputChannels()); ++i){
                        normedLabel[i].setIcon(new ImageIcon(resize(arrayToImageMono(
                                norm1.getResult(), i, norm1.getOutputWidth(), norm1.getOutputHeight()), 48, 48)));
                    }
                }

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
                        historyData, 1, 0);
                historyLabel.setIcon(new ImageIcon(lineGraph));

                //System.out.println(Arrays.stream(output).mapToObj(d -> String.format("%.2f", d)).collect(Collectors.joining(",")));

                count[0]++;
                nn.setImageIndex(nn.getImageIndex() + 1);
                if(count[0] >= MINI_BATCH){

                    nn.joinBatch();
                    batchCount[0]++;
                    System.out.printf("%5d %4d %.2f/m %s %s%n", batchCount[0],
                            count[0], MINI_BATCH * 60 * 1000. / (System.currentTimeMillis() - pStart[0]),
                            ConvolutionForwardKernel.INSTANCE.getExecutionMode(),
                            ConvolutionBackwordKernel.INSTANCE.getExecutionMode());

                    for(NeuralLayer layer : nn.getLayers()){
                        System.out.printf("%s result: %.2f～%.2f average %.2f ", layer.getName(),
                                layer.getResultStatistics().getMin(),
                                layer.getResultStatistics().getMax(),
                                layer.getResultStatistics().getAverage());
                        if(layer instanceof LerningLayer){
                            DoubleSummaryStatistics ws = ((LerningLayer)layer).getWeightStatistics();
                            System.out.printf("weight: %.2f～%.2f average %.2f ",
                                    ws.getMin(), ws.getMax(), ws.getAverage());
                            DoubleSummaryStatistics bs = ((LerningLayer)layer).getBiasStatistics();
                            System.out.printf("bias: %.8f～%.8f average %.2f ",
                                    bs.getMin(), bs.getMax(), bs.getAverage());
                        }
                        System.out.println();
                    }

                    count[0] = 0;
                    pStart[0] = System.currentTimeMillis();
                    nn.prepareBatch();

                    //一段目のフィルタの表示
                    //全結合一段の表示
                    firstFc.setIcon(new ImageIcon(createGraph(256, 128, fc1.getResult())));
                    //全結合二段の表示
                    lastResult.setIcon(new ImageIcon(createGraph(256, 128, output)));

                    firstBias.setIcon(new ImageIcon(createGraph(500, 128, conv1.getBias())));
                    secondBias.setIcon(new ImageIcon(createGraph(500, 128,
                            conv2.getBias())));
                    fc1Bias.setIcon(new ImageIcon(createGraph(500, 128, fc1.getBias())));
                    fc2Bias.setIcon(new ImageIcon(createGraph(500, 128, fc2.getBias())));

                    // 1時間に一回保存する
                    int hour = LocalTime.now().getHour();
                    if(lastHour[0] != hour){
                        lastHour[0] = hour;
                        try(Writer w = Files.newBufferedWriter(Paths.get(FILENAME))){
                            nn.writeAsJson(w);
                        }catch(IOException ex){
                            throw new UncheckedIOException(ex);
                        }
                    }
                }
            });
            long end = System.currentTimeMillis();
            System.out.println(end - start);
            System.out.printf("%.2fm%n", (end - start) / 1000. / 60);
            nn.setImageIndex(0);
        }
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

}
