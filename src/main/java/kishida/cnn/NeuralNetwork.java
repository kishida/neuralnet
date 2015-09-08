/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.io.StringWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Random;
import kishida.cnn.activation.LogisticFunction;
import kishida.cnn.layers.ConvolutionLayer;
import kishida.cnn.layers.FullyConnect;
import kishida.cnn.layers.InputLayer;
import kishida.cnn.layers.MaxPoolingLayer;
import kishida.cnn.layers.MultiNormalizeLayer;
import kishida.cnn.layers.NeuralLayer;
import kishida.cnn.util.FloatUtil;
import kishida.cnn.util.RandomWriter;
import lombok.Getter;
import lombok.Setter;

/**
 *
 * @author naoki
 */
public class NeuralNetwork {
    @Getter
    private float learningRate;
    @Getter
    private float weightDecay;

    @JsonIgnore
    @Getter @Setter
    Random random;

    @JsonIgnore
    @Getter @Setter
    Random imageRandom;
    private byte[] imageRandomState;

    @Getter
    private int miniBatch;
    @Getter
    private float momentam;

    @Getter
    private List<NeuralLayer> layers;

    @Getter @Setter
    private int loop;

    @Getter @Setter
    private int imageIndex;


    public NeuralNetwork() {
        this(0.01f, 0.0005f, 128, 0.9f, 1234, 2345, new ArrayList<>());
    }

    public NeuralNetwork(float learningRate, float weightDecay, int miniBatch, float momentam,
            long randomSeed, long imageRandomSeed, List<NeuralLayer> layers){
        this(learningRate, weightDecay, miniBatch, momentam,
                null, randomSeed, null, imageRandomSeed, 0, 0, layers);
    }

    @JsonCreator
    public NeuralNetwork(
            @JsonProperty("learningRage") float learningRate,
            @JsonProperty("weightDecay") float weightDecay,
            @JsonProperty("miniBatch") int miniBatch,
            @JsonProperty("momentam") float momentam,
            @JsonProperty("random") byte[] randomState,
            @JsonProperty("randomSeed") long randomSeed,
            @JsonProperty("imageRandom") byte[] imageRandomState,
            @JsonProperty("imageRandomSeed") long imageRandomSeed,
            @JsonProperty("loop") int loop,
            @JsonProperty("imageIndex") int imageIndex,
            @JsonProperty("layers") List<NeuralLayer> layers) {
        this.learningRate = learningRate;
        this.weightDecay = weightDecay;
        this.miniBatch = miniBatch;
        this.momentam = momentam;
        this.layers = layers;
        this.imageIndex = imageIndex;
        this.loop = loop;
        if(randomState != null){
            random = RandomWriter.getRandomFromState(randomState);
        }else{
            random = new Random(randomSeed);
        }
        if(imageRandomState != null){
            imageRandom = RandomWriter.getRandomFromState(imageRandomState);
        }else{
            imageRandom = new Random(imageRandomSeed);
        }
    }

    public void init(){
        layers.forEach(layer -> layer.setParent(this));
        for(int i = 1; i < layers.size(); ++i){
            layers.get(i).setPreLayer(layers.get(i - 1));
        }
    }

    @JsonProperty("random")
    byte[] getRandomState(){
        return RandomWriter.getRandomState(random);
    }

    @JsonProperty("imageRandom")
    byte[] getImageRandomState(){
        if(imageRandomState == null){
            saveImageRandomState();
        }
        return imageRandomState;
    }
    public void saveImageRandomState(){
        imageRandomState = RandomWriter.getRandomState(imageRandom);
    }

    public void writeAsJson(Writer writer) throws IOException{
        ObjectMapper mapper = new ObjectMapper();
        mapper.enable(SerializationFeature.INDENT_OUTPUT);
        mapper.writeValue(writer, this);
    }

    public static NeuralNetwork readFromJson(Reader reader) throws IOException{
        ObjectMapper mapper = new ObjectMapper();
        return mapper.readValue(reader, NeuralNetwork.class);
    }

    public Optional<NeuralLayer> findLayerByName(String name){
        return layers.stream()
                .filter(layer -> name.equals(layer.getName()))
                .findFirst();
    }

    public float[] forward(float[] readData, float[] correctData){
        ((InputLayer)layers.get(0)).setInput(readData);
        for(int i = 1; i < layers.size(); ++i){
            layers.get(i).forward();
        }
        float[] output = layers.get(layers.size() - 1).getResult();
        if (!FloatUtil.toDoubleStream(output).allMatch(d -> Double.isFinite(d))) {
            throw new RuntimeException("there are some infinite value");
        }

        //誤差を求める
        float[] delta = new float[output.length];
        for(int idx = 0; idx < output.length; ++idx){
            delta[idx] = correctData[idx] - output[idx];
        }
        //逆伝播
        for(int i = layers.size() - 1; i >= 1; --i){
            delta = layers.get(i).backward(delta);
        }

        return output;
    }

    public void joinBatch(){
        layers.forEach(NeuralLayer::joinBatch);
    }
    public void prepareBatch() {
        layers.forEach(NeuralLayer::prepareBatch);
    }

    public static void main(String[] args) throws IOException {
        NeuralNetwork nn = new NeuralNetwork();
        nn.getLayers().addAll(Arrays.asList(
                new InputLayer(20, 20),
                new ConvolutionLayer("conv1", 3, 7, 2, 1, true),
                new MaxPoolingLayer("pool", 3, 2),
                new MultiNormalizeLayer("norm1", 5, .0001f, true),
                new FullyConnect("test", 3, 0, 1, new LogisticFunction(), true)));
        nn.init();
        nn.random.nextInt();
        StringWriter sw = new StringWriter();
        nn.writeAsJson(sw);
        System.out.println(sw);

        NeuralNetwork v = NeuralNetwork.readFromJson(new StringReader(
"{\n" +
"  \"weightDecay\" : 5.0E-4,\n" +
"  \"miniBatch\" : 128,\n" +
"  \"random\" : \"c3EAfgAAAT/wWGBKFyCXAAATnQ6sF654\",\n" +
"  \"imageRandom\" : \"c3EAfgAAAAAAAAAAAAAAAAAABd7s70R4\",\n" +
"  \"momentam\" : 0.9,\n" +
"  \"layers\" : [ {\n" +
"    \"InputLayer\" : {\n" +
"      \"width\" : 250,\n" +
"      \"height\" : 220,\n" +
"      \"name\" : \"input\"\n" +
"    }\n" +
"  }, {\n" +
"    \"ConvolutionLayer\" : {\n" +
"      \"name\" : \"conv1\",\n" +
"      \"filter\" : null,\n" +
"      \"bias\" : [ 1.0, 1.0, 1.0 ],\n" +
"      \"filterDelta\" : null,\n" +
"      \"biasDelta\" : [ 0.0, 0.0, 0.0 ],\n" +
"      \"stride\" : 2,\n" +
"      \"filterSize\" : 7,\n" +
"      \"useGpu\" : true\n" +
"    }\n" +
"  }, {\n" +
"    \"MaxPoolingLayer\" : {\n" +
"      \"name\" : \"pool\",\n" +
"      \"size\" : 3,\n" +
"      \"stride\" : 2\n" +
"    }\n" +
"  }, {\n" +
"    \"MultiNormalizeLayer\" : {\n" +
"      \"name\" : \"norm1\",\n" +
"      \"size\" : 5,\n" +
"      \"threshold\" : 1.0E-4,\n" +
"      \"useGpu\" : true\n" +
"    }\n" +
"  }, {\n" +
"    \"FullyConnect\" : {\n" +
"      \"name\" : \"test\",\n" +
"      \"outputSize\" : 3,\n" +
"      \"weight\" : [ 0.0014115907, 0.0043465886, 0.01138472, -0.0013297468, "
                + "-0.0060525155, -0.0109255025, -0.015493984, 0.011872963, -0.0015145391 ],\n" +
"      \"initBias\" : 0.5, " +
"      \"bias\" : [ 0.0, 0.2, 0.4 ],\n" +
"      \"weightDelta\" : [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],\n" +
"      \"biasDelta\" : [ 0.0, 0.0, 0.0 ],\n" +
"      \"dropoutRate\" : 1.0,\n" +
"      \"activation\" : \"LogisticFunction\",\n" +
"      \"useGpu\" : true\n" +
"    }\n" +
"  } ],\n" +
"  \"learningRate\" : 0.01\n" +
"}"));
        System.out.println(nn.random.nextInt());
        System.out.println(v.random.nextInt());
        v.findLayerByName("test").ifPresent(layer -> {
            FullyConnect f = (FullyConnect) layer;
            System.out.println(f.getActivation().getClass());
            System.out.println(Arrays.toString(f.getBias()));
        });
        v.init();
        v.findLayerByName("test").ifPresent(layer -> {
            FullyConnect f = (FullyConnect) layer;
            System.out.println(f.getActivation().getClass());
            System.out.println(Arrays.toString(f.getBias()));
        });
    }

}
