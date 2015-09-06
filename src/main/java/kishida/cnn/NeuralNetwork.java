/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn;

import com.fasterxml.jackson.annotation.JsonCreator;
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
import java.util.Random;
import kishida.cnn.activation.LogisticFunction;
import kishida.cnn.layers.ConvolutionLayer;
import kishida.cnn.layers.FullyConnect;
import kishida.cnn.layers.InputLayer;
import kishida.cnn.layers.MaxPoolingLayer;
import kishida.cnn.layers.MultiNormalizeLayer;
import kishida.cnn.layers.NeuralLayer;
import lombok.Getter;

/**
 *
 * @author naoki
 */
public class NeuralNetwork {
    @Getter
    private float learningRate;
    @Getter
    private float weightDecay;

    private Random random = new Random(1234);
    @Getter
    private int miniBatch;
    @Getter
    private float momentam;

    @Getter
    private List<NeuralLayer> layers;

    public NeuralNetwork() {
        this(0.01f, 0.0005f, 128, 0.9f, new ArrayList<>());
    }

    @JsonCreator
    public NeuralNetwork(
            @JsonProperty("learningRage") float learningRate,
            @JsonProperty("weightDecay") float weightDecay,
            @JsonProperty("miniBatch") int miniBatch,
            @JsonProperty("momentam") float momentam,
            @JsonProperty("layers") List<NeuralLayer> layers) {
        this.learningRate = learningRate;
        this.weightDecay = weightDecay;
        this.miniBatch = miniBatch;
        this.momentam = momentam;
        this.layers = layers;
    }

    public void init(){
        for(int i = 1; i < layers.size(); ++i){
            layers.get(i).setPreLayer(layers.get(i - 1));
        }
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
    public static void main(String[] args) throws IOException {
        NeuralNetwork nn = new NeuralNetwork();
        nn.getLayers().addAll(Arrays.asList(
                new InputLayer(250, 220),
                new ConvolutionLayer("conv1", 3, 7, 2, 1, .2f, true),
                new MaxPoolingLayer("pool", 3, 2),
                new MultiNormalizeLayer("norm1", 5, .0001f, true),
                new FullyConnect("test", 3, 0, 1, new LogisticFunction(), .001f, true)));
        nn.init();
        StringWriter sw = new StringWriter();
        nn.writeAsJson(sw);
        System.out.println(sw);

        NeuralNetwork v = NeuralNetwork.readFromJson(new StringReader(
"{\n" +
"  \"weightDecay\" : 5.0E-4,\n" +
"  \"miniBatch\" : 128,\n" +
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
"      \"bias\" : [ 0.0, 0.0, 0.0 ],\n" +
"      \"weightDelta\" : [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],\n" +
"      \"biasDelta\" : [ 0.0, 0.0, 0.0 ],\n" +
"      \"dropoutRate\" : 1.0,\n" +
"      \"activation\" : {\n" +
"        \"LogisticFunction\" : { }\n" +
"      },\n" +
"      \"useGpu\" : true\n" +
"    }\n" +
"  } ],\n" +
"  \"learningRate\" : 0.01\n" +
"}"));
        System.out.println(v.getMiniBatch());
    }


}
