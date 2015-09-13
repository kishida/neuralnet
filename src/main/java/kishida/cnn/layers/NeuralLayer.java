/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.layers;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import java.util.DoubleSummaryStatistics;
import java.util.Objects;
import kishida.cnn.NeuralNetwork;
import kishida.cnn.util.FloatUtil;
import lombok.Getter;
import lombok.Setter;

/**
 *
 * @author naoki
 */
@JsonTypeInfo(use=JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.WRAPPER_OBJECT)
@JsonSubTypes({
    @JsonSubTypes.Type(FullyConnect.class),
    @JsonSubTypes.Type(ConvolutionLayer.class),
    @JsonSubTypes.Type(MultiNormalizeLayer.class),
    @JsonSubTypes.Type(MaxPoolingLayer.class),
    @JsonSubTypes.Type(InputLayer.class),

})
public abstract class NeuralLayer {
    @Getter
    String name;

    @JsonIgnore
    @Getter
    float[] result;

    @Setter
    NeuralLayer preLayer;

    @Setter
    NeuralNetwork parent;

    public NeuralLayer(String name) {
        this.name = name;
    }

    public float[] forward() {
        Objects.requireNonNull(preLayer, "preLayer is null on " + name);
        return forward(preLayer.result);
    }

    public float[] backward(float[] delta) {
        return backward(preLayer.result, delta);
    }

    public abstract float[] forward(float[] in);

    public abstract float[] backward(float[] in, float[] delta);

    public void prepareBatch(){
        // do nothing as default
    }
    public void joinBatch(){
        // do nothing as default
    }

    @JsonIgnore
    public abstract int getOutputSize();

    @JsonIgnore
    public DoubleSummaryStatistics getResultStatistics(){
        return FloatUtil.summary(result);
    }

}
