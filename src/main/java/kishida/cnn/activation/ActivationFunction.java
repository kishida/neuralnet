/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.activation;

import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;

/** 活性化関数 */
@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.WRAPPER_OBJECT)
@JsonSubTypes({
        @JsonSubTypes.Type(LogisticFunction.class),
        @JsonSubTypes.Type(RectifiedLinear.class),
        @JsonSubTypes.Type(SoftMaxFunction.class),
})
public abstract class ActivationFunction {

    public abstract float apply(float value);

    public void applyAfter(float[] values) {
        for(int i = 0; i < values.length; ++i){
            values[i] = apply(values[i]);
        }
    }

    /** 微分 */
    public abstract float diff(float value);

}
