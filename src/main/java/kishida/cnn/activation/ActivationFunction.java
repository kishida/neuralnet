/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.activation;

/** 活性化関数 */
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
