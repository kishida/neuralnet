/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.activation;

import java.util.Arrays;

/** 活性化関数 */
public abstract class ActivationFunction {

    public abstract double apply(double value);

    public double[] applyAfter(double[] values) {
        return Arrays.stream(values).map(this::apply).toArray();
    }

    /** 微分 */
    public abstract double diff(double value);

}
