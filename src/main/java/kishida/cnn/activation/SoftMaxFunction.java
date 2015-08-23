/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.activation;

import java.util.Arrays;

/** ソフトマックス */
public class SoftMaxFunction extends ActivationFunction {

    @Override
    public double apply(double value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public double[] applyAfter(double[] values) {
        double total = Arrays.stream(values).parallel()
                .map((double d) -> Math.exp(d)).sum();
        return Arrays.stream(values).parallel()
                .map((double d) -> Math.exp(d) / total).toArray();
    }

    @Override
    public double diff(double value) {
        return value * (1 - value);
    }

}
