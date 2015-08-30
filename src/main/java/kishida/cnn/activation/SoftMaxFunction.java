/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.activation;

import java.util.Arrays;
import java.util.stream.IntStream;

/** ソフトマックス */
public class SoftMaxFunction extends ActivationFunction {

    @Override
    public double apply(double value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void applyAfter(double[] values) {
        double total = Arrays.stream(values).parallel()
                .map((double d) -> Math.exp(Math.min(700, d))).sum();
        IntStream.range(0, values.length).parallel().forEach(i -> {
            values[i] = Math.exp(Math.min(700, values[i])) / total;
        });
    }

    @Override
    public double diff(double value) {
        return value * (1 - value);
    }

}
