/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.activation;

import java.util.stream.IntStream;

/** ソフトマックス */
public class SoftMaxFunction extends ActivationFunction {

    @Override
    public float apply(float value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void applyAfter(float[] values) {
        float total = (float)IntStream.range(0, values.length).parallel()
                .mapToDouble(i -> values[i])
                .map(d -> Math.exp(Math.min(700, d))).sum();
        IntStream.range(0, values.length).parallel().forEach(i -> {
            values[i] = (float)Math.exp(Math.min(700, values[i])) / total;
        });
    }

    @Override
    public float diff(float value) {
        return value * (1 - value);
    }

    @Override
    public String getName() {
        return "softmax";
    }

}
