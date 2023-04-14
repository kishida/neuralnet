/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.activation;

/**
 *
 * @author naoki
 */
public class LinearFunction extends ActivationFunction {

    @Override
    public float apply(float value) {
        return value;
    }

    @Override
    public float diff(float value) {
        return 1;
    }
    @Override
    public String getName() {
        return "linear";
    }

}
