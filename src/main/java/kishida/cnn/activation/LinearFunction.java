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
    public double apply(double value) {
        return value;
    }

    @Override
    public double diff(double value) {
        return 1;
    }

}
