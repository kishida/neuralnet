/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.activation;

/** 正規化線形関数 */
public class RetifierdLinear extends ActivationFunction {

    @Override
    public float apply(float value) {
        return value >= 0 ? value : 0;
    }

    @Override
    public float diff(float value) {
        return value >= 0 ? 1 : 0;
    }

}
