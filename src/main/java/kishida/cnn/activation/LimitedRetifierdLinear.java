/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.activation;

/** 正規化線形関数 */
public class LimitedRetifierdLinear extends ActivationFunction {
    double limit;

    public LimitedRetifierdLinear(double limit) {
        this.limit = limit;
    }

    @Override
    public double apply(double value) {
        return Math.max(0, Math.min(value, 2));
    }

    @Override
    public double diff(double value) {
        return value >= 0 && value <= 2 ? 1 : 0;
    }

}
