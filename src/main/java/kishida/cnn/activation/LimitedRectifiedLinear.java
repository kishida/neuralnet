/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.activation;

/** 正規化線形関数 */
public class LimitedRectifiedLinear extends ActivationFunction {
    float limit;

    public LimitedRectifiedLinear(float limit) {
        this.limit = limit;
    }

    @Override
    public float apply(float value) {
        return Math.max(0, Math.min(value, 2));
    }

    @Override
    public float diff(float value) {
        return value >= 0 && value <= 2 ? 1 : 0;
    }

    @Override
    public String getName() {
        return "limitrelu";
    }

}
