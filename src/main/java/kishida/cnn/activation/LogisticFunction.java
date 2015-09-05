/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.activation;

/** ロジスティックシグモイド関数 */
public class LogisticFunction extends ActivationFunction {

    @Override
    public float apply(float value) {
        return 1 / (1 + (float)Math.exp(-value));
    }

    @Override
    public float diff(float value) {
        return value * (1 - value);
    }

}
