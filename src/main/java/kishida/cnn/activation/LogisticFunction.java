/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.activation;

/** ロジスティックシグモイド関数 */
public class LogisticFunction extends ActivationFunction {

    @Override
    public double apply(double value) {
        return 1 / (1 + Math.exp(-value));
    }

    @Override
    public double diff(double value) {
        return value * (1 - value);
    }

}
