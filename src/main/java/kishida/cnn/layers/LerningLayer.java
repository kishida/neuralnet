/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.layers;

import java.util.DoubleSummaryStatistics;

/**
 *
 * @author naoki
 */
public interface LerningLayer {
    DoubleSummaryStatistics getWeightStatistics();
    DoubleSummaryStatistics getBiasStatistics();
}
