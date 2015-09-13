/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.layers;

import com.fasterxml.jackson.annotation.JsonIgnore;
import java.util.DoubleSummaryStatistics;

/**
 *
 * @author naoki
 */
public interface LerningLayer {
    @JsonIgnore
    DoubleSummaryStatistics getWeightStatistics();
    @JsonIgnore
    DoubleSummaryStatistics getBiasStatistics();
}
