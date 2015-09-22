/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.layers;

import com.jogamp.opencl.CLBuffer;
import java.nio.FloatBuffer;

/**
 *
 * @author naoki
 */
public interface FullGpuEnabled {
    default boolean isUseGpu(){
        return true;
    }
    CLBuffer<FloatBuffer> getBufResult();
    void forward(CLBuffer<FloatBuffer> input);
}
