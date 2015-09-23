/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.layers;

import com.jogamp.opencl.CLBuffer;
import java.nio.FloatBuffer;
import kishida.cnn.opencl.OpenCL;

/**
 *
 * @author naoki
 */
public interface FullGpuEnabled {
    default boolean isUseGpu(){
        return true;
    }
    CLBuffer<FloatBuffer> getBufResult();
    void forward(CLBuffer<FloatBuffer> bufInput);
    CLBuffer<FloatBuffer> backwardBuf(CLBuffer<FloatBuffer> bufInput, CLBuffer<FloatBuffer> bufDelta);
    default CLBuffer<FloatBuffer> backwardBuf(CLBuffer<FloatBuffer> bufInput, float[] delta){
        CLBuffer<FloatBuffer> bufDelta = OpenCL.createReadBuffer(delta);
        OpenCL.getQueue().putWriteBuffer(bufDelta, false);
        CLBuffer<FloatBuffer> result = backwardBuf(bufInput, bufDelta);
        bufDelta.release();
        return result;
    }

}
