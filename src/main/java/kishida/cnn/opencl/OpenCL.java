/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.cnn.opencl;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLMemory;
import com.jogamp.opencl.CLProgram;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.Arrays;
import lombok.Getter;

/**
 *
 * @author naoki
 */
public class OpenCL {

    static CLContext ctx;
    @Getter
    static CLCommandQueue queue;
    static CLDevice device;

    public static void prepare(){
        ctx = CLContext.create();
        device = ctx.getMaxFlopsDevice();
        System.out.println(device);
        queue = device.createCommandQueue();
    }

    public static void release(){
        queue.finish();
        ctx.release();
        ctx = null;
    }

    public static CLContext getCtx() {
        if(ctx == null){
            prepare();
        }
        return ctx;
    }

    public static CLProgram compile(String path){
        try {
            return getCtx().createProgram(OpenCL.class.getResourceAsStream("/kernels/" + path))
                    .build();
        } catch (IOException ex) {
            throw new UncheckedIOException(ex);
        }
    }

    public static CLBuffer<FloatBuffer> createReadBuffer(float[] data){
        CLBuffer<FloatBuffer> buf = getCtx().createFloatBuffer(
                data.length, CLMemory.Mem.READ_ONLY);
        buf.getBuffer().put(data).rewind();//rewindしないと不安定になる
        return buf;
    }
    public static CLBuffer<FloatBuffer> createReadWriteBuffer(float[]... data){
        CLBuffer<FloatBuffer> buf = createReadWriteBuffer(
                Arrays.stream(data).mapToInt(d -> d.length).sum());
        FloatBuffer fb = buf.getBuffer();
        Arrays.stream(data).forEach(fb::put);
        fb.rewind();//rewindしないと不安定になる
        return buf;
    }
    public static CLBuffer<FloatBuffer> createReadWriteBuffer(int size){
        CLBuffer<FloatBuffer> buf = getCtx().createFloatBuffer(
                size, CLMemory.Mem.READ_WRITE);
        return buf;

    }
    public static CLBuffer<FloatBuffer> createWriteBuffer(int size){
        return getCtx().createFloatBuffer(size, CLMemory.Mem.WRITE_ONLY);
    }
    public static CLBuffer<IntBuffer> createReadBuffer(int[] data){
        CLBuffer<IntBuffer> buf = getCtx().createIntBuffer(
                data.length, CLMemory.Mem.READ_ONLY);
        buf.getBuffer().put(data).rewind();
        return buf;
    }
    public static CLCommandQueue execute(CLKernel kernel, int range){
        int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 128);
        int globalWorkSize = roundUp(localWorkSize, range);
        kernel.putArg(range);
        return getQueue().put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize);
    }
    static  int roundUp(int groupSize, int globalSize){
        return ((globalSize + groupSize - 1) / groupSize) * groupSize;
    }
}
