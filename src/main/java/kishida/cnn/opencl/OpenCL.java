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
import com.jogamp.opencl.CLMemory;
import com.jogamp.opencl.CLProgram;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.FloatBuffer;
import lombok.Getter;

/**
 *
 * @author naoki
 */
public class OpenCL {

    static CLContext ctx;
    @Getter
    static CLCommandQueue queue;

    public static void prepare(){
        ctx = CLContext.create();
        CLDevice device = ctx.getMaxFlopsDevice();
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
        buf.getBuffer().put(data);
        return buf;
    }
    public static CLBuffer<FloatBuffer> createReadWriteBuffer(float[] data){
        CLBuffer<FloatBuffer> buf = createReadWriteBuffer(data.length);
        buf.getBuffer().put(data);
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
}
