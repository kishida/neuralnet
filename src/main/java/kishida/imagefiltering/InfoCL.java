/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.imagefiltering;

import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;

/**
 *
 * @author naoki
 */
public class InfoCL {
    public static void main(String[] args) {
        CLContext ctx = CLContext.create();
        CLDevice dev = ctx.getMaxFlopsDevice();
        System.out.println(dev);
        System.out.println(memSizeString(dev.getGlobalMemSize()));
        System.out.println(memSizeString(dev.getLocalMemSize()));
        System.out.println(dev.getMaxWorkGroupSize());
        ctx.release();
    }
    static String memSizeString(long memSize){
        double size = memSize;
        String[] unit = {"", "K", "M", "G", "T"};
        for(int i = 0; i < unit.length; ++i){
            if(size < 1024){
                return String.format("%.1f%sB", size, unit[i]);
            }
            size /= 1024;
        }
        return String.format("%.1f%sPB", size);
    }
}
