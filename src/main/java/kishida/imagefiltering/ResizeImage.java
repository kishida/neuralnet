/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.imagefiltering;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import javax.imageio.ImageIO;

/**
 *
 * @author naoki
 */
public class ResizeImage {
    public static void main(String[] args) throws Exception{
        Path path = Paths.get("C:\\Users\\naoki\\Desktop\\sampleimg");
        Path to = path.resolveSibling("sampleimg" + (256 + 32));
        Files.walk(path)
                .filter(p -> !Files.isDirectory(p))
                .forEach(p -> {
                    try {
                        BufferedImage img = ImageIO.read(p.toFile());
                        BufferedImage resized = ConvolutionalNet.resize(img, 256 + 32, 256 + 32);
                        Path rel = path.relativize(p);
                        Path output = to.resolve(rel);
                        Files.createDirectories(output.getParent());
                        ImageIO.write(resized, "jpg", output.toFile());
                    } catch (IOException ex) {
                        throw new UncheckedIOException(ex);
                    }
                });
                
    }
}
