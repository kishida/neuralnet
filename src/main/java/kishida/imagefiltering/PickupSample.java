/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kishida.imagefiltering;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

/**
 *
 * @author naoki
 */
public class PickupSample {
    public static void main(String[] args) throws IOException {
        String from = "C:\\Users\\naoki\\Downloads\\256_ObjectCategories\\256_ObjectCategories";
        String to = "C:\\Users\\naoki\\Desktop\\256_ObjectCategories80";
        Random r = new Random(1234);
        Path fromPath = Paths.get(from);
        Files.list(fromPath).filter(Files::isDirectory).forEach(dir -> {
            try {
                List<Path> files = Files.list(dir).collect(Collectors.toList());
                Collections.shuffle(files, r);
                for(int i = 0; i < files.size(); ++i){
                    Path f = files.get(i);
                    Path toFile = Paths.get(to, i < 70 ? "learn" : "test")
                            .resolve(fromPath.relativize(f));
                    Files.createDirectories(toFile.getParent());
                    Files.copy(f, toFile);
                }
            } catch (IOException ex) {
                throw new UncheckedIOException(ex);
            }
        });
    }
}
