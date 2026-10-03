// A persistent UTF-8/base64 bridge. Requires Java 11+ source-file launching.
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import vn.pipeline.*;

class VnCoreNLPBridge {
    public static void main(String[] args) throws Exception {
        PrintStream protocol = System.out;
        // Library diagnostics must never become protocol responses.
        System.setOut(System.err);
        VnCoreNLP pipeline = new VnCoreNLP(new String[]{"wseg"});
        BufferedReader input = new BufferedReader(
            new InputStreamReader(System.in, StandardCharsets.UTF_8));
        protocol.println("READY");
        String line;
        while ((line = input.readLine()) != null) {
            String text = new String(Base64.getDecoder().decode(line), StandardCharsets.UTF_8);
            Annotation annotation = new Annotation(text);
            pipeline.annotate(annotation);
            List<String> sentences = new ArrayList<>();
            for (Sentence sentence : annotation.getSentences()) {
                sentences.add(sentence.getWordSegmentedSentence());
            }
            String segmented = String.join(" ", sentences);
            protocol.println(Base64.getEncoder().encodeToString(
                segmented.getBytes(StandardCharsets.UTF_8)));
            protocol.flush();
        }
    }
}
