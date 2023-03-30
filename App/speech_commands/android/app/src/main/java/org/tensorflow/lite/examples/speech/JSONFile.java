package org.tensorflow.lite.examples.speech;

import java.io.Serializable;
import java.util.List;

public class JSONFile implements Serializable {

    private List<float[][]> data;

    public List<float[][]> getMfccs() {
        return data;
    }

    //getters and setters
}
