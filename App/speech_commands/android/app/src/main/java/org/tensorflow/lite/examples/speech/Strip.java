package org.tensorflow.lite.examples.speech;

import com.jlibrosa.audio.JLibrosa;

public class Strip {
    public float[][][][] superimpose(background, overlay):
    overlayed = background + overlay
            s = jlibrosa.feature.melspectrogram(overlayed, 16000)
    s1 = librosa.power_to_db(s, ref=np.max)
    x = np.zeros((128, 33), dtype=float)
            for i in range(s1.shape[0]):
            for j in range(s1[i].size):
    x[i][j] = s1[i][j]

    res = x.reshape(128, 33, 1)
            return res
}
