import numpy as np
from hbm_runtime import HB_HBMRuntime
inf = HB_HBMRuntime("kws.hbm")
import time
import paddle
import paddleaudio
from paddleaudio.compliance.kaldi import fbank


THRES = 60000


def audio_trunc(audio_arr, thres=THRES):
    length = audio_arr.shape[1]
    if length > thres:
        audio_arr = audio_arr[:, :thres]
        return audio_arr
    elif length < thres:
        pad_zero = paddle.zeros((1,THRES), dtype=audio_arr.dtype)
        pad_zero[:, :length] = audio_arr
        return pad_zero
    
feat_func = lambda waveform, sr: fbank(
    waveform=paddle.to_tensor(waveform), 
    sr=sr, 
    frame_shift=10, 
    frame_length=25, 
    n_mels=80)

key_test_load = paddleaudio.load('sample.wav')
key_test_load = (audio_trunc(key_test_load[0]), key_test_load[1])
keyword_feat = feat_func(*key_test_load)
key_input = keyword_feat.unsqueeze(0).numpy()

out = inf.run(key_input)['kws']['sigmoid_1.tmp_0']
keyword_score = np.max(out).item()
print(keyword_score)
