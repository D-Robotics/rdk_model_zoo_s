import soundfile as sf
from scipy.io import wavfile
import scipy.signal as sps
import numpy as np

import libmodel_task


with open("vocab.json", "r", encoding="utf-8-sig") as f:
    d = eval(f.read())
res = dict((v, k) for k, v in d.items())
res[69] = "[PAD]"
res[68] = "[UNK]"

inf = libmodel_task.ModelTask()
inf.ModelInit("asr.hbm")


def reduce_channel(input_file, output_file):
    sample_rate, data = wavfile.read(input_file)

    if len(data.shape) == 2 and data.shape[1] != 1:
        mono_data = np.mean(data, axis=1).astype(data.dtype)
        wavfile.write(output_file, sample_rate, mono_data)
    else:
        print("Input file is not 8-channel.")
        

def _normalize(x):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return np.squeeze((x - mean) / np.sqrt(var + 1e-5))


def remove_adjacent(item):
    nums = list(item)
    a = nums[:1]
    for item in nums[1:]:
        if item != a[-1]:
            a.append(item)
    return "".join(a)


AUDIO_MAXLEN = 30000
new_rate = 16000
audio_file_path = "chi_sound.wav"
reduce_channel(audio_file_path, audio_file_path)
sampling_rate, data = wavfile.read(audio_file_path)
samples = round(len(data) * float(new_rate) / sampling_rate)
new_data = sps.resample(data, samples)
speech = np.array(new_data, dtype=np.float32)
speech = _normalize(speech)[None]
if speech.shape[1] < AUDIO_MAXLEN:
    padding = np.zeros((speech.shape[0], AUDIO_MAXLEN - speech.shape[1]))
    speech = np.concatenate([speech, padding], axis=-1).astype(np.float32)
else:
    speech = speech[:,:AUDIO_MAXLEN]

input_data = [speech]
output_arr = np.array(inf.ModelInfer(input_data)[0]).reshape(1,93,3503)
prediction = np.argmax(output_arr, axis=-1)
_t1 = "".join([res[i] for i in list(prediction[0])])
cleaned_text = _t1.replace("<pad>", "")
print(cleaned_text)