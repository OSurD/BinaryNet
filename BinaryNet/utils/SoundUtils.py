import torchaudio
import torch

EXPECTED_SR = 16000 
TARGET_LEN = 16000

def pad_or_trim_center(waveform, target_len=TARGET_LEN, pad_value=0.0):
    """
    Центрированное выравнивание длины аудиосигнала по последней оси.
    """
    L = waveform.shape[-1]
    if L == target_len:
        return waveform

    if L < target_len:
        total = target_len - L
        left = total // 2
        right = total - left
        return torch.nn.functional.pad(waveform, (left, right), mode="constant", value=pad_value)

    # L > target_len
    start = (L - target_len) // 2
    end = start + target_len
    return waveform[..., start:end]

N_FFT = 512
N_MELS = 128

mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=EXPECTED_SR, n_fft=N_FFT, n_mels=N_MELS, center=True, power=2.0
)

ampl2db = torchaudio.transforms.AmplitudeToDB(stype="power")

def prepareMelSpectrogram(waveform):
    return ampl2db(mel(waveform))