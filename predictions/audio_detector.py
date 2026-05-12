import wave
import subprocess
import os
import numpy as np


class AudioSirenDetector:
    """FFT-based emergency siren detector.

    Emergency sirens operate mostly in 700–2000 Hz.
    get_confidence() returns the fraction of audio energy in that band
    at the timestamp corresponding to the given video frame.
    """

    SIREN_LOW_HZ  = 700
    SIREN_HIGH_HZ = 2000

    def extract_audio(self, video_path, out_wav="temp_audio.wav"):
        """Extract mono 22050 Hz WAV from a video file using ffmpeg.

        Returns path to WAV file, or None if ffmpeg is unavailable.
        """
        try:
            subprocess.run(
                [
                    "ffmpeg", "-i", video_path,
                    "-vn", "-ar", "22050", "-ac", "1",
                    "-f", "wav", out_wav, "-y", "-loglevel", "quiet",
                ],
                check=True,
                capture_output=True,
            )
            return out_wav
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def _read_wav(self, wav_path):
        with wave.open(wav_path, "rb") as wf:
            sr = wf.getframerate()
            nc = wf.getnchannels()
            raw = wf.readframes(wf.getnframes())
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        if nc > 1:
            data = data[::nc]  # keep first channel only
        return sr, data

    def get_confidence(self, wav_path, frame_number, fps, window_sec=1.0):
        """Return fraction of energy in siren frequency band for this frame's timestamp.

        Returns 0.0 on any error so callers don't need try/except.
        """
        try:
            sr, data = self._read_wav(wav_path)
            t = frame_number / max(fps, 1)
            s = int(max(0.0, t - window_sec / 2) * sr)
            e = int((t + window_sec / 2) * sr)
            chunk = data[s:e]
            if len(chunk) < 512:
                return 0.0
            mag   = np.abs(np.fft.rfft(chunk))
            freqs = np.fft.rfftfreq(len(chunk), 1.0 / sr)
            mask  = (freqs >= self.SIREN_LOW_HZ) & (freqs <= self.SIREN_HIGH_HZ)
            total = np.sum(mag)
            return float(np.sum(mag[mask]) / total) if total > 0 else 0.0
        except Exception:
            return 0.0
