import time
import whisper
from pathlib import Path

""" Set the audio file path and weight path
"""
audio_dir_pathlib = Path("./data")
audio_name = Path("audio.mp3")
audio_path = str(audio_dir_pathlib.joinpath(audio_name))

weights_dir_pathlib = Path("./whisper")
weights_name = "large-v2.pt"

""" Loading the model
"""
start = time.time()
model = whisper.load_model(str(weights_dir_pathlib.joinpath(weights_name)), device="cpu")
model.half()
model.cuda()
for m in model.modules():
    if isinstance(m, whisper.model.LayerNorm):
        m.float()
end = time.time()
print(f"Loading the whisper model is done. Elapsed: {end-start}")

""" Run transcription (with GPU)
"""
result = model.transcribe(audio_path, verbose=True, language="japanese", beam_size=5, fp16=True, without_timestamps=True)
print(f"Transcription is done, Elapsed: {end-start}")
print(result["text"])
