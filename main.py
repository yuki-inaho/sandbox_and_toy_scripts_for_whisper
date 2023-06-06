import time
import whisper
from pathlib import Path
from typing import Iterator, TextIO

def write_txt(transcript: Iterator[dict], file: TextIO):
    for segment in transcript:
        print(segment["text"].strip(), file=file, flush=True)


""" Set the audio file path and weight path
"""
audio_dir_pathlib = Path("./data")
audio_name = Path("230607_0014.mp3")
audio_path = str(audio_dir_pathlib.joinpath(audio_name))

weights_name = "large-v2.pt"

output_txt_path = Path(audio_name).stem + ".txt"

""" Loading the model
"""
start = time.time()
model = whisper.load_model("large-v2", device="cpu")
model.half()
model.cuda()
for m in model.modules():
    if isinstance(m, whisper.model.LayerNorm):
        m.float()
end = time.time()
print(f"Loading the whisper model is done. Elapsed: {end-start}")

""" Run transcription (with GPU)
"""
start = time.time()
result = model.transcribe(audio_path, verbose=True, language="japanese", beam_size=5, fp16=True, without_timestamps=False)
end = time.time()
print(f"Transcription is done, Elapsed: {end-start}")

""" Dump the text file
"""
with open(output_txt_path, "w", encoding="utf-8") as txt:
    write_txt(result["segments"], file=txt)
