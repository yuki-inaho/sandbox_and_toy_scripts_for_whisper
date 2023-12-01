import time
import whisper
import argparse
from pathlib import Path
from typing import Iterator, TextIO


def write_txt(transcript: Iterator[dict], file: TextIO):
    for segment in transcript:
        print(segment["text"].strip(), file=file, flush=True)


def main(args):
    """Set the audio file path and weight path"""
    audio_path = args.audio_file_path
    assert Path(audio_path).exists(), f"Audio file not found: {audio_path}"

    # Load models
    audio_name = Path(audio_path).name
    output_txt_path = Path(audio_name).stem + ".txt"

    """ Loading the model """
    start = time.time()
    model = whisper.load_model(args.model_size, device="cpu")
    if args.device == "cuda":
        if args.fp16:
            model.half()
            model.cuda()
            for m in model.modules():
                if isinstance(m, whisper.model.LayerNorm):
                    m.float()
        else:
            model.cuda()
    end = time.time()
    print(f"Loading the whisper model is done. Elapsed: {end-start}")

    """ Run transcription (with GPU) """
    print(f"Start transcription [with timestamps] flag: {args.with_timestamps}")
    start = time.time()
    result = model.transcribe(
        audio_path,
        temperature=args.temperature,
        verbose=args.verbose,
        language=args.language,
        beam_size=args.beam_size,
        fp16=args.fp16,
        word_timestamps=args.with_timestamps,
    )
    end = time.time()
    print(f"Transcription is done, Elapsed: {end-start}")

    """ Dump the text file """
    with open(output_txt_path, "w", encoding="utf-8") as txt:
        write_txt(result["segments"], file=txt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio files using Whisper")
    parser.add_argument(
        "-a",
        "--audio-file-path",
        type=str,
        default="data",
        help="Path to the audio file",
    )
    parser.add_argument(
        "--model-size", type=str, default="large-v3", help="Model size to use"
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda",
        help="Device to use for model (cpu or cuda)",
    )
    parser.add_argument(
        "--language", type=str, default="Japanese", help="Language of the audio"
    )
    parser.add_argument(
        "--beam-size", type=int, default=5, help="Beam size for the model"
    )
    parser.add_argument(
        "-t", "--temperature", type=float, default=0.0, help="Temperature for the model"
    )
    parser.add_argument("--fp16", action="store_true", help="Use half precision")
    parser.add_argument(
        "-wit",
        "--with-timestamps",
        action="store_true",
        help="Do not include timestamps in transcription",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose mode"
    )

    args = parser.parse_args()
    main(args)
