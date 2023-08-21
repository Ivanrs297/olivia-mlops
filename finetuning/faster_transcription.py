'''
Transcribe segments of audio files with faster-whisper
'''

import argparse
import json
from pathlib import Path
from faster_whisper import WhisperModel


def transcribe(audiopath, device, compute_type, cpu_threads, model_size):
    ''' Transcribe audiofile into segments '''
    model = WhisperModel(model_size,
                         device=device,
                         compute_type=compute_type,
                         cpu_threads=cpu_threads)
    segments, _ = model.transcribe(audiopath, beam_size=5)
    audio_id = Path(audiopath).stem
    print(f"Transcribing {audio_id}...")
    entries = []
    corrected_seek = 0
    last_seek = 0
    for segment in segments:
        if segment.seek != last_seek:
            corrected_seek = last_seek
        # print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        entries.append({
            "id": segment.id,
            "seek": corrected_seek,
            "start": segment.start,
            "end": segment.end,
            "tokens": segment.tokens,
            "text": segment.text,
            "temperature": segment.temperature,
            "avg_logprob": segment.avg_logprob,
            "compression_ratio": segment.compression_ratio,
            "no_speech_prob": segment.no_speech_prob
        })
        last_seek = segment.seek
    with open("faster_transcriptions/" + audio_id + ".json", 'w', encoding="utf-8") as json_file:
        json.dump({"segments": entries}, json_file, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("audio", type=str,
                        help="audio file(s) to transcribe")
    parser.add_argument("--device", type=str,
                        help="device", default="cuda")
    parser.add_argument("--compute_type", type=str,
                        help="compute_type", default="float16")
    parser.add_argument("--cpu_threads", type=int,
                        help="cpu_threads", default=4)
    parser.add_argument("--model_size", type=str,
                        help="model_size", default="whisper-small-olivia-001-ct2")

    args = parser.parse_args().__dict__
    transcribe(
        args.pop("audio"),
        args.pop("device"),
        args.pop("compute_type"),
        args.pop("cpu_threads"),
        args.pop("model_size"))
