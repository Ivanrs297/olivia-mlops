import os
import subprocess
from typing import Optional, List, Dict, Any
import time
import psutil
import GPUtil
import matplotlib.pyplot as plt
# import whisper
from faster_whisper import WhisperModel
import whisperx
from whisperx import load_audio, load_align_model, align
from whisperx.diarize import DiarizationPipeline, assign_word_speakers
from pathlib import Path
import torch

def transcribe(audio_file: str, model_name: str, device: str = "cuda") -> Dict[str, Any]:
    """
    Transcribe an audio file using a speech-to-text model.

    Args:
        audio_file: Path to the audio file to transcribe.
        model_name: Name of the model to use for transcription.
        device: The device to use for inference (e.g., "cpu" or "cuda").

    Returns:
        A dictionary representing the transcript, including the segments, the language code, and the duration of the audio file.
    """
    model = WhisperModel(model_name,
                         device=device,
                         compute_type="float16",
                         cpu_threads=4)
    segments, _ = model.transcribe(audio_file, beam_size=5)
    audio_id = Path(audio_file).stem
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
    
    return {
        "segments": entries,
        "language_code": "es",
    }

def align_segments(
    segments: List[Dict[str, Any]],
    language_code: str,
    audio_file: str,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Align the transcript segments using a pretrained alignment model.

    Args:
        segments: List of transcript segments to align.
        language_code: Language code of the audio file.
        audio_file: Path to the audio file containing the audio data.
        device: The device to use for inference (e.g., "cpu" or "cuda").

    Returns:
        A dictionary representing the aligned transcript segments.
    """
    model_a, metadata = load_align_model(language_code=language_code, device=device)
    result_aligned = align(segments, model_a, metadata, audio_file, device)
    return result_aligned


def diarize(audio_file, hf_token: str) -> Dict[str, Any]:
    """
    Perform speaker diarization on an audio file.

    Args:
        audio_file: Path to the audio file to diarize.
        hf_token: Authentication token for accessing the Hugging Face API.

    Returns:
        A dictionary representing the diarized audio file, including the speaker embeddings and the number of speakers.
    """
    diarization_pipeline = DiarizationPipeline(use_auth_token=hf_token, device="cuda")
    diarization_result = diarization_pipeline(audio_file, min_speakers=2, max_speakers=3)
    return diarization_result


def assign_speakers(
    diarization_result: Dict[str, Any], aligned_segments: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Assign speakers to each transcript segment based on the speaker diarization result.

    Args:
        diarization_result: Dictionary representing the diarized audio file, including the speaker embeddings and the number of speakers.
        aligned_segments: Dictionary representing the aligned transcript segments.

    Returns:
        A list of dictionaries representing each segment of the transcript, including the start and end times, the
        spoken text, and the speaker ID.
    """
    result_segments = assign_word_speakers(
        diarization_result, aligned_segments
    )
    result_segments = result_segments["segments"]
    
    # results_segments_w_speakers: List[Dict[str, Any]] = []
    # for result_segment in result_segments:
    #     if "speaker" in result_segment.keys():
    #         results_segments_w_speakers.append(
    #             {
    #                 "start": result_segment["start"],
    #                 "end": result_segment["end"],
    #                 "text": result_segment["text"],
    #                 "speaker": result_segment["speaker"],
    #             }
    #         )
    # return results_segments_w_speakers
    return result_segments


def transcribe_and_diarize(
    audio_file: str,
    hf_token: str,
    model_name: str,
    device: str = "cuda",
) -> List[Dict[str, Any]]:
    """
    Transcribe an audio file and perform speaker diarization to determine which words were spoken by each speaker.

    Args:
        audio_file: Path to the audio file to transcribe and diarize.
        hf_token: Authentication token for accessing the Hugging Face API.
        model_name: Name of the model to use for transcription.
        device: The device to use for inference (e.g., "cpu" or "cuda").

    Returns:
        A list of dictionaries representing each segment of the transcript, including the start and end times, the
        spoken text, and the speaker ID.
    """
    audio_data = load_audio(audio_file)
    transcript = transcribe(audio_file, model_name, device)
    print("transcript made.")
    aligned_segments = align_segments(
        transcript["segments"], transcript["language_code"], audio_file, device
    )
    print("alignment made.")
    diarization_result = diarize(audio_data, hf_token)
    print("diarization made.")

    print(diarization_result)
    # return
    
    results_segments_w_speakers = assign_speakers(diarization_result, aligned_segments)

    return

    # Print the results in a user-friendly way
    for i, segment in enumerate(results_segments_w_speakers):
        print(f"Segment {i + 1}:")
        print(f"Start time: {segment['start']:.2f}")
        print(f"End time: {segment['end']:.2f}")
        print(f"Speaker: {segment['speaker']}")
        print(f"Transcript: {segment['text']}")
        print("")

    return results_segments_w_speakers


transcribe_and_diarize(audio_file="audio_samples/F-285.wav", hf_token='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', model_name="whisper-small-olivia-001-ct2", device="cuda")