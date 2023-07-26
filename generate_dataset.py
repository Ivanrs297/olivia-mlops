import json
import numpy as np
import random
from pydub import AudioSegment
from datasets import Dataset, DatasetDict
from datetime import timedelta
import os

current_directory = os.getcwd()

random.seed(0)

with open("main.custom_segments.json", "r", encoding="utf-8") as jsonfile:
    data = json.load(jsonfile)

audios = np.unique([item["source"] for item in data])

count_dict = {}

count = 0
for audio in audios:
    audio_count = len([item for item in data if item["source"] == audio])
    print(audio, audio_count)
    count += audio_count
    count_dict[audio] = audio_count

print("Total", count)


test_count = int(count * 0.15)
val_count = int(count * 0.15)
train_count = count - (test_count + val_count)
print("Estimated Distribution: ", test_count, val_count, train_count)

sorted_audios = sorted(count_dict.keys(), key=lambda x: count_dict[x], reverse=True)


test_audios = []
curr = sorted_audios.pop()
test_recount = sum([count_dict[item] for item in test_audios])
while (test_recount + count_dict[curr]) < test_count:
    test_audios.append(curr)
    test_recount = sum([count_dict[item] for item in test_audios])
    curr = sorted_audios.pop()

val_audios = []
val_recount = sum([count_dict[item] for item in val_audios])
while (val_recount + count_dict[curr]) < val_count:
    val_audios.append(curr)
    val_recount = sum([count_dict[item] for item in val_audios])
    curr = sorted_audios.pop()

sorted_audios.append(curr)
train_audios = sorted_audios

print("Test basket", sum([count_dict[item] for item in test_audios]))
print("Val basket", sum([count_dict[item] for item in val_audios]))
print("Train basket", sum([count_dict[item] for item in train_audios]))


test_list = [
    segment
    for segment in data
    if segment["source"] in test_audios and segment["end"] > segment["start"]
]
val_list = [
    segment
    for segment in data
    if segment["source"] in val_audios and segment["end"] > segment["start"]
]
train_list = [
    segment
    for segment in data
    if segment["source"] in train_audios and segment["end"] > segment["start"]
]


def convert_seconds_to_time_format(seconds):
    # Create a timedelta object from the number of seconds
    time_delta = timedelta(seconds=seconds)

    # Use the timedelta to calculate hours, minutes, seconds, and milliseconds
    hours = time_delta.seconds // 3600
    remaining_seconds = time_delta.seconds % 3600
    minutes = remaining_seconds // 60
    seconds = remaining_seconds % 60
    milliseconds = time_delta.microseconds // 1000

    # Format the time as a string
    time_format = "{:02d}:{:02d}:{:02d}:{:03d}".format(
        hours, minutes, seconds, milliseconds
    )

    return time_format


def file_exists(file_path):
    return os.path.exists(file_path)


def trim_audio(input_path, output_path, start_ms, end_ms):
    """
    Trim an audio file from start_ms to end_ms and save the result to output_path.

    Parameters:
        input_path (str): Path to the input audio file.
        output_path (str): Path where the trimmed audio will be saved.
        start_ms (int): Start time of the trim in milliseconds.
        end_ms (int): End time of the trim in milliseconds.
    """
    if file_exists(output_path):
        trimmed_audio = AudioSegment.from_file(output_path)
    else:
        audio = AudioSegment.from_file(input_path)
        trimmed_audio = audio[start_ms:end_ms]
        trimmed_audio.export(
            output_path, format="wav"
        )  # You can change the format if needed
    return trimmed_audio


def total_milliseconds(seconds):
    """
    Calculate the total number of milliseconds from a number of seconds.

    Parameters:
        seconds (seconds): The number of seconds for which to calculate the total milliseconds.

    Returns:
        int: The total number of milliseconds in a number of seconds.
    """
    td = timedelta(seconds=seconds)
    return td.seconds * 1000 + td.microseconds // 1000


print("preprocessing test list...")
for item in test_list:
    input_name = f"audio_samples/{item['source']}.wav"
    start_formatted = convert_seconds_to_time_format(item["start"]).replace(":", "")
    end_formatted = convert_seconds_to_time_format(item["end"]).replace(":", "")
    # output_name = f"segments/{item['source']}-{start_formatted}-{end_formatted}.wav"
    output_name = os.path.join(current_directory, f"segments/{item['source']}-{start_formatted}-{end_formatted}.wav")
    start_ms = total_milliseconds(item["start"])
    end_ms = total_milliseconds(item["end"])
    trimmed_audio = trim_audio(input_name, output_name, start_ms, end_ms)
    start_ms_td, end_ms_td = timedelta(milliseconds=start_ms), timedelta(
        milliseconds=end_ms
    )
    dur_original = (end_ms_td - start_ms_td).total_seconds()
    dur_final = trimmed_audio.duration_seconds
    print(output_name, start_ms, end_ms, dur_original, dur_final)
    assert dur_original == dur_final
    item["audio"] = output_name
    item["sentence"] = item["label"]

print("preprocessing val list...")
for item in val_list:
    input_name = f"audio_samples/{item['source']}.wav"
    start_formatted = convert_seconds_to_time_format(item["start"]).replace(":", "")
    end_formatted = convert_seconds_to_time_format(item["end"]).replace(":", "")
    # output_name = f"segments/{item['source']}-{start_formatted}-{end_formatted}.wav"
    output_name = os.path.join(current_directory, f"segments/{item['source']}-{start_formatted}-{end_formatted}.wav")
    start_ms = total_milliseconds(item["start"])
    end_ms = total_milliseconds(item["end"])
    trimmed_audio = trim_audio(input_name, output_name, start_ms, end_ms)
    start_ms_td, end_ms_td = timedelta(milliseconds=start_ms), timedelta(
        milliseconds=end_ms
    )
    dur_original = (end_ms_td - start_ms_td).total_seconds()
    dur_final = trimmed_audio.duration_seconds
    print(output_name, start_ms, end_ms, dur_original, dur_final)
    assert dur_original == dur_final
    item["audio"] = output_name
    item["sentence"] = item["label"]

print("preprocessing train list...")
for item in train_list:
    input_name = f"audio_samples/{item['source']}.wav"
    start_formatted = convert_seconds_to_time_format(item["start"]).replace(":", "")
    end_formatted = convert_seconds_to_time_format(item["end"]).replace(":", "")
    # output_name = f"segments/{item['source']}-{start_formatted}-{end_formatted}.wav"
    output_name = os.path.join(current_directory, f"segments/{item['source']}-{start_formatted}-{end_formatted}.wav")
    start_ms = total_milliseconds(item["start"])
    end_ms = total_milliseconds(item["end"])
    trimmed_audio = trim_audio(input_name, output_name, start_ms, end_ms)
    start_ms_td, end_ms_td = timedelta(milliseconds=start_ms), timedelta(
        milliseconds=end_ms
    )
    dur_original = (end_ms_td - start_ms_td).total_seconds()
    dur_final = trimmed_audio.duration_seconds
    print(output_name, start_ms, end_ms, dur_original, dur_final)
    assert dur_original == dur_final
    item["audio"] = output_name
    item["sentence"] = item["label"]

dataset_dict = DatasetDict(
    {
        "train": Dataset.from_list(train_list),
        "validation": Dataset.from_list(val_list),
        "test": Dataset.from_list(test_list),
    }
)

print(dataset_dict)

save_dir = "datasets/olivia_segments"
dataset_dict.save_to_disk(save_dir)
