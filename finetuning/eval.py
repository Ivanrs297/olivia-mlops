# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch

# import the relavant libraries for loggin in
from huggingface_hub import HfApi, HfFolder

from torcheval.metrics.functional import word_error_rate

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Functions and procedures

import unicodedata
import string

def normalize_text(text):
    # Remove punctuation
    translator = str.maketrans("", "", string.punctuation)
    text_without_punctuation = text.translate(translator)
    
    # Lowercase the text
    normalized_text = text_without_punctuation.lower()
    
    # Normalize Unicode characters (e.g., accented characters)
    normalized_text = unicodedata.normalize('NFD', normalized_text)
    normalized_text = ''.join([char for char in normalized_text if not unicodedata.combining(char)])
    
    return normalized_text


def login_hugging_face(token: str) -> None:
    """
    Loging to Hugging Face portal with a given token.
    """
    api = HfApi(token=token)
    # api.set_access_token(token)
    folder = HfFolder()
    folder.save_token(token)

    return None


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Use Data Collator to perform Speech Seq2Seq with padding
    """
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def compute_metrics(pred_str, label_str):
    """
    Define evaluation metrics. We will use the Word Error Rate (WER) metric.
    For more information, check:
    """

    wer = 100 * word_error_rate(pred_str, label_str) #metric.compute(predictions=pred_str, references=label_str)

    return wer

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# STEP 0. Loging to Hugging Face
# get your account token from https://huggingface.co/settings/tokens
token = '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'
login_hugging_face(token)
print('We are logged in to Hugging Face now!')


# STEP 1. Download Dataset
from datasets import DatasetDict, load_from_disk

common_voice = load_from_disk("datasets/olivia_segments")
common_voice = common_voice.remove_columns(
    ['_id', 'index', 'label', 'start', 'end', 'state', 'source', 'word_count']
)

print(common_voice)

####### ---------------- ########

print(len(common_voice["validation"]))

import whisperx
import gc

device = "cuda" 
batch_size = 8 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
model = whisperx.load_model("large-v2", device, compute_type=compute_type, language="es")

import numpy as np
from tqdm import tqdm 

error = 0
i = 0
for example in tqdm(common_voice["validation"]):
    # device = "cuda" 
    # batch_size = 8 # reduce if low on GPU mem
    # compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
    
    # 1. Transcribe with original whisper (batched)
    # model = whisperx.load_model("large-v2", device, compute_type=compute_type, language="es")
    
    audio = whisperx.load_audio(example["audio"])
    result = model.transcribe(audio, batch_size=batch_size)
    # print(result["segments"]) # before alignment
    
    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache();
    # del model
    
    # 2. Align whisper output
    # model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    # result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    
    # print(result) # after alignment

    prediction = " ".join([s["text"] for s in result["segments"]])
    ground_truth = example["sentence"]

    prediction = normalize_text(prediction)
    ground_truth = normalize_text(ground_truth)
    
    error = error + compute_metrics([prediction], [ground_truth])
    # print("wer", error)
    
    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

print("Average error", error / len(common_voice["validation"]))