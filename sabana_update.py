# # Load model
from torch import cuda, bfloat16
import transformers

model_id = "meta-llama/Llama-2-13b-chat-hf"  #'meta-llama/Llama-2-70b-chat-hf'

device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16,
)

# begin initializing HF items, need auth token for these
hf_auth = "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"
model_config = transformers.AutoConfig.from_pretrained(
    model_id, use_auth_token=hf_auth, cache_dir="./hub"
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map="auto",
    use_auth_token=hf_auth,
    cache_dir="./hub",
)
model.eval()
print(f"Model loaded on {device}")

# # Load tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id, use_auth_token=hf_auth, cache_dir="./hub"
)

stop_list = ["\nHuman:", "\n```\n"]

stop_token_ids = [tokenizer(x)["input_ids"] for x in stop_list]

import torch

stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]

from transformers import StoppingCriteria, StoppingCriteriaList


# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids) :], stop_ids).all():
                return True
        return False


stopping_criteria = StoppingCriteriaList([StopOnTokens()])

generate_text = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task="text-generation",
    # we pass model parameters here too
    stopping_criteria=stopping_criteria,  # without this model rambles during chat
    temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,  # mex number of tokens to generate in the output
    repetition_penalty=1.1,  # without this output begins repeating
)

from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma, FAISS
import re
from olivia_questions import questions

llm = HuggingFacePipeline(pipeline=generate_text)


PROMPT_TEMPLATE = """Basado en los siguientes fragmentos de una entrevista, responde la pregunta del final (si en los fragmentos no se encuentra la respuesta a la pregunta del final, contesta 'NA').

Fragmentos:
{context}

Pregunta:
{question}

Respuesta:
"""


def get_update(audio_id):
    # load document
    filepath = f"./diari/{audio_id}.vtt"
    loader = TextLoader(filepath, encoding="utf-8")
    documents = loader.load()

    # load embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    )

    # clean documents
    pattern = r"\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}\.\d{3}"
    source_documents = []
    for doc in documents:
        doc.page_content = doc.page_content.replace("WEBVTT", "")
        doc.page_content = re.sub(pattern, "", doc.page_content)
        doc.page_content = doc.page_content.replace("\n\n\n", "\n\n")
        source_documents.append(doc)

    # split documents into texts
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
    texts = text_splitter.split_documents(source_documents)

    # generate vectore store
    vectorstore = FAISS.from_documents(texts, embeddings)

    # generate QA chain
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": prompt}
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        # retriever=vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 6}),
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs=chain_type_kwargs,
        # verbose=True
    )

    # generate patch update
    update = {}
    for qn in questions:
        print("Q:", query)
        query = qn["question"]
        answer = qa_chain.run(query)
        answer = answer.split("\n")[0].strip()
        if "categories" in qn.keys() and qn["categories"] is list:
            cats = Chroma.from_texts(qn["categories"])
            scores = cats.similarity_search_with_score(answer)
            highest = max(scores, key=lambda x: x[1])
            if highest[1] < 0.5 and qn["otherKey"] is not None:
                update[qn["otherKey"]] = answer
                print("A:", answer)
            else:
                update["key"] = highest[0]
                print("A:", highest[0])
        else:
            update["key"] = answer
            print("A:", answer)
        print("\n")

    return update

import requests


def connect():
    # change endpoint and data
    url = f"http://localhost:8080/api/login"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    data = {"email": "", "password": ""}
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        token = response.json()
    else:
        raise Exception(response.text)
    return token


def get_expedientes(token):
    # TODO: change endpoint
    url = f"http://localhost:8080/api/expediente"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        audio_list = response.json()
    else:
        raise Exception(response.text)
    return audio_list

def update_expediente(token, audio_id, update):
    # TODO: change endpoint
    url = f"https://api.example.com/resource/{audio_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    data = update
    response = requests.patch(url, json=data, headers=headers)
    if response.status_code == 200:
        print("PATCH request was successful.")
    else:
        raise Exception(response.text)


import whisperx
import gc
device = "cuda" 
batch_size = 8 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
tmodel = whisperx.load_model("large-v2", device, compute_type=compute_type, language="es")
model_a, metadata = whisperx.load_align_model(language_code="es", device=device)

def transcribe(audiopath):
    # 1. Transcribe with original whisper (batched)
    audio = whisperx.load_audio(audiopath)
    result = tmodel.transcribe(audio, batch_size=batch_size)
    # 2. Align whisper output
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    prediction = " ".join([s["text"] for s in result["segments"]])
    return prediction

from glob import glob
from pathlib import Path
import time

def run():
    token = connect()
    audios = glob("audio_samples")
    for audio in audios:
        print(f"Processing {audio_id}\n")
        start_time = time.time()
        audiopath = Path(audio)
        audio_id = audiopath.stem
        transcription = transcribe(audiopath)
        try:
            update = get_update(audio_id)
            update_expediente(token, audio_id, update)
        except Exception as error:
            print(f"Error when processing {audio_id}:", error)
        end_time = time.time()
        execution_time = end_time - start_time
        print("\nExecution time:", execution_time, "seconds")
        print()