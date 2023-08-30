# # Load model
from torch import cuda, bfloat16
import transformers
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from olivia_questions import questions
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma, FAISS
from olivia_questions import questions
import requests
import whisperx
import gc
from glob import glob
from pathlib import Path
import time
import subprocess
import os

# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids) :], stop_ids).all():
                return True
        return False
    
def get_update(audio_id):
    # load document
    filepath = f"./transcriptions/{audio_id}.vtt"
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
        query = qn["question"]
        print("Q:", query)
        answer = qa_chain.run(query)
        answer = answer.split("\n")[0].strip()
        if "categories" in qn.keys():
            print("Available Categories", qn["categories"])
            cats = FAISS.from_texts(qn["categories"], embeddings)
            scores = cats.similarity_search_with_score(answer)
            highest = max(scores, key=lambda x: x[1])
            if highest[1] < 0.5 and qn["otherKey"] is not None:
                update[qn["otherKey"]] = answer
                print("A:", answer)
            else:
                update[qn["key"]] = highest[0].page_content
                print("A:", highest[0].page_content)
        elif "type" in qn.keys() and qn["type"] == "boolean":
            update[qn["key"]] = len(answer) > 0
            print("A:", len(answer) > 0)
        else:
            update[qn["key"]] = answer
            print("A:", answer)
        print("\n")

    return update

def connect():
    # change endpoint and data
    # url = f"http://localhost:8080/api/login"
    url = f"https://api.olivia-fairlac.org/api/login"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    data = {"email": "david@gmail.com", "password": "ol1v14"}
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        res = response.json()
        token = res["token"]
    else:
        raise Exception(response.text)
    return token

def get_cedula_by_expediente_id(token, id):
    url = f"https://api.olivia-fairlac.org/api/cedula/{id}"
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

def get_expediente_by_id(token, id):
    # TODO: change endpoint
    url = f"https://api.olivia-fairlac.org/api/expediente/{id}"
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

def get_expedientes(token):
    # TODO: change endpoint
    url = f"https://api.olivia-fairlac.org/api/expediente"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    response = requests.get(url, headers=headers)
    results = []
    if response.status_code == 200:
        items = response.json()["docs"]
        for item in items:
            details = get_expediente_by_id(token, item["_id"])
            cedula = get_cedula_by_expediente_id(token, item["_id"])
            results.append({**item, **cedula, **details})
    else:
        raise Exception(response.text)
    return results

def update_expediente(token, expediente_id, update):
    res = get_cedula_by_expediente_id(token, expediente_id)
    cedula_id = res["cedula"]["_id"]
    url = f"https://api.olivia-fairlac.org/api/cedula/{cedula_id}?expediente={expediente_id}"
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

def download_audio(url, exp_id):
    response = requests.get(url)
    if response.status_code == 200:
        with open(f"pending/{exp_id}.wav", "wb") as file:
            file.write(response.content)
        print("File downloaded successfully.")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

def transcribe(audio_path):
    command = f"whisperx {audio_path} --model large-v2 --align_model WAV2VEC2_ASR_LARGE_LV60K_960H --batch_size 4 --compute_type int8 --output_dir transcriptions --language es --output_format vtt --hf_token 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"
    
    # Split the command into a list of arguments
    args = command.split()
    
    # Use subprocess to run the command
    try:
        subprocess.run(args, check=True)
        return 0
    except subprocess.CalledProcessError as e:
        print("Command execution failed.")

def transcribe_pending():
    audios = glob("pending/*")
    for audio in audios:
        print(f"Processing {audio}\n")
        start_time = time.time()
        audiopath = Path(audio)
        audio_id = audiopath.stem
        r = transcribe(audiopath)
        if r == 0:
            try:
                os.remove(audiopath)
                print("Processed")
                print()
            except OSError as e:
                print(f"Error removing the file: {e}")
                print()

def answer_pending():
    trans = glob("transcriptions/*")
    for audio in trans:
        print(f"Processing {audio}\n")
        start_time = time.time()
        audiopath = Path(audio)
        audio_id = audiopath.stem
        try:
            update = get_update(audio_id)
            update_expediente(token, audio_id, update)
            try:
                os.remove(audiopath)
                print("Processed")
                print()
            except OSError as e:
                print(f"Error removing the file: {e}")
                print()
        except Exception as error:
            print(f"Error when processing {audio_id}:", error)
            print()
        end_time = time.time()
        execution_time = end_time - start_time
        print("\nExecution time:", execution_time, "seconds")
        print()
        break
    return update

def update_expediente(token, expediente_id, update):
    res = get_cedula_by_expediente_id(token, expediente_id)
    cedula_id = res["cedula"]["_id"]
    url = f"https://api.olivia-fairlac.org/api/cedula/{cedula_id}?expediente={expediente_id}"
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

def run():
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

    stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]

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

    llm = HuggingFacePipeline(pipeline=generate_text)


    PROMPT_TEMPLATE = """Basado en los siguientes fragmentos de una entrevista, responde la pregunta del final (si en los fragmentos no se encuentra la respuesta a la pregunta del final, responde 'NA').

    Fragmentos:
    {context}

    Pregunta:
    {question}

    Respuesta:
    """

    device = "cuda" 
    batch_size = 8 # reduce if low on GPU mem
    compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
    # tmodel = whisperx.load_model("large-v2", device, compute_type=compute_type, language="es")
    # model_a, metadata = whisperx.load_align_model(language_code="es", device=device)

    token = connect()

    tmodel = whisperx.load_model("large-v2", device, compute_type=compute_type, language="es")
    model_a, metadata = whisperx.load_align_model(language_code="es", device=device)

    transcribe_pending()

    upt = answer_pending()

    update_expediente(token, "64dbb5286fbd221ffb8209ec", upt)

if __name__ == "__main__":
    run()