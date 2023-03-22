from fastapi import FastAPI
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from typing import List
import torch
from utils import get_model, get_tokenizer
from params import SPECIAL_TOKENS, MAXLEN, OUTPUT_FILE, URL
import gdown
import os


if not os.path.exists(OUTPUT_FILE):
    gdown.download(URL, OUTPUT_FILE)
tokenizer = get_tokenizer(special_tokens=SPECIAL_TOKENS)
model = get_model(tokenizer, special_tokens=SPECIAL_TOKENS, load_model_path=OUTPUT_FILE)
model.eval()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("cpu")
model = model.to(device)


app = FastAPI(title="Shakespeare")

# allow CORS requests from any host so that the JavaScript can communicate with the server
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class Request(BaseModel):
    text: str


class AllModelsResponse(BaseModel):
    output: List[str]


@app.post("/predict", response_model=AllModelsResponse)
async def predict(request: Request):
    prompt = SPECIAL_TOKENS["bos_token"] + request.text
    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    generated = generated.to(device)

    output = []
    # generate 10 samples
    sample_outputs = model.generate(
        generated,
        do_sample=True,
        min_length=50,
        max_length=MAXLEN,
        top_k=30,
        top_p=0.7,
        temperature=0.9,
        repetition_penalty=2.0,
        num_return_sequences=10,
    )

    for i, sample_output in enumerate(sample_outputs):
        text = tokenizer.decode(sample_output, skip_special_tokens=True)
        output.append(text)

    response = AllModelsResponse(output=output)
    return response
