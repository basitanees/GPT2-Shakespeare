DEBUG = False

INPUT_DIR = "articles"

USE_APEX = True
APEX_OPT_LEVEL = "O1"

MODEL = "gpt2"  # {gpt2, gpt2-medium, gpt2-large, gpt2-xl}

UNFREEZE_LAST_N = 6  # The last N layers to unfreeze for training

SPECIAL_TOKENS = {
    "bos_token": "<|BOS|>",
    "eos_token": "<|EOS|>",
    "unk_token": "<|UNK|>",
    "pad_token": "<|PAD|>",
    "sep_token": "<|SEP|>",
}

MAXLEN = 768  # {768, 1024, 1280, 1600}

TRAIN_SIZE = 0.8

if USE_APEX:
    TRAIN_BATCHSIZE = 4
    BATCH_UPDATE = 16
else:
    TRAIN_BATCHSIZE = 2
    BATCH_UPDATE = 32

EPOCHS = 4
LR = 5e-4
EPS = 1e-8
WARMUP_STEPS = 1e2

SEED = 2020

URL = "https://drive.google.com/uc?id=1xf9C4NuO4SCufVFaULauR26AecTQOLLa&confirm=t"
OUTPUT_FILE = "pytorch_model.bin"
