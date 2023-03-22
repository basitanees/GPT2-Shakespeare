import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForPreTraining,
    AdamW,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    BeamScorer,
    Trainer,
)
from params import MODEL


def get_tokenizer(special_tokens=None):
    tokenizer = AutoTokenizer.from_pretrained(MODEL)  # GPT2Tokenizer

    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
        print("Special tokens added")
    return tokenizer


def get_model(tokenizer, special_tokens=None, load_model_path=None):

    # GPT2LMHeadModel
    if special_tokens:
        config = AutoConfig.from_pretrained(
            MODEL,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            sep_token_id=tokenizer.sep_token_id,
            pad_token_id=tokenizer.pad_token_id,
            output_hidden_states=False,
        )
    else:
        config = AutoConfig.from_pretrained(
            MODEL, pad_token_id=tokenizer.eos_token_id, output_hidden_states=False
        )

    # ----------------------------------------------------------------#
    model = AutoModelForPreTraining.from_pretrained(
        MODEL, config=config
    )  # .to(torch.device("cpu"))

    if special_tokens:
        # Special tokens added, model needs to be resized accordingly
        model.resize_token_embeddings(len(tokenizer))

    if load_model_path:
        model.load_state_dict(
            torch.load(load_model_path, map_location=torch.device("cpu"))
        )

    # model.cuda()
    return model
