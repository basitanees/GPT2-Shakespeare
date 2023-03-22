# GPT2-Shakespeare

## GPT-2 finetuned on Shakespeare dataset

This repo hosts a GPT-2 model finetuned on the Shakespeare dataset: 
* https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

Run the command bash start.sh to host the model on http://127.0.0.1:8000. You may either test it on the docs page at:
* http://127.0.0.1:8000/docs
* or post the payload to http://127.0.0.1:8000/predict/ in the format:
    {
        "text": "Your text here"
    }

The response dict contains a list of 10 possible return sequences in the form:
    {
        "output": ["text1", "text2", ..., "text10"]
    }

