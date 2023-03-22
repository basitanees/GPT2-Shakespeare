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

The response dict contains a return sequence in the form:
    {
        "output": "text"
    }

In order to finetune the GPT-2 model, we use the official huggingface GPT-2 pretrained model. The finetuning notebook can be viewed here: https://colab.research.google.com/drive/1aK5DL6om3LkXdMWr7E1F8BPE5Vo1bgyD?usp=sharing

During finetuning, we keep the first six transformer hidden layers frozen. Remaining are finetuned.
For the dataset, we preprocess the input sequences such that that each character and their statement is considered a single sequence. We append the bos and eos token at the start and end respectively for each sequence. We finetune for 4 epochs. 

Note: I was not sure if hosting on a cloud computing service meant hosting the code on github or deploying the code on a service like aws. For now, I have uploaded the repo to Github which can be run locally by cloning, installing requirements and running start.sh