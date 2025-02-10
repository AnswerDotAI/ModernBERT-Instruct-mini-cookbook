# ModernBERT-Large-Instruct

Welcome to this mini cookbook repository! It only contains two scripts, which should be all you need to get started with understanding how ModernBERT-Large-Instruct works and get to playing with it!

- `train.py` is a script that trains the model, with the Answer Token Prediction objective, and the possiblity to mix in a proportion of the normal MLM objective or the Dummy MLM objective from the paper for a certain proportion of examples. This is the script you'll want to use to fine-tune it for your own tasks!
- `eval.py` runs the evaluation of `ModernBERT-Large-Instruct` on the MMLU dataset, to give you an idea of what that looks like. The script is purposefully very unoptimized so you can truly see how each step works!

## Installation

To run these scripts, you only really need the basics: `transformers` (a version that supports ModernBERT-Large-Instruct), `datasets`, `torch`, `tqdm`. For evaluation, you'll also need `sklearn` for the accuracy score, but that's really only for convenience!

```bash
pip install -r requirements.txt
```

If you have a GPU, we assume that you'll be using the flash attention 2 implementation, as it's by far the most efficient one. As a result, you also need to install the `flash-attn` package:

```bash
pip install flash-attn
```

## Training

To train the model, you can use the `train.py` script. Check out all of its arguments to get a good idea of how it works. The two key things you'll need to modify is:

- Provide your own label map, so we know how to map your labels' column to the answer token.
- Modify the prompt to match your actual task.
ss
In AGNews, we use verbose labels, but it works really well by using letter prefixes for each answers! In fact, this is how we perform most tasks. In `sweep_20newsgroups.py`, you can see how we finetuned the model on 20newsgroups, for example!

## Inference

Inference is really simple, because all we're doing is running a single forward pass using the Masked Language Modelling head. The `eval.py` is a standalone and shows you how the model was evaluated on MMLU.

If you just want to get the model up and running really quick, this is a fully-contained inference example:

```python
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Load model and tokenizer
model_name = "answerdotai/ModernBERT-Large-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    model = AutoModelForMaskedLM.from_pretrained(model_name, attn_implementation="flash_attention_2")
else:
    model = AutoModelForMaskedLM.from_pretrained(model_name)

model.to(device)

# Format input for classification or multiple choice. This is a random example from MMLU.
text = """You will be given a question and options. Select the right answer.
QUESTION: If (G, .) is a group such that (ab)^-1 = a^-1b^-1, for all a, b in G, then G is a/an
CHOICES:
- A: commutative semi group
- B: abelian group
- C: non-abelian group
- D: None of these
ANSWER: [unused0] [MASK]"""

# Get prediction
inputs = tokenizer(text, return_tensors="pt").to(device)
outputs = model(**inputs)
mask_idx = (inputs.input_ids == tokenizer.mask_token_id).nonzero()[0, 1]
pred_id = outputs.logits[0, mask_idx].argmax()
answer = tokenizer.decode(pred_id)
print(f"Predicted answer: {answer}")  # Outputs: B
```

## Citation

If you use this code or build upon the ModernBERT-Large-Instruct model or idea, please cite the following paper:

```bibtex
@misc{clavié2025itsmasksimpleinstructiontuning,
      title={It's All in The [MASK]: Simple Instruction-Tuning Enables BERT-like Masked Language Models As Generative Classifiers}, 
      author={Benjamin Clavié and Nathan Cooper and Benjamin Warner},
      year={2025},
      eprint={2502.03793},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.03793}, 
}
```
