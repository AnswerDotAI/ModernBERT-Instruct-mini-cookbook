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
