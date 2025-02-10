import os
import json
import random
import torch
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    Trainer, 
    TrainingArguments
)
from torch.utils.data import Dataset

# Set global seed
random.seed(42)
torch.manual_seed(42)

class CustomMLMDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, template_fn, max_length=512, is_test=False, true_mlm_proportion=0.0):
        """
        hf_dataset: a Hugging Face dataset object (train or test)
        tokenizer: our tokenizer
        template_fn: function(example) -> (prompt_text, correct_label_letter)
        max_length: maximum token length for tokenization
        is_test: whether this is a test split
        true_mlm_proportion: chance to do “real” MLM masking
        """
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.template_fn = template_fn
        self.max_length = max_length
        self.mlm_probability = 0.3
        self.is_test = is_test
        self.true_mlm_proportion = true_mlm_proportion

    def __len__(self):
        return len(self.hf_dataset)
   
    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        prompt, label_letter = self.template_fn(item)
        tokenized = self.tokenizer(
            prompt,
            padding='max_length', 
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = tokenized['input_ids'][0].clone()
        attention_mask = tokenized['attention_mask'][0].clone()
        labels_tensor = torch.full_like(input_ids, -100)

        # Either do “real” MLM or mask the last token of the answer
        do_real_mlm = (random.random() < self.true_mlm_proportion) and (not self.is_test)
        if do_real_mlm:
            probability_matrix = torch.full_like(input_ids, self.mlm_probability, dtype=torch.float)
            special_tokens_mask = torch.tensor(
                self.tokenizer.get_special_tokens_mask(input_ids.tolist(), already_has_special_tokens=True),
                dtype=torch.bool
            )
            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            
            input_ids[masked_indices] = self.tokenizer.mask_token_id
            labels_tensor[masked_indices] = tokenized['input_ids'][0][masked_indices]
        else:
            # In our templates we assume a single [MASK] appears in the answer area
            sep_positions = (input_ids == self.tokenizer.sep_token_id).nonzero()
            if len(sep_positions) > 0:
                last_non_sep = sep_positions[-1].item() - 1
            else:
                last_non_sep = (attention_mask == 1).nonzero()[-1].item()
            original_token = input_ids[last_non_sep].clone()
            input_ids[last_non_sep] = self.tokenizer.mask_token_id
            labels_tensor[last_non_sep] = original_token

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels_tensor,
            # We also keep the correct answer letter for evaluation purposes.
            'label_letter': label_letter  
        }

def save_metrics(metrics, out_path):
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=4)


def newsgroup_template(example):
    label_descriptions = {
        'alt.atheism': ('A', 'Discussion of atheism and criticism of religion'),
        'comp.graphics': ('B', 'Computer graphics, rendering, and visualization'),
        'comp.os.ms-windows.misc': ('C', 'General Microsoft Windows topics and issues'),
        'comp.sys.ibm.pc.hardware': ('D', 'IBM PC compatible hardware and components'),
        'comp.sys.mac.hardware': ('E', 'Apple Macintosh hardware and components'),
        'comp.windows.x': ('F', 'X Window System software and configuration'),
        'misc.forsale': ('G', 'General marketplace for selling items'),
        'rec.autos': ('H', 'Cars, automotive repair and motorsports'),
        'rec.motorcycles': ('I', 'Motorcycles, riding and maintenance'),
        'rec.sport.baseball': ('J', 'Baseball discussion and news'),
        'rec.sport.hockey': ('K', 'Hockey discussion and news'),
        'sci.crypt': ('L', 'Cryptography and data security'),
        'sci.electronics': ('M', 'Electronics and electrical engineering'),
        'sci.med': ('N', 'Medicine, health and medical science'),
        'sci.space': ('O', 'Space exploration and astronomy'),
        'soc.religion.christian': ('P', 'Christianity-specific religious discussion'),
        'talk.politics.guns': ('Q', 'Gun politics and firearm regulations'),
        'talk.politics.mideast': ('R', 'Middle East politics and current events'),
        'talk.politics.misc': ('S', 'General political discussion and debate'),
        'talk.religion.misc': ('T', 'General religious discussion across faiths')
    }
    options = "\n".join([f"{letter}: {label}, {desc}" for label, (letter, desc) in label_descriptions.items()])
    label_map = {label: letter for label, (letter, _) in label_descriptions.items()}
    text_content = example['text'].strip()
    label_letter = label_map[example['label_text']]
    prompt = f"""** Instructions **    
Given a short post from an online forum, determine which newsgroup category it belongs to.
** Content **
{text_content}
What's the category of the post above?
** Options **
{options}
** Answer **
Answer: [unused0] {label_letter}"""
    return prompt, label_letter


dataset_configs = {
    "20newsgroups": {
        "load_args": {"path": "SetFit/20_newsgroups"},
        "train_split": "train",
        "test_split": "test",
        "template_fn": newsgroup_template,
        "max_length": 4096  # as in the example; adjust if needed
    },
}

# --- Hyperparameter sweep settings ---
learning_rates = [2e-5, 3e-5, 5e-5]
epoch_counts = [1, 2, 3]

# Directory to save finetuned models and metrics
output_dir = Path("/sweep_results/mlm")
output_dir.mkdir(exist_ok=True)

BASE_MODEL = "answerdotai/ModernBERT-Large-Instruct"

# --- Evaluation function (using our MLM fill-in strategy) ---
def predict_answer(example, model, tokenizer, option_tokens):
    # Tokenize input and get logits
    inputs = tokenizer(example['text'], return_tensors="pt", max_length=8192, truncation=True).to('cuda')
    outputs = model(**inputs)
    
    # Identify the mask token position (we assume one mask token exists)
    mask_indices = (inputs.input_ids == tokenizer.mask_token_id).nonzero()
    if len(mask_indices) == 0:
        return {"pred": ""}
    # For simplicity, use the last mask token position
    mask_index = mask_indices[-1, 1]
    mask_logits = outputs.logits[0, mask_index]
    probabilities = torch.nn.functional.softmax(mask_logits, dim=-1)
    
    # For each option token, sum probabilities (in our templates, one token per option)
    option_probs = {}
    for opt in option_tokens:
        # We assume that tokenizing the option (e.g., "A") yields a single token after the special token.
        token_id = tokenizer.encode(opt, add_special_tokens=False)[0]
        option_probs[opt] = probabilities[token_id].item()
    predicted_token = max(option_probs, key=option_probs.get)
    return {"pred": predicted_token}

def evaluate_model(model, tokenizer, processed_test_dataset, template_fn):
    # Recreate the “processed” test examples (each as a dict with 'text' and 'text_label')
    processed_examples = []
    # For each example in the test set, we re-run the template function and record the ground truth.
    for ex in processed_test_dataset:
        prompt, label_letter = template_fn(ex)
        # Modify the prompt for evaluation: remove the provided answer letter and insert the mask token.
        # We assume that the prompt contains the unique substring "[unused0]".
        if "[unused0]" in prompt:
            # Split at "[unused0]" and add the mask token in its place.
            before, _ = prompt.split("[unused0]", 1)
            eval_prompt = before + "[unused0] " + tokenizer.mask_token
        else:
            # In case the marker isn't found, fall back to the original prompt.
            eval_prompt = prompt
        processed_examples.append({"text": eval_prompt, "text_label": label_letter})
    
    # Build the option tokens: we assume options are always letters starting with "A".
    option_set = sorted({ex["text_label"] for ex in processed_examples})
    option_tokens = option_set  # e.g. ["A", "B", "C", ...]
    
    correct = 0
    total = 0
    for ex in processed_examples:
        pred = predict_answer(ex, model, tokenizer, option_tokens)["pred"].strip()
        if pred == ex["text_label"]:
            correct += 1
        total += 1
    accuracy = correct / total if total > 0 else 0.0
    return accuracy

# --- Main sweep loop ---
for ds_name, config in dataset_configs.items():
    print(f"==== Processing dataset: {ds_name} ====")
    ds = load_dataset(**config["load_args"])
    train_ds = ds[config["train_split"]]
    test_ds = ds[config["test_split"]]
    template_fn = config["template_fn"]
    max_length = config["max_length"]

    # Create tokenizer and model for this run (starting from the same base model)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL, attn_implementation="flash_attention_2").to('cuda')
    
    # Create our dataset wrappers
    train_dataset = CustomMLMDataset(train_ds, tokenizer, template_fn, max_length=max_length, is_test=False)
    eval_dataset  = CustomMLMDataset(test_ds, tokenizer, template_fn, max_length=max_length, is_test=True)

    bs = 32
    wd = 0.01
    for lr in learning_rates:
        for epochs in epoch_counts:
            hp_str = f"{ds_name}_lr{lr}_ep{epochs}_wd{wd}_bs{bs}"
            print(f"--- Running sweep: {hp_str} ---")
            run_output_dir = output_dir / hp_str
            run_output_dir.mkdir(exist_ok=True, parents=True)

            per_device_batch_size = 1 if ds_name == "20newsgroups" else 8
            devices = 4
            accum_steps = bs // (devices * per_device_batch_size)
            
            training_args = TrainingArguments(
                output_dir=str(run_output_dir / "model"),
                num_train_epochs=epochs,
                per_device_train_batch_size=per_device_batch_size,
                per_device_eval_batch_size=per_device_batch_size,
                logging_dir=str(run_output_dir / "logs"),
                bf16=True,
                logging_steps=10,
                evaluation_strategy="no",  # we use our custom evaluation below
                learning_rate=lr,
                weight_decay=0.01,
                warmup_ratio=0.05,
                save_strategy='no',
                gradient_accumulation_steps=accum_steps
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
            )
            
            # Train the model
            trainer.train()
            trainer.save_model(str(run_output_dir / "finetuned_model"))
            tokenizer.save_pretrained(str(run_output_dir / "finetuned_model"))
            
            # Evaluate the model using our custom evaluation function.
            # For evaluation we build a processed test dataset by mapping the original test examples via template_fn.
            test_accuracy = evaluate_model(model, tokenizer, test_ds, template_fn)
            print(f"Accuracy for {hp_str}: {test_accuracy:.2%}")
            
            # Save metrics in a JSON file
            metrics = {
                "dataset": ds_name,
                "learning_rate": lr,
                "epochs": epochs,
                "weight_decay": wd,
                "accuracy": test_accuracy,
            }
            metrics_file = run_output_dir / "metrics.json"
            save_metrics(metrics, metrics_file)


            del model
            model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL, attn_implementation="flash_attention_2").to('cuda')
