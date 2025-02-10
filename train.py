import argparse
import random
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    Trainer, 
    TrainingArguments,
    set_seed
)

class CustomMLMDataset(Dataset):
    def __init__(self, dataset, tokenizer, prompt, label_map, is_test=False, true_mlm_proportion=0.0, dummy_examples=False):
        """
        Args:
            dataset: A Hugging Face dataset split.
            tokenizer: A transformers tokenizer.
            is_test: Boolean indicating if this is a test/evaluation dataset.
            true_mlm_proportion: Float probability for using full MLM masking.
            dummy_examples: Boolean flag for using dummy examples.
            label_map: A dict mapping raw label names to desired label names.
        """
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.mlm_probability = 0.3
        self.is_test = is_test
        self.true_mlm_proportions = true_mlm_proportion
        self.dummy_examples = dummy_examples
        # Provide a default label map if one is not given.
        self.label_map = label_map
        self.prompt = prompt

    def __len__(self):
        return len(self.dataset)
   
    def __getitem__(self, idx):
        item = self.dataset[idx]
        # Convert the raw label using the provided label map.
        label = self.label_map[item['label_text']]

        # Format the input text with instructions and options.
        inputs = self.prompt.format(txt=item['text'].strip(), label=label)

        tokenized_input = self.tokenizer(
            inputs,
            padding='max_length', 
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        if self.dummy_examples:
            input_ids = tokenized_input['input_ids'][0]
            attention_mask = tokenized_input['attention_mask'][0]
        else:
            # Clone to avoid modifying the original tensors.
            input_ids = tokenized_input['input_ids'][0].clone()
            attention_mask = tokenized_input['attention_mask'][0].clone()

        labels = torch.full_like(input_ids, -100)
        
        # Decide if we use full MLM masking or only mask the last non-special token.
        do_real_mlm = random.random() < self.true_mlm_proportions

        if do_real_mlm and not self.is_test:
            # Create a probability matrix for MLM masking.
            probability_matrix = torch.full_like(input_ids, self.mlm_probability, dtype=torch.float)
            special_tokens_mask = torch.tensor(
                self.tokenizer.get_special_tokens_mask(input_ids.tolist(), already_has_special_tokens=True),
                dtype=torch.bool
            )
            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            
            input_ids[masked_indices] = self.tokenizer.mask_token_id
            labels[masked_indices] = tokenized_input['input_ids'][0][masked_indices]
        else:
            # Only mask the last token before the final [SEP] (or last token with attention).
            sep_positions = (input_ids == self.tokenizer.sep_token_id).nonzero()
            if len(sep_positions) > 0:
                last_non_sep = sep_positions[-1].item() - 1
            else:
                last_non_sep = (attention_mask == 1).nonzero()[-1].item()
                
            original_token = input_ids[last_non_sep].clone()
            input_ids[last_non_sep] = self.tokenizer.mask_token_id
            labels[last_non_sep] = original_token
            
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def train_model(args):
    # Load raw datasets.
    print(f"Loading dataset: {args.dataset}")
    raw_ds = load_dataset(args.dataset)
    raw_train = raw_ds['train']
    raw_test = raw_ds['test']
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model, attn_implementation="flash_attention_2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    prompt = """** Instructions **    
Given a short news report, determine which news category it belongs to.
** Content **
What is the category of the following news report?
News Report: {txt}
** Options **
- Business: It is about business or financial news.
- World: It's about world news.
- Sports: It's about sports.
- Tech: It's about Science or Technology, or related companies.
** Answer **
Answer: [unused0] {label}"""

    label_map = json.load(open(args.label_map))

    # Create our custom datasets.
    train_dataset = CustomMLMDataset(
        raw_train,
        tokenizer,
        prompt=prompt,
        label_map=label_map,
        is_test=False,
        true_mlm_proportion=args.true_mlm_proportion,
        dummy_examples=args.dummy_examples,
        max_length=args.max_length
    )
    eval_dataset = CustomMLMDataset(
        raw_test,
        tokenizer,
        prompt=prompt,
        label_map=label_map,
        is_test=True,
        true_mlm_proportion=args.true_mlm_proportion,
        dummy_examples=args.dummy_examples,
        max_length=args.max_length
    )

    # Configure training arguments.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        logging_dir='./logs',
        bf16=True,
        logging_steps=args.logging_steps,
        evaluation_strategy=args.eval_strategy,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_steps=args.eval_steps
    )

    # Initialize the Trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Begin training.
    print("Starting training...")
    trainer.train()

    # Save the model and tokenizer.
    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete and model saved.")


def main():
    parser = argparse.ArgumentParser(description="Finetune a model ModernBERT-Instruct style.")
    parser.add_argument("--num_train_epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per device during training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size per device during evaluation.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every X updates steps.")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate every X updates steps.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--output_dir", type=str, default="./finetuned_model_agn", help="Directory to save the final model.")
    parser.add_argument("--dataset", type=str, default="SetFit/ag_news", help="Name of the dataset to load (from the Hugging Face hub). In this example, we use the AG News dataset.")
    parser.add_argument("--model", type=str, default="answerdotai/ModernBERT-Large-Instruct", help="Path or identifier of the pretrained model. Use a raw BERT-like model like answerdotai/ModernBERT-Large if pretraining, or answerdotai/ModernBERT-Instruct-best if finetuning.")
    parser.add_argument("--true_mlm_proportion", type=float, default=0.0, help="Proportion of training samples to use normal MLM masking on rather than Answer Token Prediction.")
    parser.add_argument("--save-strategy", type=str, default="no", help="Whether to save the model and tokenizer.")
    parser.add_argument("--save-steps", type=int, default=500, help="Save the model and tokenizer every X updates steps.")
    parser.add_argument("--save-total-limit", type=int, default=1, help="Total number of checkpoints to save.")
    parser.add_argument("--eval-strategy", type=str, default="no", help="Whether to evaluate the model during training.")
    parser.add_argument("--eval-steps", type=int, default=500, help="Evaluate the model every X updates steps.")
    parser.add_argument("--dummy_examples", action='store_true', help="If set, use dummy labels for [MASK] instead of cloning tensors.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum length of the input text. 2048 was used for pretraining, but AGNews is very short so we use 512/")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--label_map", type=str, default="./agnews_examples/agnews_label_map.json", help="Path to a JSON file that contains the label map for the dataset.")
    args = parser.parse_args()

    # Set seeds for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    set_seed(args.seed)


    train_model(args)

if __name__ == "__main__":
    random.seed(42)
    main()