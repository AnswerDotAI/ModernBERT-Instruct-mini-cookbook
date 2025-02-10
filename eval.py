import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score

##############################################
# MMLU Prompt Formatting (MLM style)
##############################################
def format_mmlu_example(example):
    """
    Formats a single example from the MMLU dataset into a prompt suitable for 
    masked language modeling evaluation.
    
    The prompt includes a question, a list of answer choices, and a masked slot 
    for the answer. The gold (correct) answer is converted into its corresponding 
    letter.
    
    Args:
        example (dict): A single example from the MMLU dataset. Expected to contain:
            - 'question': the question text.
            - 'choices': a list of answer choices.
            - 'answer': an integer index (0-3) corresponding to the correct answer.
    
    Returns:
        dict: A dictionary with:
            - 'text': the formatted prompt text.
            - 'label': the correct answer letter (e.g., 'A', 'B', 'C', or 'D').
    """
    mc_prefix = (
        "You will be given a question as well as a list of options. "
        "Read the question carefully and select the right answer from the list.\n"
        "QUESTION:\n"
    )
    letters = ['A', 'B', 'C', 'D']
    
    # Extract the question and choices.
    question = example['question'].strip()
    choices = example['choices']
    # Map the numerical answer (0-3) to a corresponding letter.
    label = letters[example['answer']]
    
    # Build the prompt text.
    text = mc_prefix + question + "\nCHOICES:\n"
    for i, choice in enumerate(choices):
        text += f"- {letters[i]}: {choice}\n"
    text += "\nANSWER:\nAnswer: [unused0] [MASK]"
    
    return {'text': text, 'label': label}


@torch.no_grad()
def predict_answer_mlm(example, model, tokenizer, option_mapping, option_tokens, model_name):
    """
    Given a formatted prompt example with a masked token, this function performs 
    a forward pass through the model, locates the [MASK] token, and computes 
    probabilities over candidate answer tokens.
    
    Args:
        example (dict): A dictionary with keys 'text' and 'label' (gold answer).
        model (PreTrainedModel): The masked language model.
        tokenizer (PreTrainedTokenizer): The tokenizer corresponding to the model.
        option_mapping (dict): Mapping from candidate token strings to canonical answer letters.
        option_tokens (list): List of candidate token strings to check (e.g., "A", "B", etc.).
        model_name (str): The model name (used for any necessary text adjustments).
    
    Returns:
        str: The predicted answer letter.
    """
    inp_text = example['text']
    
    # Tokenize the prompt text.
    inputs = tokenizer(inp_text, return_tensors="pt").to(model.device)
    outputs = model(**inputs)
    
    # Locate all indices for the mask token; use the last occurrence.
    mask_indices = (inputs.input_ids == tokenizer.mask_token_id).nonzero()
    mask_logits = outputs.logits[0, mask_indices[-1, 1]]
    
    # Compute softmax probabilities over the vocabulary.
    probabilities = torch.nn.functional.softmax(mask_logits, dim=-1)
    
    # For each candidate option, get the probability of its token.
    option_ids = [tokenizer.encode(t)[1] for t in option_tokens]
    option_probs = {}
    for token, token_id in zip(option_tokens, option_ids):
        canonical = option_mapping[token]
        option_probs[canonical] = option_probs.get(canonical, 0) + probabilities[token_id].item()
    
    # Select the option (letter) with the highest probability.
    predicted_token = max(option_probs.items(), key=lambda x: x[1])[0]
    return predicted_token


def evaluate_mmlu(model, tokenizer, examples):
    """
    Evaluates the model on a list of formatted MMLU examples.
    
    It runs the prediction function on each example, compares the predicted 
    answer letter to the gold answer, and computes overall accuracy.
    
    Args:
        model (PreTrainedModel): The masked language model.
        tokenizer (PreTrainedTokenizer): The model's tokenizer.
        examples (list): List of formatted examples (each a dict with 'text' and 'label').
    
    Returns:
        tuple: (accuracy, predictions list, gold answers list)
    """

    # Deprecated but kept for legacy reasons pre-tokenizer fix.
    option_mapping = {
        "A": "A", " A": "A",
        "B": "B", " B": "B",
        "C": "C", " C": "C",
        "D": "D", " D": "D",
    }
    option_tokens = ["A", " A", "B", " B",
                     "C", " C", "D", " D",]
    
    preds = []
    golds = []
    
    # Evaluate each example.
    for ex in tqdm(examples, desc="Evaluating MMLU"):
        gold = ex.get('label')
        pred = predict_answer_mlm(ex, model, tokenizer, option_mapping, option_tokens, model.config._name_or_path)
        preds.append(pred.strip())
        golds.append(gold.strip())
    
    # Compute accuracy over the entire dataset.
    acc = accuracy_score(golds, preds)
    return acc, preds, golds

##############################################
# Main Execution
##############################################
def main():
    # Choose device: use CUDA if available.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Specify the base model name to evaluate.
    model_name = "answerdotai/ModernBERT-Large-Instruct"
    print(f"Loading model and tokenizer for '{model_name}' ...")
    
    # Load tokenizer and model.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    try:
        model = AutoModelForMaskedLM.from_pretrained(model_name, attn_implementation="flash_attention_2")
    except:
        model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.to(device)
    
    print("Loading MMLU dataset...")
    mmlu_dataset = load_dataset('cais/mmlu', 'all')['test']
    
    print("Formatting MMLU examples...")
    formatted_examples = [format_mmlu_example(ex) for ex in mmlu_dataset]
    
    print("Evaluating MMLU...")
    accuracy, _, _ = evaluate_mmlu(model, tokenizer, formatted_examples)
    print(f"MMLU Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()