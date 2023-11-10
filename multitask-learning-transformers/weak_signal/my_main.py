# %%
import logging
import torch
import nltk
import numpy as np
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from tqdm import tqdm as tqdm1
import transformers
from accelerate import Accelerator
from filelock import FileLock
from transformers import set_seed, Trainer
from transformers.file_utils import is_offline_mode
from utils.arguments import parse_args
from multitask_model import BertForMTPairwiseRanking
# from multitask_data_collator import MultitaskTrainer, NLPDataCollator
from checkpoint_model import save_model
from pathlib import Path
# %%

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)
# %%


# def main():

# %%
# args = parse_args()
# %%

max_length = 512
model_name_or_path = "bert-base-uncased"
per_device_train_batch_size = 4
output_dir = "output"
num_train_epochs = 1

signal_list = [
    "syllable_count",
    "lexicon_count",
    "flesch_reading_ease",
    "flesch_kincaid_grade",
    "smog_index",
    "gunning_fog",
    "automated_readability_index",
    "dale_chall_readability_score",
    "sentence_count", # avg_sentences
    "difficult_words",
]

task_labels_map = {"oneStopEnglish": 3}  # elementary, intermediate, advanced

# %%
raw_dataset = load_dataset(
    "csv",
    data_files={
        "train": "data/ose/short_train_pair.csv",
        "validation": "data/ose/short_val_pair.csv",
        # "train": "data/ose/train_pair.csv",
        # "validation": "data/ose/val_pair.csv",
    },
)
print(raw_dataset['train'][0])
print(raw_dataset['train'].features)

# %%
# Initialize tokenizer and model from pretrained checkpoint
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)

model = BertForMTPairwiseRanking.from_pretrained(
    # args.model_name_or_path,
    model_name_or_path,
    task_labels_map=task_labels_map,
    signal_list=signal_list,
)
# print(model.bert.embeddings.word_embeddings.weight.data_ptr())

# %%
def preprocess_function(example_batch):
    """
    Convert a batch of examples to features for a given model
    example_batch: a batch of examples from dataset
        - text1
        - text2
        - label1
        - label2
    features: 
        - input_ids (two sentences concatenated) (dim = [batch_size, 2, max_length])
        - attention_mask (dim = [batch_size, 2, max_length])
        - labels (weak signals relative ranking) (dim = [batch_size, len(signal_list)])
    """
    import textstat

    original_keys = ["input_ids", "attention_mask"]
    input1 = list(example_batch["text1"])
    input2 = list(example_batch["text2"])
    assert len(input1) == len(input2)

    # Compute relative order for signals for each item in the batch as labels
    # dim: (batch_size, len(signal_list))
    overall_labels = []
    for data_1, data_2 in zip(input1, input2):
        computed_signals_1 = [textstat.__dict__[signal](data_1) for signal in signal_list]
        computed_signals_2 = [textstat.__dict__[signal](data_2) for signal in signal_list]
        relative_orders = [int(x > y) for x, y in zip(computed_signals_1, computed_signals_2)]
        overall_labels.append(relative_orders)
    labels = torch.tensor(overall_labels)

    # Convert text to features
    # dim: (batch_size * 2, max(length of text1, length of text2))
    num_elements = [len(input1), len(input2)]
    inputs = list(example_batch["text1"]) + list(example_batch["text2"])
    features = tokenizer(
        inputs,
        truncation=True,
        padding="longest",
    )

    for key in original_keys:
        example_batch[key] = torch.tensor(features[key])
    
    # Make labels as the same shape as input_ids
    example_batch["labels"] = torch.cat([labels, labels], dim=0)

    assert example_batch["input_ids"].shape[0] == sum(num_elements), f"Shape mismatch: {example_batch['input_ids'].shape[0]} vs. {sum(num_elements)}"
    # print(example_batch["input_ids"].shape)
    # assert example_batch["attention_mask"].shape == (sum(num_elements), max_length), f"Shape mismatch: {example_batch['attention_mask'].shape}"
    # print(f"Shape of labels: {example_batch['labels'].shape}")

    return example_batch


    # Concatenate two sentences after convert them into tensors
    # dim: (batch_size, 2, max_length)
    # for key in original_keys:
    #     print(type(example_batch[f"{key}_1"]))
    #     print(example_batch[f"{key}_1"].shape)
    #     example_batch[key] = torch.stack(
    #         (example_batch[f"{key}_1"], example_batch[f"{key}_2"]), dim=1
    #     )
    #     del example_batch[f"{key}_1"]
    #     del example_batch[f"{key}_2"]
    #     # check if the shape is correct
    #     assert example_batch[key].shape == (
    #         len(inputs),
    #         2,
    #         max_length,
    #     ), f"Shape mismatch: {example_batch[key].shape}"

    # Get label, i.e., the relative ranking of the two sentences
    # dim: (batch_size, len(signal_list))
    # labels = []
    # for idx in range(len(signal_list)):
    #     labels.append(
    #         np.array(computed_signals_list[0][idx]) > np.array(computed_signals_list[1][idx])
    #     )
    # # convert to tensor
    # example_batch["labels"] = torch.tensor(labels).T
    # assert example_batch["labels"].shape == (len(inputs), len(signal_list))



    # for idx in [1, 2]:
    #     inputs = list(example_batch[f"text{idx}"])
    #     features = tokenizer(
    #         inputs,
    #         truncation=True,
    #         padding="longest",
    #     )
    #     for key in original_keys:
    #         example_batch[f"{key}_{idx}"] = torch.tensor(features[key])

    #     # Compute signals for each item in the batch as labels
    #     # dim: (batch_size, len(signal_list))
    #     computed_signals = [] # list of list, each inner list is the computed signals for a batch
    #     for signal in signal_list:
    #         computed_signals.append([textstat.__dict__[signal](text) for text in inputs])
    #     computed_signals_list.append(computed_signals)

# %%
column_names = raw_dataset['train'].column_names
encoded_dataset = raw_dataset.map(
    preprocess_function,
    remove_columns=column_names, # remove original columns, which are not used by the model
    batched=True,
    load_from_cache_file=False,
)
print(encoded_dataset['train'][0])
print(encoded_dataset['train'].features)

# encoded_dataset sample 10 examples
# encoded_dataset = encoded_dataset.select(range(10))

# %%
train_dataset = encoded_dataset["train"]
eval_dataset = encoded_dataset["validation"]
print(train_dataset[0])
print(eval_dataset[0])

# verify that the features are correct
assert set(train_dataset[0].keys()) == set(["input_ids", "attention_mask", "labels"]), f"Keys mismatch: {set(train_dataset[0].keys())}"
# assert set(train_dataset[0].keys()) == set(["input_ids_1", "input_ids_2", "attention_mask_1", "attention_mask_2", "labels"]), f"Keys mismatch: {set(train_dataset[0].keys())}"
# assert set(eval_dataset[0].keys()) == set(["input_ids_1", "input_ids_2", "attention_mask_1", "attention_mask_2", "labels"]), f"Keys mismatch: {set(eval_dataset[0].keys())}"

# %%
# trainer = MultitaskTrainer(
#     model=model,
#     args=transformers.TrainingArguments(
#         output_dir=output_dir,
#         # output_dir=args.output_dir,
#         overwrite_output_dir=True,
#         learning_rate=1e-5,
#         do_train=True,
#         num_train_epochs=num_train_epochs,
#         # num_train_epochs=args.num_train_epochs,
#         per_device_train_batch_size=per_device_train_batch_size,
#         # per_device_train_batch_size=args.per_device_train_batch_size,
#         save_steps=3000,
#     ),
#     # data_collator=NLPDataCollator(),
#     train_dataset=train_dataset,
# )
# trainer.args.local_rank = -1

# set CUDA device
trainer = Trainer(
    model=model,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        # output_dir=args.output_dir,
        overwrite_output_dir=True,
        learning_rate=1e-5,
        do_train=True,
        num_train_epochs=num_train_epochs,
        # num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        # per_device_train_batch_size=args.per_device_train_batch_size,
        save_steps=3000,
    ),
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    # eval_dataset=encoded_dataset["validation"],
    # coompute_metrics=multitask_compute_metrics,
)
# set CUDA device


# trainer.args.local_rank = -1
print(f"Trainer is using device: {trainer.args.device}")

# %%
trainer.train()

# %%
def multitask_compute_metrics(eval_dataset, batch_size=8):
    print("modified!")
    preds_dict = {}
    # TODO: 改成每個 signal 的 accuracy

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    for task_name in ["quora_keyword_pairs", "spaadia_squad_pairs"]:
        val_len = len(eval_dataset[task_name])
        acc = 0.0
        for index in range(0, val_len):
            # TODO: 改成 batch

            # forward pass
            inputs = eval_dataset[task_name][index]["input_ids"].reshape(1, -1)  # dim 1 is batch size!!
            inputs = inputs.to(device)
            logits = model(inputs, task_name=task_name)[0]

            # compute accuracy
            labels = eval_dataset[task_name][index]["label"]
            predictions = torch.argmax(
                torch.FloatTensor(torch.softmax(logits, dim=1).detach().cpu().tolist()),
                dim=1,
            )
            acc += sum(np.array(predictions) == np.array(labels))
        acc = acc / val_len
        print(f"Task name: {task_name} \t Accuracy: {acc}")
# %%
# evaluate on given tasks
model.eval()
multitask_compute_metrics(eval_dataset=eval_dataset, batch_size=8)
# %%

# save model with timestamp
trainer.save_model(f"./model{datetime.now().strftime('%m%d%H')}")

# model_file = Path(f"./{task_name}_model/pytorch_model.bin")
# config_file = Path(f"./{task_name}_model/config.json")

# %%


# if __name__ == "__main__":
#     main()
