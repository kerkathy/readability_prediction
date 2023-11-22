# # %%
import os
# #os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# use only 0 and 3 GPUs
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['WANDB_WATCH']="all"

# %%
# Standard library imports
from datetime import datetime
from pathlib import Path
import logging

# Third party imports
import nltk
import numpy as np
import pandas as pd
import textstat
import torch
import transformers
import wandb
from datasets import load_dataset, DatasetDict
from filelock import FileLock
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm as tqdm1
from tqdm.auto import tqdm
from transformers import set_seed, Trainer, TrainingArguments, EvalPrediction
from transformers.file_utils import is_offline_mode

# Local application imports
import evaluate
from checkpoint_model import save_model
from model import BertForMTPairwiseRanking
from utils.arguments import parse_args
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
def main():
    args = parse_args()

    # %%
    # debug
    # args = None
    # %%
    wandb.init(project="readability", name="AllData_epoch10_cuda0")

    # set default values for arguments
    max_seq_length = 512 if args is None else args.max_seq_length
    model_name_or_path = "bert-base-uncased" if args is None else args.model_name_or_path
    per_device_train_batch_size = 4 if args is None else args.per_device_train_batch_size
    output_dir = "output" if args is None else args.output_dir
    num_train_epochs = 1 if args is None else args.num_train_epochs
    train_file = "data/ose/short_train_pair.csv" if args is None else args.train_file
    validation_file = "data/ose/short_val_pair.csv" if args is None else args.validation_file

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
    # Initialize tokenizer and model from pretrained checkpoint
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)

    model = BertForMTPairwiseRanking.from_pretrained(
        model_name_or_path,
        task_labels_map=task_labels_map,
        signal_list=signal_list,
    )

    # %%
    def compute_threshold(dataset):
        """
        1st pass: 
        Compute the threshold for each signal based on the differences between two texts
        Also store the signal values in the dataset
        """
        num_signals = len(signal_list)
        all_diffs = []

        input_1 = dataset["text1"]
        input_2 = dataset["text2"]
        assert len(input_1) == len(input_2), f"len(input_1) != len(input_2): {len(input_1)} != {len(input_2)}"
        num_examples = len(input_1)
        computed_signals_1 = np.zeros((num_examples, num_signals))
        computed_signals_2 = np.zeros((num_examples, num_signals))

        for i, signal in enumerate(signal_list):
            computed_signals_1[:, i] = [textstat.__dict__[signal](data_1) for data_1 in input_1]
            computed_signals_2[:, i] = [textstat.__dict__[signal](data_2) for data_2 in input_2]

        diffs = np.abs(computed_signals_1 - computed_signals_2) # dim: [num_examples, num_signals]
        all_diffs.append(diffs)
        for i, signal in enumerate(signal_list):
            # dataset = dataset.add_column(signal, list(zip(computed_signals_1[:, i], computed_signals_2[:, i])))
            dataset = dataset.add_column(f"{signal}_text1", computed_signals_1[:, i])
            dataset = dataset.add_column(f"{signal}_text2", computed_signals_2[:, i])
        all_diffs = np.concatenate(all_diffs, axis=0)
        thresholds = {signal: np.percentile(all_diffs[:, i], 40) for i, signal in enumerate(signal_list)}
        return dataset, thresholds

    def preprocess_function(example_batch, thresholds):
        """
        2nd pass:
        Convert a batch of examples to features for a given model
        example_batch: a batch of examples from dataset
            - text1 (str)
            - text2 (str)
            - label1 (int)
            - label2 (int)
            - signal values (tuple of floats)
        features: 
            - input_ids (two sentences concatenated) (dim = [batch_size, 2, max_length])
            - attention_mask (dim = [batch_size, 2, max_length])
            - labels (weak signals relative ranking) (dim = [batch_size, 2, len(signal_list)])
        """
        input_1 = list(example_batch["text1"])
        input_2 = list(example_batch["text2"])
        assert len(input_1) == len(input_2)
        batch_size = len(input_1)

        # check if each text is a string
        for idx, (data_1, data_2) in enumerate(zip(input_1, input_2)):
            assert isinstance(data_1, str), f"data_1 from the {idx}th example is not a string: {data_1}"
            assert isinstance(data_2, str), f"data_2 from the {idx}th example is not a string: {data_2}"

        # Compute relative order for signals for each item in the batch as labels
        overall_labels = np.zeros((batch_size, len(signal_list)))
        for i, signal in enumerate(signal_list):
            # x, y = np.array(example_batch[signal]).T
            x1 = np.array(example_batch[f"{signal}_text1"])
            x2 = np.array(example_batch[f"{signal}_text2"])
            overall_labels[(x1 > x2 + thresholds[signal]), i] = 1
            overall_labels[(x1 < x2 - thresholds[signal]), i] = -1

        labels = torch.tensor(overall_labels) # labels.shape = (batch_size, len(signal_list))

        # Convert text to features, they should be truncate separately
        features_1 = tokenizer(
            input_1,
            truncation=True,
            padding="max_length",
            max_length=max_seq_length // 2,
        )
        features_2 = tokenizer(
            input_2,
            truncation=True,
            padding="max_length",
            max_length=max_seq_length // 2,
        )

        original_keys = ["input_ids", "attention_mask"]
        for key in original_keys:
            # combine features_1 and features_2
            tensor_1 = torch.tensor(features_1[key])
            tensor_2 = torch.tensor(features_2[key])
            example_batch[key] = torch.stack((tensor_1, tensor_2), dim=1) # example_batch[key].shape = [batch_size, 2, max_length]
        
        example_batch["labels"] = torch.stack((labels, labels), dim=1) # example_batch.shape = [batch_size, 2, len(signal_list)]

        print(f"example_batch['input_ids'].shape = {example_batch['input_ids'].shape}")
        print(f"example_batch['attention_mask'].shape = {example_batch['attention_mask'].shape}")
        print(f"example_batch['labels'].shape = {example_batch['labels'].shape}")

        # TODO inter-signal joint ranking labels

        return example_batch

    # %%
    raw_dataset = load_dataset(
        "csv",
        data_files={
            "train": train_file,
            "validation": validation_file,
        },
    )
    assert len(raw_dataset['train']['text1']) == len(raw_dataset['train']['text2']), f"len(text1) != len(text2): {len(raw_dataset['train']['text1'])} != {len(raw_dataset['train']['text2'])}"

    # %%
    encoded_dataset = DatasetDict()
    
    for split in ["train", "validation"]:
        print(f"Processing {split} dataset...")
        raw_dataset[split], threshold = compute_threshold(raw_dataset[split]) # need to be computed before mapping to preprocess function
        print(f"Removing columns: {raw_dataset[split].column_names}")
        encoded_dataset[split] = raw_dataset[split].map(
            preprocess_function,
            remove_columns=raw_dataset[split].column_names, # remove original columns, which are not used by the model
            batched=True,
            load_from_cache_file=False,
            fn_kwargs={"thresholds": threshold},
        )

    # encoded_dataset sample 10 examples
    # encoded_dataset = encoded_dataset.select(range(10))

    # %%
    train_dataset = encoded_dataset["train"]
    eval_dataset = encoded_dataset["validation"]

    # verify that the features are correct
    assert set(train_dataset[0].keys()) == set(["input_ids", "attention_mask", "labels"]), f"Keys mismatch: {set(train_dataset[0].keys())}"

    # %%
    # def compute_metrics(p: EvalPrediction):
    #     # TODO
    #     """
    #     For each weak signal, compute its ranking accuracy.
    #     """
    #     preds, labels = p.predictions, p.label_ids
    #     print(f"preds.shape = {preds.shape}")
    #     print(f"labels.shape = {labels.shape}")


    #     # preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    #     preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
    #     metric = evaluate.load("accuracy")
    #     # for signal in signal_list:

    #     # prediction[idx] = torch.where(
    #     #     logits_1[idx] - logits_2[idx] > self.margin[idx],
    #     #     1,
    #     #     torch.where(
    #     #         logits_1[idx] - logits_2[idx] < -self.margin[idx]
    #     #         -1,
    #     #         0,
    #     #     ),
    #     # )
    #     result = metric.compute(predictions=preds, references=labels)

    #     if len(result) > 1:
    #         result["combined_score"] = np.mean(list(result.values())).item()
    #     return result

    # %%
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            learning_rate=1e-5,
            do_train=True,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            # save_steps=3000,
            # evaluation_strategy="epoch",
            # save_strategy="epoch",
            # logging_strategy="epoch",
            logging_strategy="steps",
            logging_steps=100,
            report_to="wandb",
        ),
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        # eval_dataset=eval_dataset,
        # compute_metrics=compute_metrics,
    )

    # trainer.args.local_rank = -1
    print(f"Trainer is using device: {trainer.args.device}")

    trainer.train()

    # %%
    # evaluate on given tasks
    # model.eval()
    # compute_metrics(eval_dataset=eval_dataset, batch_size=8)
    # %%

    # save model with timestamp
    trainer.save_model(f"./model_{datetime.now().strftime('%m%d%H')}")

    # model_file = Path(f"./{task_name}_model/pytorch_model.bin")
    # config_file = Path(f"./{task_name}_model/config.json")

    # %%

if __name__ == "__main__":
    main()