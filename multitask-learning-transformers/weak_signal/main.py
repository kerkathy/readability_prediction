# %%
import os
# #os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['WANDB_WATCH']="all"

# %%
# Standard library imports
import logging
import random
from datetime import datetime
from pathlib import Path

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
from transformers import (
    BertConfig,
    EvalPrediction,
    HfArgumentParser,
    Trainer, 
    TrainingArguments, 
    default_data_collator,
    set_seed, 
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.file_utils import is_offline_mode
from transformers.integrations import WandbCallback

# Local application imports
import evaluate
from checkpoint_model import save_model
from model import BertForMTPairwiseRanking
# from utils.arguments import parse_args
from utils.args import DataTrainingArguments, ModelArguments
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
    # args = parse_args()
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    set_seed(training_args.seed)
    
    data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

    if training_args.do_predict and data_args.test_file is not None:
        data_files["test"] = data_args.test_file
    # %%
    run = wandb.init(project="readability", name="Data30_epoch1_eval")
    run.log_code('.')

    # set default values for arguments
    # max_seq_length = 512 if data_args is None else min(data_args.max_seq_length, tokenizer.model_max_length)
    # model_name_or_path = "bert-base-uncased" if model_args is None else model_args.model_name_or_path
    # per_device_train_batch_size = 4 if training_args is None else training_args.per_device_train_batch_size
    # output_dir = "output" if args is None else args.output_dir
    # num_train_epochs = 1 if args is None else args.num_train_epochs
    # train_file = "data/ose/short_train_pair.csv" if args is None else data_args.train_file
    # validation_file = "data/ose/short_val_pair.csv" if args is None else data_args.validation_file

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

    task_labels_map = {"ose": 3}  # elementary, intermediate, advanced

    data_files = {"train": data_args.train_file, "validation": data_args.validation_file}
    if training_args.do_predict and data_args.test_file is not None:
        data_files["test"] = data_args.test_file

    raw_datasets = load_dataset(
        "csv",
        data_files=data_files,
    )

    # %%
    # Initialize tokenizer and model from pretrained checkpoint
    if data_args.task_name is not None:
        num_labels = task_labels_map[data_args.task_name]

    config = BertConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        token=model_args.token,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        token=model_args.token,
    )

    model = BertForMTPairwiseRanking.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        task_labels_map=task_labels_map,
        signal_list=signal_list,
        token=model_args.token,
    )
    wandb.watch(model, log="all")

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    if data_args.pad_to_max_length:
        padding = "max_length"

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
            padding=padding,
            max_length=max_seq_length // 2,
        )
        features_2 = tokenizer(
            input_2,
            truncation=True,
            padding=padding,
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
    # Preprocess the datasets
    
    for split in raw_datasets.keys():
        print(f"Processing {split} dataset...")
        raw_datasets[split], threshold = compute_threshold(raw_datasets[split]) # need to be computed before mapping to preprocess function
        # print(f"Removing columns: {raw_datasets[split].column_names}")
        raw_datasets[split] = raw_datasets[split].map(
            preprocess_function,
            remove_columns=raw_datasets[split].column_names, # remove original columns, which are not used by the model
            batched=True,
            load_from_cache_file=False,
            fn_kwargs={"thresholds": threshold},
        )

    # %%
    # Set up datasets for training, evaluation, and prediction
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # Log a few random samples from the training set for eyeball test:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")


    # %%
    # verify that the features are correct
    assert set(train_dataset[0].keys()) == set(["input_ids", "attention_mask", "labels"]), f"Keys mismatch: {set(train_dataset[0].keys())}"

    # %%
    def compute_metrics(p: EvalPrediction):
        # TODO
        """
        For each weak signal, compute its ranking accuracy.
        """
        print("**Compute metrics**")
        print(p)
        metric = evaluate.load("accuracy")
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        print("preds.shape:", preds.shape)
        print("p.predictions.shape:", p.predictions.shape)
        preds = np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

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
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # %%
    # save model with timestamp
    trainer.save_model(f"./model_{datetime.now().strftime('%m%d%H')}")

    # model_file = Path(f"./{task_name}_model/pytorch_model.bin")
    # config_file = Path(f"./{task_name}_model/config.json")

    # %%
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", combined if task is not None and "mnli" in task else metrics)

    if training_args.do_predict:
        # TODO
        logger.info("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]
        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            predict_dataset = predict_dataset.remove_columns("label")
            predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")


if __name__ == "__main__":
    main()