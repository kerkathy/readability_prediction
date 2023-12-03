# %%
import os
# #os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,3"

# %%
# Standard library imports
import logging
import random
from datetime import datetime
from pathlib import Path
import json
import sys

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
logging.basicConfig(level=logging.INFO)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

class DebugTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # logging.debug(inputs)  # Add logging here
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

def write_args_to_file(model_args, data_args, training_args, output_config_file):
    # Convert each argument object into a dictionary
    dict_model_args = model_args.__dict__
    dict_data_args = data_args.__dict__
    dict_training_args = json.loads(training_args.to_json_string())

    # Merge these dictionaries into one
    merged_args = {**dict_model_args, **dict_data_args, **dict_training_args}

    # Convert the merged dictionary into a JSON string
    json_str_merged_args = json.dumps(merged_args, indent=2)

    # Write the JSON string into a file
    with open(output_config_file, 'w') as f:
        f.write(json_str_merged_args)

# %%
def main():
    # %%
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    # #====================#
    # Normal use
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        write_args_to_file(model_args, data_args, training_args, f"./args_{datetime.now().strftime('%m%d%H')}.json")

    

    # When debug, use this
    # model_args, data_args, training_args = parser.parse_json_file(json_file="args.json")
    # print("model_args:", model_args)
    # print("data_args:", data_args)
    # print("training_args:", training_args)
    # training_args.report_to = [] # when debug, don't log.
    # ====================#

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
    # only log if report to wandb
    if "wandb" in training_args.report_to:
        # add epoch, data size, batch size, do_train, do_eval
        tags = [f"Epoch: {training_args.num_train_epochs}", f"Train batch: {training_args.per_device_train_batch_size}"]
        tags.append("Data size: " + str(data_args.max_train_samples) if data_args.max_train_samples is not None else "Data size: all")
        if training_args.do_train:
            tags.append("train")
        if training_args.do_eval:
            tags.append("eval")
        
        run = wandb.init(project="readability", tags=tags)
        run.log_code('.')
        
        # log args.json to wandb
        if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
            json_file = os.path.abspath(sys.argv[1])
        else:
            json_file = f"./args_{datetime.now().strftime('%m%d%H')}.json"
        json_artifact = wandb.Artifact("args", type="args")
        json_artifact.add_file(json_file)
        run.log_artifact(json_artifact)

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
        return_dict=True, # If you are instantiating your model with a config, you need to pass this return_dict=True to the config create
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

    # if training_args.report_to == "wandb":
    if "wandb" in training_args.report_to:
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

        logging.debug(f"example_batch['input_ids'].shape = {example_batch['input_ids'].shape}")
        logging.debug(f"type(example_batch['input_ids']) = {type(example_batch['input_ids'])}")
        logging.debug(f"example_batch['attention_mask'].shape = {example_batch['attention_mask'].shape}")
        logging.debug(f"type(example_batch['attention_mask']) = {type(example_batch['attention_mask'])}")
        logging.debug(f"example_batch['labels'].shape = {example_batch['labels'].shape}")
        logging.debug(f"type(example_batch['labels']) = {type(example_batch['labels'])}")

        # TODO inter-signal joint ranking labels

        return example_batch

    # %%
    # Preprocess the datasets
    for split in raw_datasets.keys():
        logging.info(f"Processing {split} dataset...")
        raw_datasets[split], threshold = compute_threshold(raw_datasets[split]) # need to be computed before mapping to preprocess function
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
        assert set(train_dataset[0].keys()) == set(["input_ids", "attention_mask", "labels"]), f"Keys mismatch: {set(train_dataset[0].keys())}"

        # Log a few random samples from the training set for eyeball test:
        for index in random.sample(range(len(train_dataset)), 2):
            logger.debug(f"Sample {index} of the training set: {train_dataset[index]}.")

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        assert set(eval_dataset[0].keys()) == set(["input_ids", "attention_mask", "labels"]), f"Keys mismatch: {set(train_dataset[0].keys())}"

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # %%
    def compute_metrics(p: EvalPrediction):
        # p.prediction returns everything except loss from model.forward()
        """
        For each weak signal, compute its ranking accuracy.
        """
        logging.info("**Compute metrics**")

        metric = evaluate.load("accuracy")

        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions # (len(dataset), num_signals, 3)
        preds = np.argmax(preds, axis=2) # (len(dataset), num_signals)
        preds = preds - 1 # (len(dataset), num_signals)

        labels = p.label_ids # (len(dataset), 2, num_signals)
        labels = labels[:, 0, :] # (len(dataset), num_signals)

        # make sure preds and labels are aligned in shape
        assert preds.shape == labels.shape, f"preds.shape = {preds.shape}, labels.shape = {labels.shape}"

        preds, labels = preds.flatten(), labels.flatten()

        result = metric.compute(predictions=preds, references=labels)

        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    # %%
    trainer = DebugTrainer(
    # trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )


    # %%
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

        trainer.save_model(f"./model_{datetime.now().strftime('%m%d%H')}")  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # %%
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    # %%
    if training_args.do_predict:
        # TODO: 等第三階段好了再改
        logger.info("*** Predict ***")

        # p.prediction returns everything except loss from model.forward()
        """
        For each weak signal, compute its ranking accuracy.
        """
        logging.info("**Compute metrics**")

        p = trainer.predict(predict_dataset=predict_dataset)
        # p = p.remove_columns("label")

        metric = evaluate.load("accuracy")

        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions # (len(dataset), num_signals, 3)
        preds = np.argmax(preds, axis=2) # (len(dataset), num_signals)
        preds = preds - 1 # (len(dataset), num_signals)

        labels = p.label_ids # (len(dataset), 2, num_signals)
        labels = labels[:, 0, :] # (len(dataset), num_signals)

        # make sure preds and labels are aligned in shape
        assert preds.shape == labels.shape, f"preds.shape = {preds.shape}, labels.shape = {labels.shape}"

        preds, labels = preds.flatten(), labels.flatten()

        result = metric.compute(predictions=preds, references=labels)

        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()

        output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{task}.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                logger.info(f"***** Predict results {task} *****")
                writer.write("index\tprediction\n")
                for index, item in enumerate(preds):
                    # item = label_list[item]
                    writer.write(f"{index}\t{item}\n")


# %%
if __name__ == "__main__":
    main()



    # %%
    ##### Debugging #####
    # # select one batch from eval_dataset
    # batch = None
    # # Create DataLoader
    # eval_dataloader = trainer.get_eval_dataloader()

    # # Get a batch of data
    # for batch in eval_dataloader:
    #     # Convert lists in batch to tensors
    #     print(batch)
    #     print(type(batch))
    #     # list of tensors to tensor
    #     # batch_tensors = {key: torch.stack(value) for key, value in batch.items()}
    #     print("=======", type(batch['input_ids']))
    #     print("-------", len(batch['input_ids']))
    #     print("=======", type(batch['input_ids'][0]))
    #     print("-------", len(batch['input_ids'][0]))
    #     print("=======", type(batch['input_ids'][0][0]))
    #     print("=======", type(batch['input_ids'][0][0][0]))
        
    #     # input_ids is a list of list of tensors --> tensor
    #     # stacked_tensors = torch.stack([torch.stack(sublist) for sublist in batch['input_ids']])
    #     # print("stacked_tensors.shape:", stacked_tensors.shape)
    #     # print("stacked_tensors:", type(stacked_tensors))

    #     # batch_tensors = {
    #     #     'input_ids': torch.stack([torch.stack(sublist) for sublist in batch['input_ids']]),
    #     #     'attention_mask': torch.stack([torch.stack(sublist) for sublist in batch['attention_mask']]),
    #     #     'labels': torch.stack([torch.stack(sublist) for sublist in batch['labels']]),
    #     # }

    #     # print("type(batch_tensors['input_ids']):", type(batch_tensors['input_ids']))
    #     # print("type(batch_tensors['attention_mask']):", type(batch_tensors['attention_mask']))
    #     # print("type(batch_tensors['labels']):", type(batch_tensors['labels']))

    #     # Send batch to model for prediction
    #     prediction = model(**batch)
    #     break

    # print("type(logits):", type(prediction.logits))
    # print("logits.shape:", prediction.logits.shape) # torch.Size([10, 8, 3])
    # print("label.shape:", batch['labels'].shape) # torch.Size([8, 2, 10])