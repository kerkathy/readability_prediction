# %%
import logging
import torch
import nltk
import nlp
import numpy as np
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from tqdm import tqdm as tqdm1
import transformers
from accelerate import Accelerator
from filelock import FileLock
from transformers import set_seed
from transformers.file_utils import is_offline_mode
from utils.arguments import parse_args
from multitask_model import BertForSequenceClassification
from preprocess import convert_to_features
from multitask_data_collator import MultitaskTrainer, NLPDataCollator
# from multitask_eval import multitask_eval_fn
from checkpoint_model import save_model
from pathlib import Path

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
    max_length = 512

    # %%

    model_name_or_path = "bert-base-uncased"
    per_device_train_batch_size = 4
    output_dir = "output"
    num_train_epochs = 1

    # %%

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    dataset_dict = {
        "quora_keyword_pairs": load_dataset(
            "csv", # modified, was "multitask_dataloader.py"
            data_files={ # modified, because hf csv reader doesn't like tsv
                # "train": "data/spaadia_squad_pairs/short_train.csv",
                # "validation": "data/spaadia_squad_pairs/short_val.csv",
                "train": "data/spaadia_squad_pairs/train.csv",
                "validation": "data/spaadia_squad_pairs/val.csv",
            },
        ),
        "spaadia_squad_pairs": load_dataset(
            "csv", # modified, was "multitask_dataloader.py"
            data_files={
                # "train": "data/spaadia_squad_pairs/short_train.csv",
                # "validation": "data/spaadia_squad_pairs/short_val.csv",
                "train": "data/spaadia_squad_pairs/train.csv",
                "validation": "data/spaadia_squad_pairs/val.csv",
            },
        ),
    }
    # %%

    for task_name, dataset in dataset_dict.items():
        print(task_name)
        print(dataset_dict[task_name]["train"][0])
        print()
    # %%

    model_names = [] * 2
    # model_names = [args.model_name_or_path] * 2
    model_names = [model_name_or_path] * 2
    config_files = model_names
    for idx, task_name in enumerate(["quora_keyword_pairs", "spaadia_squad_pairs"]):
        model_file = Path(f"./{task_name}_model/pytorch_model.bin")
        config_file = Path(f"./{task_name}_model/config.json")
        if model_file.is_file():
            model_names[idx] = f"./{task_name}_model"

        if config_file.is_file():
            config_files[idx] = f"./{task_name}_model"

    multitask_model = BertForSequenceClassification.from_pretrained(
        # args.model_name_or_path,
        model_name_or_path,
        task_labels_map={"quora_keyword_pairs": 2, "spaadia_squad_pairs": 2},
    )

    print(multitask_model.bert.embeddings.word_embeddings.weight.data_ptr())
    # %%

    def convert_to_features(
        example_batch, tokenizer, max_length=512
    ):
        inputs = list(example_batch["doc"])

        features = tokenizer(
            inputs,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt", 
        )
        features["label"] = example_batch["target"]
        return features
    
    convert_func_dict = {
        "quora_keyword_pairs": convert_to_features,
        "spaadia_squad_pairs": convert_to_features,
    }

    columns_dict = {
        "quora_keyword_pairs": ["input_ids", "attention_mask", "label"],
        "spaadia_squad_pairs": ["input_ids", "attention_mask", "label"],
    }

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
    # tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Tokenize all texts and align the labels with them.
    features_dict = {}

    for task_name, dataset in dataset_dict.items():
        features_dict[task_name] = {}
        for phase, phase_dataset in dataset.items():
            features_dict[task_name][phase] = phase_dataset.map(
                convert_func_dict[task_name],
                fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
                remove_columns=phase_dataset.column_names, # remove original columns, which are not used by the model
                batched=True,
                load_from_cache_file=False,
            )
            print(
                task_name,
                phase,
                len(phase_dataset),
                len(features_dict[task_name][phase]),
            )
            features_dict[task_name][phase].set_format(
                type="torch",
                columns=columns_dict[task_name],
            )
            print(
                task_name,
                phase,
                len(phase_dataset),
                len(features_dict[task_name][phase]),
            )
    # %%

    train_dataset = {
        task_name: dataset["train"] for task_name, dataset in features_dict.items()
    }
    # %%

    trainer = MultitaskTrainer(
        model=multitask_model,
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            # output_dir=args.output_dir,
            overwrite_output_dir=True,
            learning_rate=1e-5,
            do_train=True,
            num_train_epochs=num_train_epochs,
            # num_train_epochs=args.num_train_epochs,
            # Adjust batch size if this doesn't fit on the Colab GPU
            per_device_train_batch_size=per_device_train_batch_size,
            # per_device_train_batch_size=args.per_device_train_batch_size,
            save_steps=3000,
        ),
        data_collator=NLPDataCollator(),
        train_dataset=train_dataset,
    )
    trainer.args.local_rank = -1

    trainer.train()
    # %%

    eval_dataset = {
        task_name: dataset["validation"] for task_name, dataset in features_dict.items()
    }

    # %%
    # load from where we saved when debugging
    # multitask_model = BertForSequenceClassification.from_pretrained(
    #     "./model",
    #     task_labels_map={"quora_keyword_pairs": 2, "spaadia_squad_pairs": 2},
    # )

    def multitask_compute_metrics(multitask_model, tokenizer, eval_dataset, max_length=512, batch_size=8):
        print("modified!")
        preds_dict = {}

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        multitask_model = multitask_model.to(device)
        for task_name in ["quora_keyword_pairs", "spaadia_squad_pairs"]:
            val_len = len(eval_dataset[task_name])
            acc = 0.0
            for index in range(0, val_len):
                # TODO: 改成 batch

                # predict
                inputs = eval_dataset[task_name][index]["input_ids"].reshape(1, -1)  # dim 1 is batch size!!
                inputs = inputs.to(device)
                logits = multitask_model(inputs, task_name=task_name)[0]

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
    multitask_model.eval()
    multitask_compute_metrics(multitask_model, tokenizer, max_length=max_length, eval_dataset=eval_dataset, batch_size=8)

    # %%

    # save model
    trainer.save_model("./model_1107")

    # %%


if __name__ == "__main__":
    main()
