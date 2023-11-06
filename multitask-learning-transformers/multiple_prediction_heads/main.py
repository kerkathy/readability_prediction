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
from transformers import set_seed
from transformers.file_utils import is_offline_mode
from utils.arguments import parse_args
from multitask_model import BertForSequenceClassification
from preprocess import convert_to_features
from multitask_data_collator import MultitaskTrainer, NLPDataCollator
from multitask_eval import multitask_eval_fn
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


def main():
    args = parse_args()
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
    # %%
    # dataset_dict = {
    #     "quora_keyword_pairs": load_dataset(
    #         "multitask_dataloader.py",
    #         data_files={
    #             "train": "data/quora_keyword_pairs/short_train.tsv",
    #             # "train": "data/quora_keyword_pairs/train.tsv",
    #             "validation": "data/quora_keyword_pairs/short_dev.tsv",
    #             # "validation": "data/quora_keyword_pairs/dev.tsv",
    #         },
    #     ),
    #     "spaadia_squad_pairs": load_dataset(
    #         "multitask_dataloader.py",
    #         data_files={
    #             "train": "data/spaadia_squad_pairs/short_train.csv",
    #             # "train": "data/spaadia_squad_pairs/train.csv",
    #             "validation": "data/spaadia_squad_pairs/short_val.csv",
    #             # "validation": "data/spaadia_squad_pairs/val.csv",
    #         },
    #     ),
    # }
    dataset_dict = {
        "quora_keyword_pairs": load_dataset(
            "csv", # modified, was "multitask_dataloader.py"
            data_files={ # modified, because hf csv reader doesn't like tsv
                "train": "data/spaadia_squad_pairs/short_train.csv",
                "validation": "data/spaadia_squad_pairs/short_val.csv",
            },
        ),
        "spaadia_squad_pairs": load_dataset(
            "csv", # modified, was "multitask_dataloader.py"
            data_files={
                "train": "data/spaadia_squad_pairs/short_train.csv",
                "validation": "data/spaadia_squad_pairs/short_val.csv",
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
    # %%

    multitask_model = BertForSequenceClassification.from_pretrained(
        model_name_or_path,
        task_labels_map={"quora_keyword_pairs": 2, "spaadia_squad_pairs": 2},
    )

    print(multitask_model.bert.embeddings.word_embeddings.weight.data_ptr())
    # %%

    convert_func_dict = {
        "quora_keyword_pairs": convert_to_features,
        "spaadia_squad_pairs": convert_to_features,
    }

    columns_dict = {
        "quora_keyword_pairs": ["input_ids", "attention_mask", "labels"],
        "spaadia_squad_pairs": ["input_ids", "attention_mask", "labels"],
    }

    features_dict = {}
    for task_name, dataset in dataset_dict.items():
        features_dict[task_name] = {}
        for phase, phase_dataset in dataset.items():
            features_dict[task_name][phase] = phase_dataset.map(
                convert_func_dict[task_name],
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
            local_rank = -1,
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
    # %%
    trainer.train()

    # %%
    ## evaluate on given tasks
    # multitask_eval_fn(multitask_model, args.model_name_or_path, dataset_dict)
    multitask_eval_fn(multitask_model, model_name_or_path, dataset_dict)
    # %%

    # save model
    trainer.save_model("./model")

    # %%


if __name__ == "__main__":
    main()
