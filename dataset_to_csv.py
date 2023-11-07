"""
from OneStopEnglishCorpus/Texts-Together-OneCSVperFile directory
split files in this directory into train:val:test = 3:1:1
and then write into train_pair.csv, val_pair.csv, test_pair.csv

Read the dataset containing pairs and convert it into csv format
columns: text1, label1, text2, label2
labels: 0 (ELE), 1 (INT), 2 (ADV)

output file format:
text1, label1, text2, label2
sentence1, 0, sentence2, 1
sentence3, 2, sentence4, 0
...

"""
# %%
from pathlib import Path
import csv
import random
import pandas as pd

source_dir = "OneStopEnglishCorpus/Texts-Together-OneCSVperFile"
# collect all file names
file_names = []
for file_path in Path(source_dir).rglob("*.csv"):
    file_names.append(file_path)
print(f"Number of files: {len(file_names)}")

# shuffle the file names
random.seed(42)
random.shuffle(file_names)

# split into train:val:test = 3:1:1
num_files = len(file_names)
num_train = num_files // 5 * 3

train_file_names = file_names[:num_train]
val_file_names = file_names[num_train : num_train + (num_files - num_train) // 2]
test_file_names = file_names[num_train + (num_files - num_train) // 2 :]
print(f"Number of train files: {len(train_file_names)}")
print(f"Number of val files: {len(val_file_names)}")
print(f"Number of test files: {len(test_file_names)}")
print(f"Total number of files: {len(train_file_names) + len(val_file_names) + len(test_file_names)}")

# %%
# read all files and write into train_pair.csv, val_pair.csv, test_pair.csv
combinations = ["ADV-ELE", "ADV-INT", "ELE-INT"]
levels_to_score = {"ELE": 0, "INT": 1, "ADV": 2}

def read_write(file_names, file_name):
    pairs = []
    for file_path in file_names:
        df = pd.read_csv(file_path, encoding='unicode_escape')
        for index, row in df.iterrows():
            for combination in combinations:
                levels = combination.split("-")
                scores = [levels_to_score[level] for level in levels]
                text1 = row[scores[0]]
                text2 = row[scores[1]]
                pairs.append([text1, scores[0], text2, scores[1]])


    with open(file_name, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["text1", "label1", "text2", "label2"])
        writer.writerows(pairs)

read_write(train_file_names, "train_pair.csv")
read_write(val_file_names, "val_pair.csv")
read_write(test_file_names, "test_pair.csv")
# %%
# Generate short file with only five data, for debugging
prefix = "short"

for file_name in ["train_pair.csv", "val_pair.csv", "test_pair.csv"]:
    df = pd.read_csv(file_name)
    df = df[:5]
    df.to_csv(f"{prefix}_{file_name}", index=False)
# %%
