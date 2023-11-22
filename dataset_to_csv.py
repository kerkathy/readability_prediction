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
# %%

# collect all file names
source_dir = "OneStopEnglishCorpus/Texts-Together-OneCSVperFile"
file_names = ["OneStopEnglishCorpus/Texts-Together-OneCSVperFile/Amazon.csv"]

# %%
file_names = []
for file_path in Path(source_dir).rglob("*.csv"):
    file_names.append(file_path)
print(f"Number of files: {len(file_names)}")

# %%
def detect_file_encoding(file_path):
    import chardet
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

# %%
# shuffle the file names
random.seed(42)
random.shuffle(file_names)

# split into train:val:test = 3:1:1
# so that some file names will be for train, some for val, some for test
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
encoding = detect_file_encoding(train_file_names[0])

def read_write(input_file_paths, output_file_name):
    pairs = []
    for file_path in input_file_paths:
        df = pd.read_csv(file_path, encoding=encoding)
        for index, row in df.iterrows():
            for combination in combinations:
                levels = combination.split("-")
                scores = [levels_to_score[level] for level in levels]
                # Check if the row has values for both levels
                if pd.notnull(row.iloc[scores[0]]) and pd.notnull(row.iloc[scores[1]]):
                    text1 = row.iloc[scores[0]]
                    text2 = row.iloc[scores[1]]
                    pairs.append([text1, scores[0], text2, scores[1]])

    with open(output_file_name, "w") as f: # don't specify encoding, use default
    # with open(output_file_name, "w", encoding=encoding) as f:
        writer = csv.writer(f)
        writer.writerow(["text1", "label1", "text2", "label2"])
        writer.writerows(pairs)
    print(f"File {output_file_name} generated")

    # print num of rows in the file and num of elements in pairs
    df = pd.read_csv(output_file_name)
    print(f"Number of rows in {output_file_name}: {len(df)}")
    print(f"Number of elements in pairs: {len(pairs)}")

read_write(train_file_names, "train_pair.csv")
read_write(val_file_names, "val_pair.csv")
read_write(test_file_names, "test_pair.csv")

# %%
# Generate short file with only 30 data, for debugging
prefix = "short"

for file_name in ["train_pair.csv", "val_pair.csv", "test_pair.csv"]:
    df = pd.read_csv(file_name)
    df = df[:30]
    df.to_csv(f"{prefix}_{file_name}", index=False)
    print(f"File {prefix}_{file_name} generated")

# %%
# Check if there is any empty cells in the csv files
import pandas as pd

files = ["train_pair.csv", "val_pair.csv", "test_pair.csv"]
encoding = detect_file_encoding(files[0])

for file in files:
    df = pd.read_csv(file)
    if df.isnull().values.any():
        print(f'{file} contains empty cells.')
        # Locate empty cells
        empty_cells = df[df.isnull().any(axis=1)]
        print(empty_cells)
    else:
        print(f'{file} does not contain any empty cells.')

# %%
# Add files and dataframes of those generated files as artifacts in wandb
import wandb
import pandas as pd

run = wandb.init(project='readability')

artifact = wandb.Artifact('train_pair', type='dataset')
train_pair_df = pd.read_csv('train_pair.csv')
train_pair_table = wandb.Table(dataframe=train_pair_df)
artifact.add(train_pair_table, 'train_pair_table')
artifact.add_file('train_pair.csv')
run.log_artifact(artifact)
run.log({"train_pair_table": train_pair_table})

artifact = wandb.Artifact('val_pair', type='dataset')
val_pair_df = pd.read_csv('val_pair.csv')
val_pair_table = wandb.Table(dataframe=val_pair_df)
artifact.add(val_pair_table, 'val_pair_table')
artifact.add_file('val_pair.csv')
run.log_artifact(artifact)
run.log({"val_pair_table": val_pair_table})

artifact = wandb.Artifact('test_pair', type='dataset')
test_pair_df = pd.read_csv('test_pair.csv')
test_pair_table = wandb.Table(dataframe=test_pair_df)
artifact.add(test_pair_table, 'test_pair_table')
artifact.add_file('test_pair.csv')
run.log_artifact(artifact)
run.log({"test_pair_table": test_pair_table})

run.finish()

# %%
import wandb
import pandas as pd

run = wandb.init(project='readability', name="generate_short_file")

artifact = wandb.Artifact('short_train_pair', type='dataset')
train_pair_df = pd.read_csv('short_train_pair.csv')
train_pair_table = wandb.Table(dataframe=train_pair_df)
artifact.add(train_pair_table, 'short_train_pair_table')
artifact.add_file('short_train_pair.csv')
run.log_artifact(artifact)
run.log({"short_train_pair_table": train_pair_table})

artifact = wandb.Artifact('short_val_pair', type='dataset')
val_pair_df = pd.read_csv('short_val_pair.csv')
val_pair_table = wandb.Table(dataframe=val_pair_df)
artifact.add(val_pair_table, 'short_val_pair_table')
artifact.add_file('short_val_pair.csv')
run.log_artifact(artifact)
run.log({"short_val_pair_table": val_pair_table})

artifact = wandb.Artifact('short_test_pair', type='dataset')
test_pair_df = pd.read_csv('short_test_pair.csv')
test_pair_table = wandb.Table(dataframe=test_pair_df)
artifact.add(test_pair_table, 'short_test_pair_table')
artifact.add_file('short_test_pair.csv')
run.log_artifact(artifact)
run.log({"short_test_pair_table": test_pair_table})

run.finish()


# %%
# debug
# see if the ['Advanced'] col of first row in Amazon.csv is the same as col ['text1'] in the first row in train_pair.csv

# import pandas as pd

# # Load the data
# amazon_df = pd.read_csv('OneStopEnglishCorpus/Texts-Together-OneCSVperFile/Amazon.csv', encoding=encoding)
# train_pair_df = pd.read_csv('train_pair.csv', encoding=encoding)

# # Get the first row of the specific columns
# elementary_value = amazon_df['Advanced'].iloc[0]
# text1_value = train_pair_df['text1'].iloc[0]
# lable1_value = train_pair_df['label1'].iloc[0]

# # Compare the values
# is_same = elementary_value == text1_value
# print(f"label1: {lable1_value}")
# print(is_same)
# # %%
# # Iterate over the characters
# for i, (char1, char2) in enumerate(zip(elementary_value, text1_value)):
#     # If the characters are not the same
#     if char1 != char2:
#         print(f"Difference at index {i}: '{char1}' vs '{char2}'")

# # %%
# # check if the length are the same
# print(len(elementary_value))
# print(len(text1_value))

# %%
# To make sure the data read/write is working correctly
# import pandas as pd
# import csv


# def copy_file(source_file_path, destination_file_path):
#     # Read the source file using pandas
#     df = pd.read_csv(source_file_path, encoding=encoding)

#     # Write the content to the destination file using csv.writer
#     with open(destination_file_path, 'w', newline='', encoding=encoding) as destination_file:
#         writer = csv.writer(destination_file)
#         writer.writerow(df.columns)
#         for index, row in df.iterrows():
#             writer.writerow(row)

# source_file_path = 'OneStopEnglishCorpus/Texts-Together-OneCSVperFile/WNL What\'s the secret.csv'
# destination_file_path = 'debug_copy_of_train_file.csv'

# # Use the function
# copy_file(source_file_path, destination_file_path)

# # Load the data
# df1 = pd.read_csv(destination_file_path, encoding=encoding)
# df2 = pd.read_csv(source_file_path, encoding=encoding)

# # Check if the two dataframes are equal
# is_same = df1.equals(df2)

# print(f"[{is_same}] File {source_file_path} and {destination_file_path} are the same")
# # %%
