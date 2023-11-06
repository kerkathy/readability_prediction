# Load onestop_english dataset from huggingface datasets
# Extract signal from the dataset

# %%
import os
import sys
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
# %%
# Download dataset and save to disk with `my` prefix
dataset = load_dataset('onestop_english', split='train')
dataset.save_to_disk('my_onestop_english')
# %%


def main(args):
    # %%
    # Load dataset
    dataset = load_from_disk('my_onestop_english')
    # dataset = load_dataset('onestop_english', split='train')
    # %%

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # %%

    # Extract signal
    signal = []
    for i in tqdm(range(len(dataset))):
        # Get text
        text = dataset[i]['text']

        # %%
        # extract readability signal with textstat and readability library
        # Syllable Count
        # Lexicon Count
        # Flesch Reading Ease
        # Flesch-Kincaid Grade
        # SMOG Index
        # Gunning Fog Index
        # Automated Readability Index
        # Dale-Chall Readability Score
        # Average Sentences
        # Difficult Words

        syllable_count = textstat.syllable_count(text)
        lexicon_count = textstat.lexicon_count(text)
        flesch_reading_ease = textstat.flesch_reading_ease(text)
        flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
        smog_index = textstat.smog_index(text)
        gunning_fog = textstat.gunning_fog(text)
        automated_readability_index = textstat.automated_readability_index(text)
        dale_chall_readability_score = textstat.dale_chall_readability_score(text)
        # avg_sentences = textstat.avg_sentence_length(text)
        avg_sentences = textstat.sentence_count(text)
        difficult_words = textstat.difficult_words(text)
        # %%

    # Save signal
    signal = np.array(signal)
    np.save(os.path.join(args.output_dir, 'signal.npy'), signal)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--output_dir', type=str, default='./')
    args = parser.parse_args()
    main(args)