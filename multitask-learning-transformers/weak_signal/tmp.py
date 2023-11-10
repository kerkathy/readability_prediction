# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import numpy as np
# %%

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

# %%
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# %%
from transformers import AutoTokenizer
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.modeling_bert import BertForTokenClassification, BertForSequenceClassification

x = ["hello! it's me speaking", "yes, how may I help you?"]

# Compare two bert models with and without pooling layer, both without downstream tasks
official_bert_model = BertModel.from_pretrained("bert-base-uncased")
official_bert_model_no_pooling = BertModel.from_pretrained("bert-base-uncased", add_pooling_layer=False)
print(official_bert_model.config)
print(official_bert_model_no_pooling.config)
print(official_bert_model)
print(official_bert_model_no_pooling)

with torch.no_grad():
    output_bert = official_bert_model(**encoding)
    output_bert_no_pooling = official_bert_model_no_pooling(**encoding)

print(output_bert.last_hidden_state.shape)
print(output_bert.pooler_output.shape)
print(output_bert_no_pooling.last_hidden_state.shape)

# Compare two bert models of different downstream tasks
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
encoding = tokenizer(x, return_tensors="pt", padding=True, truncation=True)

official_token_cls_model = BertForTokenClassification.from_pretrained("bert-base-uncased")
official_seq_cls_model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
print(official_token_cls_model.config)
print(official_seq_cls_model.config)
print(official_token_cls_model)
print(official_seq_cls_model)

with torch.no_grad():
    output_token_cls = official_token_cls_model(**encoding)
    output_seq_cls = official_seq_cls_model(**encoding)

print(output_token_cls.logits.shape)
print(output_seq_cls.logits.shape)


# %%
from multitask_model import BertForMTPairwiseRanking

multitask_model = BertForMTPairwiseRanking.from_pretrained("bert-base-uncased")

# %%
from transformers import AutoTokenizer
# %%
x1 = ("hello! it's me speaking", "yes, how may I help you?")
x2 = ["yes, how may I help you?", "thanks! I will call you back later"]
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
encoding_list = [tokenizer(*x, return_tensors="pt", padding=True, truncation=True) for x in [x1, x2]]
for encoding in encoding_list:
    print(encoding['input_ids'].shape)
    print(tokenizer.decode(encoding['input_ids'][0]))
    print(encoding['attention_mask'])
    print(encoding['attention_mask'].shape)
    print(encoding['token_type_ids'])
    print(encoding['token_type_ids'].shape)

# %%
