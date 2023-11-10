"""
Implementation borrowed from transformers package and extended to support multiple prediction heads:

https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_bert.py
"""
# %%
import torch
from torch import nn

from typing import Optional, Tuple, List

import transformers
from transformers import BertTokenizer
from transformers import models
from transformers.modeling_outputs import SequenceClassifierOutput

from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    BERT_INPUTS_DOCSTRING,
    # _TOKENIZER_FOR_DOC,
    _CHECKPOINT_FOR_DOC,
    _CONFIG_FOR_DOC,
    BertModel,
)

from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings_to_model_forward,
)

# define a custom output
from dataclasses import dataclass
from transformers.utils import ModelOutput

@dataclass
class MTPairwiseRankingModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[List[torch.FloatTensor]] = None
    attentions: Optional[List[torch.FloatTensor]] = None


class BertForMTPairwiseRanking(BertPreTrainedModel):
    """
    This class overrides the BertPreTrainedModel class from transformers package.
    It is a multi-task model that predicts pairwise ranking labels for a pair of sentences.
    """
    def __init__(self, config, **kwargs):
        super().__init__(transformers.PretrainedConfig())
        self.num_labels = kwargs.get("task_labels_map", {})
        self.signal_list = kwargs.get("signal_list", []) # list of weak signals
        self.config = config
        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.regressors = [] # regressor for each signal in signal_list
        for signal in self.signal_list:
            self.regressors.append(nn.Linear(config.hidden_size, 1).to(self.device))
        self.margin = nn.Parameter(torch.ones(len(self.signal_list))) # what should be the init value?

        self.post_init()

    @add_start_docstrings_to_model_forward(
        BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    # my modified version
    # @add_code_sample_docstrings(
    #     checkpoint=_CHECKPOINT_FOR_DOC,
    #     output_type=SequenceClassifierOutput,
    #     config_class=_CONFIG_FOR_DOC,
    # )    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        assert torch.all(torch.logical_or(torch.logical_or(labels == 1, labels == -1), labels == 0)), "Labels must be 1, -1, or 0"

        # separate input_ids, attention_mask, token_type_ids for each sentence based on [SEP]

        # TODO: modify the code below to support multiple sentences

        # Seperate input_ids, attention_mask, token_type_ids for each sentence
        # We have an original_batch
        # inputs = list(example_batch["text1"]) + list(example_batch["text2"])
        total_size = len(input_ids)
        input_ids_1 = input_ids[:total_size//2]
        input_ids_2 = input_ids[total_size//2:]
        attention_mask_1 = attention_mask[:total_size//2]
        attention_mask_2 = attention_mask[total_size//2:]
        labels = labels[:total_size//2]
        inputs = [{"input_ids": input_ids.to(self.device), "attention_mask": attention_mask.to(self.device)} for input_ids, attention_mask in zip([input_ids_1, input_ids_2], [attention_mask_1, attention_mask_2])]
        outputs = []
        for input in inputs:
            outputs.append(self.bert(**input))

        pooled_outputs = [output[1] for output in outputs] # [1] is the pooled output

        pooled_outputs = [self.dropout(pooled_output) for pooled_output in pooled_outputs]
        # list of predicted signals by regressor
        logits = []
        difference = []
        # prediction = torch.zeros(len(self.signal_list))
        for idx, signal in enumerate(self.signal_list):
            self.regressors[idx] = self.regressors[idx].to(self.device)
            logit_pair = [self.regressors[idx](pooled_output) for pooled_output in pooled_outputs]
            # don't know if the code below works... is it differentiable?
            difference.append(logit_pair[0] - logit_pair[1])
            logits.append(logit_pair)

            # prediction[idx] = torch.where(
            #     logits_1[idx] - logits_2[idx] > self.margin[idx],
            #     1,
            #     torch.where(
            #         logits_1[idx] - logits_2[idx] < -self.margin[idx]
            #         -1,
            #         0,
            #     ),
            # )

        # convert logits and difference to tensors
        logits = torch.tensor(logits).to(self.device)
        difference = torch.tensor(difference).to(self.device)

        # check shape
        assert difference.shape == self.margin.shape, f"Difference and margin must have the same shape, but got difference.shape = {difference.shape} and margin.shape = {self.margin.shape}"

        loss = None
        prob_pos = torch.sigmoid(difference - self.margin)
        prob_neg = torch.sigmoid(-difference - self.margin)
        prob_neutral = 1 - prob_pos - prob_neg
        print(f"prob_pos.shape = {prob_pos.shape}")
        print(f"labels.shape = {labels.shape}")
        print(f"prob_pos[labels == 1].shape = {prob_pos[labels == 1].shape}")
        print(f"torch.log(prob_pos[labels == 1]).shape = {torch.log(prob_pos[labels == 1]).shape}")
        print(f"torch.log(prob_pos[labels == 1]).sum().shape = {torch.log(prob_pos[labels == 1]).sum().shape}")
        loss = -torch.log(prob_pos[labels == 1]).sum() - torch.log(prob_neg[labels == -1]).sum() - torch.log(prob_neutral[labels == 0]).sum()

        return MTPairwiseRankingModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=[output.hidden_states for output in outputs],
            attentions=[output.attentions for output in outputs],
        )

# multitask_model = BertForMTPairwiseRanking.from_pretrained("bert-base-uncased")

# %%


