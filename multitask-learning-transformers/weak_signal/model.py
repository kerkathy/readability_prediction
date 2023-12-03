"""
Implementation borrowed from transformers package and extended to support pairwise ranking
https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_bert.py
"""
# %%
import torch
from torch import nn

from typing import Optional, Tuple, List
import logging

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
    logits: torch.FloatTensor = None
    # logits: Optional[List[torch.FloatTensor]] = None
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
        self.regressors = nn.ModuleList()
        for signal in self.signal_list:
            self.regressors.append(nn.Linear(config.hidden_size, 1))
        self.margin = nn.Parameter(torch.ones(len(self.signal_list))) # TODO: what should be the init value?

        self.post_init()

    def print_gpu_memory_usage(self):
        # show GPU memory usage in GB
        print("Memory allocated:", torch.cuda.memory_allocated() / 1024 ** 3)
        print("Memory cached:", torch.cuda.memory_cached() / 1024 ** 3)

    @add_start_docstrings_to_model_forward(
        BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=True,
        output_hidden_states=True,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # torch.cuda.empty_cache() # clear GPU cache

        # print("**** Forward() ****")
        # print("type(labels):", type(labels))
        # print("type(input_ids):", type(input_ids))
        # print("type(attention_mask):", type(attention_mask))
        # assert torch.all(torch.logical_or(torch.logical_or(labels.eq(1, labels == -1), labels.eq(0))), "Labels must be 1, -1, or 0"
        assert input_ids.size(1) == 2, "input_ids must have first dimension = 2 but now having shape: " + str(input_ids.shape)
        assert attention_mask.size(1) == 2, "input_ids must have first dimension = 2 but now having shape: " + str(attention_mask.shape)

        # Seperate input_ids, attention_mask for each sentence
        input_ids_1, input_ids_2 = input_ids.unbind(dim=1) # shape: [1, 2, max_length] -> [1, max_length]
        attention_mask_1, attention_mask_2 = attention_mask.unbind(dim=1) # shape: [1, 2, max_length] -> [1, max_length]

        inputs = [{"input_ids": ids.to(self.device), "attention_mask": mask.to(self.device)} for ids, mask in zip([input_ids_1, input_ids_2], [attention_mask_1, attention_mask_2])]
        
        outputs = []
        for input in inputs:
            output = self.bert(
                input_ids=input["input_ids"],
                attention_mask=input["attention_mask"],
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            outputs.append(output)

        pooled_outputs = [output[1].to(self.device) for output in outputs] # [1] is the pooled output

        pooled_outputs = [self.dropout(pooled_output) for pooled_output in pooled_outputs]
        pooled_outputs = [pool_output.to(self.device) for pool_output in pooled_outputs] # shape of each pool_output: [batch_size, hidden_size]
        
        num_signals = len(self.signal_list)
        batch_size = pooled_outputs[0].shape[0]  # Assuming all pooled_outputs have the same batch size

        difference = torch.zeros(num_signals, batch_size).to(self.device) # shape: (num_signals, batch_size)

        for idx, signal in enumerate(self.signal_list):
            logit_pair = [self.regressors[idx](pooled_output) for pooled_output in pooled_outputs] # shape of each element: [batch_size, 1]
            difference[idx] = (logit_pair[0] - logit_pair[1]).squeeze(-1)

        expanded_margin = self.margin.data.new(*difference.shape).copy_(self.margin.data.unsqueeze(1).expand_as(difference))
        assert difference.shape == expanded_margin.shape, f"Difference and margin must have the same shape, but got difference.shape = {difference.shape} and margin.shape = {self.margin.shape}"

        loss = None
        prob_pos = torch.sigmoid(difference - expanded_margin) # shape: (num_signals, batch_size)
        prob_neg = torch.sigmoid(-difference - expanded_margin)
        prob_neutral = 1 - prob_pos - prob_neg
        prob_pos, prob_neg, prob_neutral = prob_pos.to(self.device), prob_neg.to(self.device), prob_neutral.to(self.device)
        
        labels = labels.unbind(dim=1)[0].transpose(0, 1) # shape: [batch_size, 2, num_signals] -> [num_signals, batch_size] (remove because it's duplicate)
        labels = labels.to(self.device)
        assert labels.shape == difference.shape, f"Labels and difference must have the same shape, but got labels.shape = {labels.shape} and difference.shape = {difference.shape}"
        
        loss = -torch.log(prob_pos[labels == 1]).sum() - torch.log(prob_neg[labels == -1]).sum() - torch.log(prob_neutral[labels == 0]).sum()

        assert loss is not None, "Loss must be computed"

        logits = torch.stack([prob_neg, prob_neutral, prob_pos], dim=-1) # shape: (num_signals, batch_size, 3)
        assert logits is not None, "Logits must be computed"
        logits = logits.transpose(0, 1)
        assert logits.shape == (batch_size, num_signals, 3), f"Logits must have shape (batch_size, num_signals, 3), but got logits.shape = {logits.shape}"

        hidden_states = [output.hidden_states for output in outputs]
        assert hidden_states is not None, "Hidden states must be computed"

        attentions = [output.attentions for output in outputs]
        assert attentions is not None, "Attentions must be computed"
        
        self.print_gpu_memory_usage()


        return MTPairwiseRankingModelOutput(
            loss=loss,
            logits=logits, # shape: (batch_size, num_signals, 3)
            hidden_states=hidden_states,
            attentions=attentions,
        )
# %%