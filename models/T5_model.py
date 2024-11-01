from typing import List

import torch
import torch.nn as nn
from nltk import tokenize as text_tokenize
from transformers import AutoTokenizer, T5EncoderModel


def get_mlp(channels: List[int], add_batchnorm: bool = True) -> nn.Sequential:
    if add_batchnorm:
        return nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(channels[i - 1], channels[i]), nn.BatchNorm1d(channels[i]), nn.ReLU()
                )
                for i in range(1, len(channels))
            ]
        )
    else:
        return nn.Sequential(
            *[
                nn.Sequential(nn.Linear(channels[i - 1], channels[i]), nn.ReLU())
                for i in range(1, len(channels))
            ]
        )


def get_mlp2(channels: List[int], add_batchnorm: bool = True) -> nn.Sequential:
    if add_batchnorm:
        return nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(channels[i - 1], channels[i]), nn.BatchNorm1d(channels[i]), nn.ReLU()
                ) if i < len(channels) - 1
                else
                nn.Sequential(
                    nn.Linear(channels[i - 1], channels[i]), nn.BatchNorm1d(channels[i])
                )
                for i in range(1, len(channels))
            ]
        )
    else:
        return nn.Sequential(
            *[
                nn.Sequential(nn.Linear(channels[i - 1], channels[i]), nn.ReLU())
                if i < len(channels) - 1
                else nn.Sequential(nn.Linear(channels[i - 1], channels[i]))
                for i in range(1, len(channels))
            ]
        )


class LanguageEncoder(torch.nn.Module):
    def __init__(self, embedding_dim,  hungging_model = None, fixed_embedding=False,
                 intra_module_num_layers=2, intra_module_num_heads=4,
                 is_fine = False, inter_module_num_layers=2, inter_module_num_heads=4,
                 ):
        """Language encoder to encode a set of hints for each sentence"""
        super(LanguageEncoder, self).__init__()

        self.is_fine = is_fine
        self.tokenizer = AutoTokenizer.from_pretrained(hungging_model)
        T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        self.llm_model = T5EncoderModel.from_pretrained(hungging_model)
        if fixed_embedding:
            self.fixed_embedding = True
            for para in self.llm_model.parameters():
                para.require_grads = False
        else:
            self.fixed_embedding = False

        input_dim = self.llm_model.encoder.embed_tokens.weight.shape[-1]

        self.intra_module = nn.ModuleList(
            [nn.TransformerEncoderLayer(input_dim, intra_module_num_heads, dim_feedforward=input_dim * 4) for _ in
             range(intra_module_num_layers)])

        self.inter_mlp = get_mlp2([input_dim, embedding_dim], add_batchnorm=True)

        if not is_fine:
            self.inter_module = nn.ModuleList(
                [nn.TransformerEncoderLayer(embedding_dim, inter_module_num_heads, dim_feedforward=embedding_dim * 4)
                 for _ in range(inter_module_num_layers)])

    def forward(self, descriptions):

        split_union_sentences = []
        for description in descriptions:
            sentences = text_tokenize.sent_tokenize(description)

            if len(sentences) > 1:
                combined_sentence = " ".join(sentences)
            else:
                combined_sentence = sentences[0]

            split_union_sentences.append(combined_sentence)

        batch_size = len(descriptions)
        num_sentence = len(split_union_sentences) // batch_size

        inputs = self.tokenizer(split_union_sentences, return_tensors="pt", padding="longest")
        shorten_sentences_indices = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        shorten_sentences_indices = shorten_sentences_indices.to(self.device)
        attention_mask = attention_mask.to(self.device)
        out = self.llm_model(input_ids=shorten_sentences_indices,
                             attention_mask=attention_mask,
                             output_attentions=False)
        description_encodings = out.last_hidden_state

        if self.fixed_embedding:
            description_encodings = description_encodings.detach()

        description_encodings = description_encodings.permute(1, 0, 2)

        description_encodings = description_encodings.permute(1, 0, 2).contiguous()
        description_encodings = description_encodings.max(dim=1)[0]

        if description_encodings.size(0) == 1:
            self.inter_mlp.eval()
            with torch.no_grad():
                description_encodings = self.inter_mlp(description_encodings)
            self.inter_mlp.train()
        else:
            description_encodings = self.inter_mlp(description_encodings)

        for idx in range(len(self.inter_module)):
            description_encodings = description_encodings + self.inter_module[idx](description_encodings)

        return description_encodings

    @property
    def device(self):
        return next(self.inter_mlp.parameters()).device

