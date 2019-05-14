from parlai.core.distributed_utils import is_distributed
from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.core.utils import padded_3d
from parlai.zoo.bert.build import download

from parlai.core.torch_generator_agent import TorchGeneratorAgent

from parlai.agents.bert_ranker.bert_dictionary import BertDictionaryAgent
from parlai.agents.bert_ranker.helpers import (get_bert_optimizer, BertWrapper, BertModel,
                      add_common_args, surround, MODEL_PATH)

import parlai.agents.transformer.modules

from  parlai.agents.transformer.modules    import TransformerDecoder
from  parlai.agents.transformer.modules    import TransformerGeneratorModel
import os
import torch
import inspect
from tqdm import tqdm


def _build_decoder(opt, dictionary, embedding=768, padding_idx=None,
                   n_positions=1024, n_segments=0):
        return TransformerDecoder(
        n_heads=opt['n_heads'],
        n_layers=opt['n_layers'],
        embedding_size=embeding,
        ffn_size=opt['ffn_size'],
        vocabulary_size=len(dictionary),
        embedding=embedding,
        dropout=opt['dropout'],
        attention_dropout=opt['attention_dropout'],
        relu_dropout=opt['relu_dropout'],
        padding_idx=padding_idx,
        learn_positional_embeddings=opt['learn_positional_embeddings'],
        embeddings_scale=opt['embeddings_scale'],
        n_positions=n_positions,
        activation=opt['activation'],
        variant=opt['variant'],
        n_segments=n_segments,
    )


class BertTransformerModule(TransformerGeneratorModel):
    def reorder_encoder_states(self, encoder_states, indices):
        # no support for beam search at this time
        return None
    
    def __init__(self, opt, dictionary):
        self.pad_idx = dictionary.pad_idx
        self.start_idx = dictionary.start_idx
        self.end_idx = dictionary.end_idx
        self.dictionary = dictionary
        print('super type:')
        print(super())
        print(inspect.getargspec(super().__init__))
        super().__init__(opt, dictionary)
        self.encoder = BertWrapper(
            BertModel.from_pretrained(opt['pretrained_path']),
            opt['embedding_size'],
            add_transformer_layer=opt['add_transformer_layer'],
            layer_pulled=opt['pull_from_layer'],
            aggregation=opt['bert_aggregation']
        )
        def reorder_encoder_states(self, encoder_states, indices):
        # no support for beam search at this time
            return None
    