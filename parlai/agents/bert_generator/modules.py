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

def to_bert_input(token_idx, null_idx):
        """ token_idx is a 2D tensor int.
        return token_idx, segment_idx and mask
        """
        segment_idx = token_idx * 0
        mask = (token_idx != null_idx)
        # nullify elements in case self.NULL_IDX was not 0
        token_idx = token_idx * mask.long()
        return token_idx, segment_idx, mask


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
        
    def forward(self, *xs, ys=None, cand_params=None, prev_enc=None, maxlen=None,
                    bsz=None):
            print('-----forward')
            print('*xs:')
            print(*xs[0])
            print(*xs[0].size())
            if ys is not None:
                # TODO: get rid of longest_label
                # keep track of longest label we've ever seen
                # we'll never produce longer ones than that during prediction
                self.longest_label = max(self.longest_label, ys.size(1))
            token_idx, segment_idx, mask = to_bert_input(*xs, self.NULL_IDX)
            print('token_idx:')
            print(token_idx)
            print(token_idx.size())
            # use cached encoding if available
            encoder_states = prev_enc if prev_enc is not None else self.encoder(token_idx.cuda(), segment_idx, mask),mask
            print('encoder_states:')
            print(encoder_states[0])
            print(encoder_states[0].size())
            if ys is not None:
                # use teacher forcing
                scores, preds = self.decode_forced(encoder_states, ys)
            else:
                scores, preds = self.decode_greedy(
                    encoder_states,
                    bsz,
                    maxlen or self.longest_label
                )

            return scores, preds, encoder_states

