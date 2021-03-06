#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.distributed_utils import is_distributed
from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.core.utils import padded_3d
from parlai.zoo.bert.build import download

from parlai.core.torch_generator_agent import TorchGeneratorAgent
from parlai.core.torch_generator_agent import TorchGeneratorModel

from parlai.agents.bert_ranker.bert_dictionary import BertDictionaryAgent
from parlai.agents.bert_ranker.helpers import (get_bert_optimizer, BertWrapper, BertModel,
                      add_common_args, surround, MODEL_PATH)

import parlai.agents.transformer.modules

from  parlai.agents.transformer.modules	import TransformerDecoder
import os
import torch
from tqdm import tqdm



def add_transformer_cmdline_args(parser):
        parser.add_argument('-esz', '--embedding-size', type=int, default=300,
                           help='Size of all embedding layers')
        parser.add_argument('-nl', '--n-layers', type=int, default=2)
        parser.add_argument('-hid', '--ffn-size', type=int, default=300,
                           help='Hidden size of the FFN layers')
        parser.add_argument('--dropout', type=float, default=0.0,
                           help='Dropout used in Vaswani 2017.')
        parser.add_argument('--attention-dropout', type=float, default=0.0,
                           help='Dropout used after attention softmax.')
        parser.add_argument('--relu-dropout', type=float, default=0.0,
                           help='Dropout used after ReLU. From tensor2tensor.')
        parser.add_argument('--n-heads', type=int, default=2,
                           help='Number of multihead attention heads')
        parser.add_argument('--learn-positional-embeddings', type='bool', default=False)
        parser.add_argument('--embeddings-scale', type='bool', default=True)
        parser.add_argument('--n-positions', type=int, default=None, hidden=True,
                           help='Number of positional embeddings to learn. Defaults '
                                'to truncate or 1024 if not provided.')
        parser.add_argument('--n-segments', type=int, default=0,
                           help='The number of segments that support the model. '
                                'If zero no segment and no langs_embedding.')
        parser.add_argument('--variant', choices={'aiayn', 'xlm'}, default='aiayn',
                           help='Chooses locations of layer norms, etc.')
        parser.add_argument('--activation', choices={'relu', 'gelu'}, default='relu',
                           help='Nonlinear activation to use. AIAYN uses relu, but '
                                'more recent papers prefer gelu.')

def _build_decoder(opt, dictionary, embedding=None, padding_idx=None,
                   n_positions=1024, n_segments=0):
        return TransformerDecoder(
        n_heads=opt['n_heads'],
        n_layers=opt['n_layers'],
        embedding_size=768,
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
class BertTrafoGeneratorAgent(TorchGeneratorAgent):
    """ TorchRankerAgent implementation of the biencoder.
        It is a standalone Agent. It might be called by the Both Encoder.
    """

    
    
    @staticmethod
    def add_cmdline_args(parser):
        TorchGeneratorAgent.add_cmdline_args(parser)
        add_transformer_cmdline_args(parser)
        add_common_args(parser)
        parser.add_argument_group('Transformer Arguments')
        parser.set_defaults(
            encode_candidate_vecs=True
        )

    def __init__(self, opt, shared=None):
        # download pretrained models
        download(opt['datapath'])
        self.pretrained_path = os.path.join(opt['datapath'], 'models',
                                            'bert_models', MODEL_PATH)
        opt['pretrained_path'] = self.pretrained_path

        self.clip = -1

        super().__init__(opt, shared)
        # it's easier for now to use DataParallel when
        self.data_parallel = opt.get('data_parallel') and self.use_cuda
        if self.data_parallel:
            self.model = torch.nn.DataParallel(self.model)
        if is_distributed():
            raise ValueError('Cannot combine --data-parallel and distributed mode')
        self.NULL_IDX = self.dict.pad_idx
        self.START_IDX = self.dict.start_idx
        self.END_IDX = self.dict.end_idx
        # default one does not average
        self.rank_loss = torch.nn.CrossEntropyLoss(reduce=True, size_average=True)

    def build_model(self):
        self.model = BertTransformerModule(self.opt,self.dict)

    @staticmethod
    def dictionary_class():
        return BertDictionaryAgent

    def init_optim(self, params, optim_states=None, saved_optim_type=None):
        self.optimizer = get_bert_optimizer([self.model],
                                            self.opt['type_optimization'],
                                            self.opt['learningrate'],
                                            fp16=self.opt.get('fp16'))

    
"""
    def share(self):
        "Share model parameters."
        shared = super().share()
        return shared
        
        """
class BertTransformerModule(TorchGeneratorModel):
    """ Groups context_encoder and transformer_encoder together.
    """
    
    
    
    
    def __init__(self, opt, dictionary):
        self.pad_idx = dictionary.pad_idx
        self.start_idx = dictionary.start_idx
        self.end_idx = dictionary.end_idx
        self.dictionary = dictionary
        self.embeddings=None
        super().__init__(self.pad_idx, self.start_idx, self.end_idx)
        if opt.get('n_positions'):
            # if the number of positions is explicitly provided, use that
            n_positions = opt['n_positions']
        else:
            # else, use the worst case from truncate
            n_positions = max(
                opt.get('truncate') or 0,
                opt.get('text_truncate') or 0,
                opt.get('label_truncate') or 0
            )
            if n_positions == 0:
                # default to 1024
                n_positions = 1024
        n_segments = opt.get('n_segments', 0)

        if n_positions < 0:
            raise ValueError('n_positions must be positive')
        
        self.encoder = BertWrapper(
            BertModel.from_pretrained(opt['pretrained_path']),
            opt['out_dim'],
            add_transformer_layer=opt['add_transformer_layer'],
            layer_pulled=opt['pull_from_layer'],
            aggregation=opt['bert_aggregation']
        )
        
        self.decoder = _build_decoder( opt, self.dictionary, self.embeddings, self.pad_idx,
            n_positions=n_positions,)
    
    
    
    def reorder_encoder_states(self, encoder_states, indices):
        # no support for beam search at this time
        return None

    def reorder_decoder_incremental_state(self, incremental_state, inds):
        # no support for incremental decoding at this time
        return None

    def output(self, tensor):
        # project back to vocabulary
        output = F.linear(tensor, self.embeddings.weight)
        return output


def to_bert_input(token_idx, null_idx):
    """ token_idx is a 2D tensor int.
        return token_idx, segment_idx and mask
    """
    segment_idx = token_idx * 0
    mask = (token_idx != null_idx)
    # nullify elements in case self.NULL_IDX was not 0
    token_idx = token_idx * mask.long()
    return token_idx, segment_idx, mask

