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

from  parlai.agents.transformer.modules    import TransformerDecoder

from .modules import BertTransformerModule
import os
import torch
from tqdm import tqdm
def add_bert_cmdline_args(parser):
    """Add command line arguments for this agent."""
    parser = parser.add_argument_group('Bert Encoder Arguments')
    parser.add_argument('--add-transformer-layer', type='bool', default=False,
                        help='Also add a transformer layer on top of Bert')
    parser.add_argument('--pull-from-layer', type=int, default=-1,
                        help='Which layer of Bert do we use? Default=-1=last one.')
    parser.add_argument('--out-dim', type=int, default=768,
                        help='For biencoder, output dimension')
    parser.add_argument('--topn', type=int, default=10,
                        help='For the biencoder: select how many elements to return')
    parser.add_argument('--data-parallel', type='bool', default=False,
                        help='use model in data parallel, requires '
                        'multiple gpus. NOTE This is incompatible'
                        ' with distributed training')
    parser.add_argument('--type-optimization', type=str,
                        default='all_encoder_layers',
                        choices=[
                            'additional_layers',
                            'top_layer',
                            'top4_layers',
                            'all_encoder_layers',
                            'all'],
                        help='Which part of the encoders do we optimize. '
                             '(Default: all_encoder_layers.)')
    parser.add_argument('--bert-aggregation', type=str,
                        default='first',
                        choices=[
                            'first',
                            'max',
                            'mean'],
                        help='How do we transform a list of output into one')
    parser.set_defaults(
        label_truncate=300,
        text_truncate=300,
        learningrate=0.00005,
        eval_candidates='inline',
        candidates='batch',
        dict_maxexs=0,  # skip building dictionary
    )


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

class BertTransformerGeneratorAgent(TorchGeneratorAgent):
    """ TorchRankerAgent implementation of the biencoder.
        It is a standalone Agent. It might be called by the Both Encoder.
    """

    
    
    """@staticmethod
    def add_cmdline_args(parser):
        TorchGeneratorAgent.add_cmdline_args(parser)
        add_transformer_cmdline_args(parser)
        add_common_args(parser)
        parser.add_argument_group('Bert Arguments')
        parser.set_defaults(
            encode_candidate_vecs=True
        )"""
    @classmethod
    def add_cmdline_args(cls, argparser):
        print('-----add_cmdline_args')
        """Add command-line arguments specifically for this agent."""
        add_bert_cmdline_args(argparser)
        agent = argparser.add_argument_group('Transformer Arguments')
        add_transformer_cmdline_args(agent)
        cls.dictionary_class().add_cmdline_args(argparser)
        super(BertTransformerGeneratorAgent, cls).add_cmdline_args(argparser)
        return agent
        
    @staticmethod
    def dictionary_class():
        print('-----dictionary_class')
        return BertDictionaryAgent

    def __init__(self, opt, shared=None):
        print('-----__init__')
        # download pretrained models
        download(opt['datapath'])
        self.pretrained_path = os.path.join(opt['datapath'], 'models',
                                            'bert_models', MODEL_PATH)
        opt['pretrained_path'] = self.pretrained_path
        super().__init__(opt, shared)
        self.clip = -1
        
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
        print('-----build_model')
        self.model = BertTransformerModule(self.opt,self.dict)
        if self.use_cuda:
            self.model.cuda()
        return self.model

    def forward(self, *xs, ys=None, cand_params=None, prev_enc=None, maxlen=None,
                bsz=None):
        print('-----forward')
        if ys is not None:
            # TODO: get rid of longest_label
            # keep track of longest label we've ever seen
            # we'll never produce longer ones than that during prediction
            self.longest_label = max(self.longest_label, ys.size(1))
        token_idx, segment_idx, mask = to_bert_input(*xs, self.NULL_IDX)
        # use cached encoding if available
        encoder_states = prev_enc if prev_enc is not None else self.encoder(token_idx, segment_idx, mask)

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

    def to_bert_input(token_idx, null_idx):
        """ token_idx is a 2D tensor int.
        return token_idx, segment_idx and mask
        """
        segment_idx = token_idx * 0
        mask = (token_idx != null_idx)
        # nullify elements in case self.NULL_IDX was not 0
        token_idx = token_idx * mask.long()
        return token_idx, segment_idx, mask

    @staticmethod
    def dictionary_class():
        return BertDictionaryAgent

    def init_optim(self, params, optim_states=None, saved_optim_type=None):
        self.optimizer = get_bert_optimizer([self.model],
                                            self.opt['type_optimization'],
                                            self.opt['learningrate'],
                                            fp16=self.opt.get('fp16'))
