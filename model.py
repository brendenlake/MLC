import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from copy import deepcopy
import matplotlib.pyplot as plt
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def describe_model(net):
    nparams = sum(p.numel() for p in net.parameters() if p.requires_grad)
    if type(net) is BIML:
        print('\nBIML specs:')
        print(' nparams=',nparams)
        print(' nlayers_encoder=',net.nlayers_encoder)
        print(' nlayers_decoder=',net.nlayers_decoder)
        print(' nhead=',net.nhead)
        print(' hidden_size=',net.hidden_size)
        print(' dim_feedforward=',net.dim_feedforward)
        print(' act_feedforward=',net.act)
        print(' dropout=',net.dropout_p)
        print(' ')
        print('')
    else:
        print('Network type ' + str(type(net)) + ' not found...')

class PositionalEncoding(nn.Module):
    #
    # Adds positional encoding to the token embeddings to introduce word order
    #
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000.) / emb_size) # size emb_size/2
        pos = torch.arange(0, maxlen).reshape(maxlen, 1) # maxlen x 1
        pos_embedding = torch.zeros((maxlen, emb_size)) # maxlen x emb_size
        pos_embedding[:, 0::2] = torch.sin(pos * den) # maxlen x emb_size/2
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2) # maxlen x 1 x emb_size
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        #  Input
        #    token_embedding: [seq_len, batch_size, embedding_dim] list of embedded tokens
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class BIML(nn.Module):
    #
    # Transformer trained for meta seq2seq learning
    #
    def __init__(self, hidden_size: int, input_size: int, output_size: int, PAD_idx_input: int, PAD_idx_output: int,
        nlayers_encoder: int=5, nlayers_decoder: int=3, nhead: int=8,
        dropout_p: float=0.1, ff_mult: int=4, activation='gelu'):
        #
        # Input        
        #  hidden_size : embedding size
        #  input_size  : number of input symbols
        #  output_size : number of output symbols
        #  PAD_idx_input : index of padding in input sequences
        #  PAD_idx_output : index of padding in output sequences
        #  nlayers_encoder : number of transformer encoder layers
        #  nlayers_decoder : number of transformer decoder layers (likely fewer than encoder for tasks with deterministic outputs)
        #  nhead : number of heads for multi-head attention
        #  dropout_p : dropout applied to symbol embeddings and transformer layers
        #  ff_mult : multiplier for hidden size of feedforward network
        #  activation: string either 'gelu' or 'relu'
        #   
        super(BIML, self).__init__()
        assert activation in ['gelu','relu']        
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size        
        self.PAD_idx_input = PAD_idx_input
        self.PAD_idx_output = PAD_idx_output
        self.nlayers_encoder = nlayers_encoder
        self.nlayers_decoder = nlayers_decoder
        self.nhead = nhead
        self.dropout_p = dropout_p
        self.dim_feedforward = hidden_size*ff_mult
        self.act = activation
        self.transformer = nn.Transformer(d_model=hidden_size, nhead=nhead, num_encoder_layers=nlayers_encoder, num_decoder_layers=nlayers_decoder,
            dim_feedforward=self.dim_feedforward, dropout=dropout_p, batch_first=True, activation=activation)
        self.positional_encoding = PositionalEncoding(emb_size=hidden_size, dropout=dropout_p)
        self.input_embedding = nn.Embedding(input_size, hidden_size)
        self.output_embedding = nn.Embedding(output_size, hidden_size)
        self.out = nn.Linear(hidden_size,output_size)

    def expand_embeddings(self):
        # Add a new dimension/symbol to the embeddings
        old_input_embed = self.input_embedding.weight.data
        old_output_embed = self.output_embedding.weight.data
        self.input_embedding = nn.Embedding(self.input_size+1, self.hidden_size)
        self.output_embedding = nn.Embedding(self.output_size+1, self.hidden_size)
        self.input_embedding.weight.data[:-1,:] = old_input_embed
        self.output_embedding.weight.data[:-1,:] = old_output_embed

    def prep_encode(self, xq_context_padded):
        # Embed source sequences and make masks
        # 
        # Input
        #  xq_context_padded : input strings as indices # batch_size x maxlen_src
        xq_context_embed = self.input_embedding(xq_context_padded) # batch_size x maxlen_src x emb_size

        # Add positional encoding to input embeddings
        src_embed = self.positional_encoding(xq_context_embed.transpose(0,1))
        src_embed = src_embed.transpose(0,1) # batch_size x maxlen_src x emb_size

        # Create masks for padded source sequences
        src_padding_mask = xq_context_padded==self.PAD_idx_input # batch_size x  maxlen_src
            # value of True means ignore
        return src_embed, src_padding_mask

    def prep_decode(self, z_padded):
        # Embed target sequences and make masks 
        #
        # Input
        #  z_padded : b*nq (batch_size) x maxlen_tgt
        #  z_lengths : b*nq list
        maxlen_tgt = z_padded.size(1)
        z_embed = self.output_embedding(z_padded) # batch_size x maxlen_tgt x emb_size

        # Add positional encoding to target embeddings
        tgt_embed = self.positional_encoding(z_embed.transpose(0,1))
        tgt_embed = tgt_embed.transpose(0,1) # batch_size x maxlen_tgt x emb_size

        # create mask for padded targets
        tgt_padding_mask = z_padded==self.PAD_idx_output # batch_size x maxlen_tgt
            # value of True means ignore

        # create diagonal mask for autoregressive control
        tgt_mask = self.transformer.generate_square_subsequent_mask(maxlen_tgt) # maxlen_tgt x maxlen_tgt
        tgt_mask = tgt_mask.to(device=DEVICE)
        return tgt_embed, tgt_padding_mask, tgt_mask

    def forward(self, z_padded, batch):
        # Forward pass through encoder and decoder
        # 
        # Input
        #  z_padded : tensor of size [b*nq (batch_size), maxlen_target] : indices for decoder input
        #  batch : struct from datasets.make_biml_batch() : includes encoder inputs
        # 
        # Output
        #   output : [b*nq x maxlen_target x output_size]
        xq_context_padded = batch['xq_context_padded'] # batch_size x maxlen_src
        src_embed, src_padding_mask = self.prep_encode(xq_context_padded)
        tgt_embed, tgt_padding_mask, tgt_mask = self.prep_decode(z_padded)
        trans_out = self.transformer(src_embed, tgt_embed, tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask)        
        output = self.out(trans_out)
        return output

    def encode(self, batch):
        # Forward pass through encoder only
        #
        # Output
        #  memory : [b*nq (batch_size) x maxlen_src x hidden_size]
        #  memory_padding_mask : [b*nq (batch_size) x maxlen_src] binary mask
        xq_context_padded = batch['xq_context_padded'] # batch_size x maxlen_src 
        src_embed, src_padding_mask = self.prep_encode(xq_context_padded)
        memory = self.transformer.encoder(src_embed, src_key_padding_mask=src_padding_mask)
        memory_padding_mask = src_padding_mask
        return memory, memory_padding_mask

    def decode(self, z_padded, memory, memory_padding_mask):
        # Forward pass through decoder only
        #
        # Input
        # 
        #  memory : [b*nq (batch_size) x maxlen_src x hidden_size] output of transformer encoder
        #  memory_padding_mask : [b*nq (batch_size) x maxlen_src x hidden_size] binary mask padding where False means leave alone
        #
        # Output
        #   output : [b*nq x maxlen_target x output_size]
        tgt_embed, tgt_padding_mask, tgt_mask = self.prep_decode(z_padded)
        trans_out = self.transformer.decoder(tgt_embed, memory,
                tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=memory_padding_mask)
        output = self.out(trans_out)
        return output