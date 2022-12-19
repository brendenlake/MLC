import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import glob
import random
import numpy as np
import pickle
from copy import deepcopy, copy
from sklearn import utils
from train_lib import seed_all, display_input_output, list_remap, score_grammar
from interpret_grammar import str_to_grammar, Grammar, Rule, get_grammar_miniscan, is_prim_var, is_var
from itertools import permutations
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generating episodes for meta-training and evaluation. 
#   Includes code for few-shot learning tasks, open-ended tasks, and additional probes of biases.

# Special symbols
SOS_token = "SOS" # start of sentence
EOS_token = "EOS" # end of sentence
PAD_token = "PAD" # padding symbol
IO_SEP = 'IO' # separator '->' between input/outputs in support examples
ITEM_SEP  = SOS_token # separator '|' between support examples in input sequence

# Default input and output symbols 
input_symbols_list_default = ['dax', 'lug', 'wif', 'zup', 'fep', 'blicket', 'kiki', 'tufa', 'gazzer']
output_symbols_list_default = ['RED', 'YELLOW', 'GREEN', 'BLUE', 'PURPLE', 'PINK']

def get_dataset(episode_type):
# --------------------------------------------
# Datasets for main experiments
# --------------------------------------------
    if episode_type == 'retrieve': # BIML (copy only) training
        D_train = DataRetrieve('train',mydir='data_algebraic', min_ns=14, max_ns=14)
        D_val = DataRetrieve('val',mydir='data_algebraic', min_ns=14, max_ns=14)
    elif episode_type == 'algebraic': # not used in paper
        D_train = DataAlg('train',mydir='data_algebraic', min_ns=14)
        D_val = DataAlg('val',mydir='data_algebraic')
    elif episode_type == 'algebraic_noise': # BIML (algebraic only) training
        D_train = DataAlg('train',mydir='data_algebraic', p_noise=0.01, min_ns=14)
        D_val = DataAlg('val',mydir='data_algebraic')
    elif episode_type == 'algebraic+biases': # BIML training
        D_train = DataAlgAndBias('train',mydir='data_algebraic',p_noise=0.01, min_ns=14)
        D_val = DataAlg('val',mydir='data_algebraic')
    elif episode_type=='few_shot_gold': # For evaluating few-shot learning task with gold outputs
        D_train = []
        D_val = DataHumanFewShot('gold', inc_support_in_query=True, do_remap=True)
    elif episode_type=='few_shot_human': # For evaluating predictions of human behavior on few-shot learning task
        D_train = []
        D_val = DataHumanFewShot('behavior', data_mult=1, inc_support_in_query=False, do_remap=True)
    elif episode_type=='few_shot_human_mult10': # same dataset as above copied 10 times
        D_train = []
        D_val = DataHumanFewShot('behavior', data_mult=10, inc_support_in_query=False, do_remap=True)    
    elif episode_type=='open_end_human_all': # complete set of human open-ended responses
        D_train = DataHumanOpenEnded('train', inc_support_in_query=True, mydir='data_human/open_ended_all')
        D_val = DataHumanOpenEndedIterative('val','data_human/open_ended_all')
    elif episode_type=='open_end_freeform': # for evaluating model productions on open-ended task
        D_train = []
        D_val = DataFreeformOpenEnded(mydir='data_human/open_ended_all/mock')
    elif episode_type=='probe_human': # for evaluating predictions of human behavior on "additional biases" probe tasks (supplement)
        D_train = []
        D_val = DataHumanProbe('val',data_mult=10)
    elif episode_type=='probe_human_w_pool': # for training BIML (within-sample) on open-ended task
        D_train = DataHumanProbe('train',inc_pool=True)
        D_val = DataHumanProbe('val',data_mult=10,inc_pool=True)
    elif episode_type=='few_shot_vanilla': # Basic seq2seq training
        D_train = DataFewShotVanilla('train')
        D_val = DataFewShotVanilla('val')
    elif episode_type=='human_vanilla': # Basic seq2seq for evaluating predictions of human behavio
        D_train = []
        D_val = DataHumanVanilla('behavior', data_mult=1, inc_support_in_query=False, do_remap=False)
# --------------------------------------------
# Datasets for training and evaluating cross-validation on open-ended tasks.
#  to generate responses on iterative open-ended task, you can uncomment out the last line of each
# --------------------------------------------
    elif episode_type=='open_end_human_cross1':
        D_train = DataHumanOpenEnded('train', inc_support_in_query=True, mydir='data_human/open_ended_cross1')        
        D_val = DataHumanOpenEnded('val', inc_support_in_query=False, mydir='data_human/open_ended_cross1')
        # D_val = DataHumanOpenEndedIterative('val','data_human/open_ended_cross1')
    elif episode_type=='open_end_human_cross2':
        D_train = DataHumanOpenEnded('train', inc_support_in_query=True, mydir='data_human/open_ended_cross2')
        D_val = DataHumanOpenEnded('val', inc_support_in_query=False, mydir='data_human/open_ended_cross2')
        # D_val = DataHumanOpenEndedIterative('val','data_human/open_ended_cross2')
    elif episode_type=='open_end_human_cross3':
        D_train = DataHumanOpenEnded('train', inc_support_in_query=True, mydir='data_human/open_ended_cross3')
        D_val = DataHumanOpenEnded('val', inc_support_in_query=False, mydir='data_human/open_ended_cross3')
        # D_val = DataHumanOpenEndedIterative('val','data_human/open_ended_cross3')
    elif episode_type=='open_end_human_cross4':
        D_train = DataHumanOpenEnded('train', inc_support_in_query=True, mydir='data_human/open_ended_cross4')
        D_val = DataHumanOpenEnded('val', inc_support_in_query=False, mydir='data_human/open_ended_cross4')
        # D_val = DataHumanOpenEndedIterative('val','data_human/open_ended_cross4')
    elif episode_type=='open_end_human_cross5':
        D_train = DataHumanOpenEnded('train', inc_support_in_query=True, mydir='data_human/open_ended_cross5')
        D_val = DataHumanOpenEnded('val', inc_support_in_query=False, mydir='data_human/open_ended_cross5')
        # D_val = DataHumanOpenEndedIterative('val','data_human/open_ended_cross5')
    elif episode_type=='joint_cross1':
        D_train1 = DataAlgAndBias('train',mydir='data_algebraic', p_noise=0.01, min_ns=14)
        D_train2 = DataAlgAndBias('train',mydir='data_algebraic', p_noise=0.01, min_ns=0)
        D_train3 = DataHumanOpenEnded('train', inc_support_in_query=True, mydir='data_human/open_ended_cross1')
        D_train = MixDataset([D_train1,D_train2,D_train3])
        D_val = DataHumanOpenEnded('val', inc_support_in_query=False, mydir='data_human/open_ended_cross1')
        # D_val = DataHumanOpenEndedIterative('val','data_human/open_ended_cross1')
    elif episode_type=='joint_cross2':
        D_train1 = DataAlgAndBias('train',mydir='data_algebraic', p_noise=0.01, min_ns=14)
        D_train2 = DataAlgAndBias('train',mydir='data_algebraic', p_noise=0.01, min_ns=0)
        D_train3 = DataHumanOpenEnded('train', inc_support_in_query=True, mydir='data_human/open_ended_cross2')
        D_train = MixDataset([D_train1,D_train2,D_train3])
        D_val = DataHumanOpenEnded('val', inc_support_in_query=False, mydir='data_human/open_ended_cross2')
        # D_val = DataHumanOpenEndedIterative('val','data_human/open_ended_cross2')
    elif episode_type=='joint_cross3':
        D_train1 = DataAlgAndBias('train',mydir='data_algebraic', p_noise=0.01, min_ns=14)
        D_train2 = DataAlgAndBias('train',mydir='data_algebraic', p_noise=0.01, min_ns=0)
        D_train3 = DataHumanOpenEnded('train', inc_support_in_query=True, mydir='data_human/open_ended_cross3')
        D_train = MixDataset([D_train1,D_train2,D_train3])
        D_val = DataHumanOpenEnded('val', inc_support_in_query=False, mydir='data_human/open_ended_cross3')
        # D_val = DataHumanOpenEndedIterative('val','data_human/open_ended_cross3')
    elif episode_type=='joint_cross4':
        D_train1 = DataAlgAndBias('train',mydir='data_algebraic', p_noise=0.01, min_ns=14)
        D_train2 = DataAlgAndBias('train',mydir='data_algebraic', p_noise=0.01, min_ns=0)
        D_train3 = DataHumanOpenEnded('train', inc_support_in_query=True, mydir='data_human/open_ended_cross4')
        D_train = MixDataset([D_train1,D_train2,D_train3])
        D_val = DataHumanOpenEnded('val', inc_support_in_query=False, mydir='data_human/open_ended_cross4')
        # D_val = DataHumanOpenEndedIterative('val','data_human/open_ended_cross4')
    elif episode_type=='joint_cross5':
        D_train1 = DataAlgAndBias('train',mydir='data_algebraic', p_noise=0.01, min_ns=14)
        D_train2 = DataAlgAndBias('train',mydir='data_algebraic', p_noise=0.01, min_ns=0)
        D_train3 = DataHumanOpenEnded('train', inc_support_in_query=True, mydir='data_human/open_ended_cross5')
        D_train = MixDataset([D_train1,D_train2,D_train3])
        D_val = DataHumanOpenEnded('val', inc_support_in_query=False, mydir='data_human/open_ended_cross5')
        # D_val = DataHumanOpenEndedIterative('val','data_human/open_ended_cross5')
# --------------------------------------------
# Legacy datasets for comparison with old code
# --------------------------------------------
    # elif episode_type == 'algebraic_loose':
    #     D_train = DataAlg('train',mydir='data_algebraic_loose', min_ns=14)
    #     D_val = DataAlg('val',mydir='data_algebraic_loose')
    elif episode_type=='legacy_open_end_human':
        D_train = DataHumanOpenEnded('train', inc_support_in_query=True, mydir='data_human/legacy_open_ended')
        D_val = DataHumanOpenEnded('val', inc_support_in_query=False, mydir='data_human/legacy_open_ended')
    elif episode_type == 'legacy_algebraic': # previously 'rules++'
        D_train = DataAlg('train',mydir='debug_data_rules++', min_ns=15)
        D_val = DataAlg('val',mydir='debug_data_rules++')
    elif episode_type == 'legacy_algebraic+biases': # previously 'rbn++'
        D_train = DataAlgAndBias('train',mydir='debug_data_rules++',p_noise=0.01, min_ns=15)
        D_val = DataAlg('val',mydir='debug_data_rules++')
    elif episode_type == 'legacy_joint': # previously 'mix_rw_v3'
        D_train1 = DataAlgAndBias('train',mydir='debug_data_rules++', p_noise=0.01, min_ns=15)
        D_train2 = DataAlgAndBias('train',mydir='debug_data_rules++', p_noise=0.01, min_ns=0)
        D_train3 = DataHumanOpenEnded('train', mydir='data_human/legacy_open_ended', inc_support_in_query=True) # like open_end_human
        D_train = MixDataset([D_train1,D_train2,D_train3])
        D_val = DataAlg('val',mydir='debug_data_rules++')
    else:
        assert False # invalid episode types
    return D_train, D_val

def update_lang_w_pad(mylang):
    # add PAD_token to the language, if not already
    if PAD_token not in mylang.symbol2index:
        return Lang(mylang.symbols)
    return mylang

class Lang:
    #  Class for converting tokens strings to token index, and vice versa.
    #   Use separate class for input and output languages
    #
    def __init__(self, symbols):
        # symbols : list of all possible symbols besides special tokens SOS, EOS, and PAD
        n = len(symbols)
        assert(SOS_token not in symbols)
        assert(EOS_token not in symbols)
        assert(PAD_token not in symbols)
        assert(PAD_token != SOS_token)
        self.symbols = symbols # list of non-special symbols
        self.index2symbol = {n: SOS_token, n+1: EOS_token, n+2: PAD_token}
        self.symbol2index = {SOS_token : n, EOS_token : n+1, PAD_token : n+2}
        for idx,s in enumerate(symbols):
            self.index2symbol[idx] = s
            self.symbol2index[s] = idx
        self.n_symbols = len(self.index2symbol)
        self.PAD_idx = self.symbol2index[PAD_token]
        self.PAD_token = PAD_token
        assert(len(self.index2symbol)==len(self.symbol2index))

    def symbols_to_tensor(self, mylist, add_eos=True):
        # Convert a list of token strings to token index (adding a EOS token at end)
        # 
        # Input
        #  mylist  : list of m symbols as strings
        #  add_eos : true/false, if true add the EOS symbol at end
        #
        # Output
        #  output : [m or m+1 LongTensor] token index for each symbol (plus EOS if appropriate)
        mylist = copy(mylist)
        if add_eos: mylist.append(EOS_token)
        indices = [self.symbol2index[s] for s in mylist]
        output = torch.LongTensor(indices) # keep on CPU since this occurs inside Dataset getitem..
        return output

    def tensor_to_symbols(self, v):
        # Convert tensor of token index to token strings, breaking where we get a EOS token.
        #   The EOS token is not included at the end in the result string list.
        # 
        # Input
        #  v : python list of m indices, or 1D tensor
        #   
        # Output
        #  mylist : list of symbols (excluding EOS)
        if torch.is_tensor(v):
            assert v.dim()==1
            v = v.tolist()
        assert isinstance(v, list)
        mylist = []
        for x in v:
            s = self.index2symbol[x]
            if s == EOS_token:
                break
            mylist.append(s)
        return mylist

def readfile(fn_in):
    # Read episode from text file
    #
    # Input
    #  fn_in : filename to read
    #
    # Output
    #   Parsed version of the file, with struct
    #   
    fid = open(os.path.join(fn_in),'r')
    lines = fid.readlines()
    lines = [line.rstrip('\n') for line in lines]
    lines = [line for line in lines if line != '']
    idx_support = lines.index('*SUPPORT*')
    idx_query = lines.index('*QUERY*')
    idx_grammar = lines.index('*GRAMMAR*')
    x_support, y_support = parse_commands(lines[idx_support+1:idx_query])
    x_query, y_query = parse_commands(lines[idx_query+1:idx_grammar])
    grammar_str = lines[idx_grammar+1:]
    grammar_str = '\n'.join(grammar_str)
    fid.close()
    return {'xs':x_support, 'ys':y_support, 'xq':x_query, 'yq':y_query, 'grammar_str':grammar_str}

def parse_commands(lines):
    # Parse lines from input files into command sequence and output sequence
    #
    # Input
    #  lines: [list of strings], each of format "IN: a b c OUT: d e f""
    lines = [l.strip() for l in lines]
    lines = [l.lstrip('IN: ') for l in lines]
    D = [l.split(' OUT: ') for l in lines]
    x = [d[0].split(' ') for d in D]
    y = [d[1].split(' ') for d in D]
    return x, y

def bundle_biml_episode(x_support,y_support,x_query,y_query,myhash,aux={}):
    # Bundle components for an episode suitable for optimizing BIML
    # 
    # Input
    #  x_support [length ns list of lists] : input sequences (each a python list of words/symbols)
    #  y_support [length ns list of lists] : output sequences (each a python list of words/symbols)
    #  x_query [length nq list of lists] : input sequences (each a python list of words/symbols)
    #  x_query [length nq list of lists] : output sequences (each a python list of words/symbols)
    #  myhash : unique string identifier for this episode (should be order invariant for examples)
    #  aux [dict] : any misc information that we want to pass along with the episode
    #
    # Output
    #  sample : dict that stores episode information
    ns = len(x_support)
    xy_support = [ITEM_SEP]
    for j in range(ns):
        xy_support += x_support[j] + [IO_SEP] + y_support[j] + [ITEM_SEP]
    x_query_context = [item + xy_support for item in x_query] # Create the combined source sequence for every query
    sample = {}
    sample['identifier'] = myhash # unique identifying string for this episode (order invariant)
    sample['xs'] = x_support # support 
    sample['ys'] = y_support
    sample['xq'] = x_query # query
    sample['yq'] = y_query
    sample['xq_context'] = x_query_context
    if aux: sample['aux'] = aux
    return sample

def make_biml_batch(samples, langs):
    # Batch episodes into a series of padded input and target tensors
    # 
    # Input
    #  samples : list of dicts from bundle_biml_episode
    #  langs : input and output version of Lang class
    assert isinstance(samples,list)
    m = len(samples)
    mybatch = {}
    mybatch['list_samples'] = samples
    mybatch['batch_size'] = m
    mybatch['xq_context'] = [] # list of source sequences (as lists) across all episodes
    mybatch['xq'] = []  # list of queries (as lists) across all episodes
    mybatch['yq'] = [] # list of query outputs (as lists) across all episodes
    mybatch['q_idx'] = [] # index of which episode each query belongs to
    mybatch['in_support'] = [] # bool list indicating whether each query is in its corresponding support set, or not
    for idx in range(m): # each episode
        sample = samples[idx]
        nq = len(sample['xq'])
        assert(nq == len(sample['yq']))
        mybatch['xq_context'] += sample['xq_context']
        mybatch['xq'] += sample['xq']
        mybatch['yq'] += sample['yq']
        mybatch['q_idx'] += [idx*torch.ones(nq, dtype=torch.int)]
        mybatch['in_support'] += [x in sample['xs'] for x in sample['xq']]
    mybatch['q_idx'] = torch.cat(mybatch['q_idx'], dim=0)
    mybatch['xq_context_padded'],mybatch['xq_context_lengths'] = build_padded_tensor(mybatch['xq_context'], langs['input'])
    mybatch['yq_padded'],mybatch['yq_lengths'] = build_padded_tensor(mybatch['yq'], langs['output'])
    mybatch['yq_sos_padded'],mybatch['yq_sos_lengths'] = build_padded_tensor(mybatch['yq'],langs['output'],add_eos=False,add_sos=True)
    return mybatch

def set_batch_to_device(batch):
    # Make sure all padded tensors are on GPU if needed
    tensors_to_gpu = [k for k in batch.keys() if '_padded' in k]
    for k in tensors_to_gpu:
        batch[k] = batch[k].to(device=DEVICE)
    return batch

def get_batch_output_pool(batch):
    # Optional info for certain tasks: Retrieve which symbols are allowed at output
    #
    # Input
    #  batch : Output from make_biml_batch
    # Output
    #  out_mask_allow : list of symbols (strings) from output vocab that are allowed
    samples_batch = batch['list_samples']
    out_mask_allow = []
    if 'output_pool' in samples_batch[0]['aux']:
        out_mask_allow = deepcopy(samples_batch[0]['aux']['output_pool'])
        assert(all([out_mask_allow == sample['aux']['output_pool'] for sample in samples_batch])), "The whole batch must have same output pool. Try batch_size=1."
        if EOS_token not in out_mask_allow: out_mask_allow.append(EOS_token)        
    else:
        assert(not any(['output_pool' in sample['aux'] for sample in samples_batch])), "The whole batch must have same output pool. Try batch_size=1."
    return out_mask_allow

def build_padded_tensor(list_seq, lang, add_eos=True, add_sos=False):
    # Transform list of python lists to a padded torch tensors
    # 
    # Input
    #  list_seq : list of n sequences (each sequence is a python list of token srings)
    #  lang : language object for translation of token string to token index
    #  add_eos : add end-of-sequence token at the end?
    #  add_sos : add start-of-sequence token at the beginning?
    #
    # Output
    #  z_padded : LongTensor (n x max_len)
    #  z_lengths : python list of sequence lengths (n-length list of scalars)
    n = len(list_seq)
    if n==0: return [],[]
    z_eos = list_seq
    if add_sos: 
        z_eos = [[SOS_token]+z for z in z_eos]
    if add_eos:
        z_eos = [z+[EOS_token] for z in z_eos]    
    z_lengths = [len(z) for z in z_eos]
    max_len = max(z_lengths) # maximum length in this episode
    z_padded = [pad_seq(z, max_len) for z in z_eos]
    z_padded = [lang.symbols_to_tensor(z, add_eos=False).unsqueeze(0) for z in z_padded]
    z_padded = torch.cat(z_padded, dim=0) # n x max_len
    return z_padded,z_lengths

def pad_seq(seq, max_length):
    # Pad token string sequence with the PAD_token symbol to achieve max_length
    #
    # Input
    #  seq : list of symbols (as strings)
    #
    # Output
    #  seq : padded list now extended to length max_length
    seq += (max_length - len(seq)) * [PAD_token]
    return seq

def make_hashable(G):
    # Create unique identifier for episodes defined by a grammar.
    #  Separate and sort rules in string format.
    #
    # Input
    #   G : string of elements separated by \n specifying the structure of an episode 
    G_str = str(G).split('\n')
    G_str = [s.strip() for s in G_str]
    G_str = [s for s in G_str if s != ''] # remove empty strings
    G_str.sort()
    out = '\n'.join(G_str)
    return out.strip()

def combine_input_output_symb(list_input_symb,list_output_symb):
    # Make new source vocabulary that combines list of input and output symbols.
    #  Include input/output and item separators, IO_SEP,ITEM_SEP.
    #  Exclude EOS_token,SOS_token,PAD_token, which will be added automatically by Lang constructor/
    #
    # Input
    #   list_input_symb : list of token symbols (strings)
    #   list_output_symb : list of token symbols (strings)
    # Output
    #   comb : combined list of tokens as strings
    additional_symb = sorted(set([IO_SEP,ITEM_SEP])-set([EOS_token,SOS_token,PAD_token]))
    comb = sorted(set(list_input_symb + list_output_symb + additional_symb))
    return comb

class DataAlg(Dataset):
    # Meta-training for few-shot grammar learning (fully algebraic)

    def __init__(self, mode, mydir, p_noise=0., inc_support_in_query=True, min_ns=0):
        # Each episode has different latent (algebraic) grammar. 
        #  The number of support items picked uniformly from min_ns...max_ns
        #
        # Input
        #  mode: 'train' or 'val'
        #  mydir : directory where data is stored
        #  p_noise : for a given symbol emission, probability that it will be from uniform distribution
        #  inc_support_in_query : default=True. Boolean. Should support items also be query items?
        #  min_ns : min number of support items in episode
        assert mode in ['train','val']
        self.mode = mode
        self.train = mode == 'train'
        self.p_noise = p_noise
        self.mydir_items = os.path.join(mydir,self.mode)
        self.list_items = glob.glob(self.mydir_items+"/*.txt") # all episode files
        self.input_symbols = input_symbols_list_default
        self.output_symbols = output_symbols_list_default
        comb = combine_input_output_symb(self.input_symbols,self.output_symbols)
        self.langs = {'input' : Lang(comb), 'output': Lang(self.output_symbols)}
        self.min_ns = min_ns # min number of support items in episode
        self.inc_support_in_query = inc_support_in_query

    def __len__(self):
        return len(self.list_items)

    def __getitem__(self, idx):
        S = readfile(self.list_items[idx])
        max_ns = len(S['xs'])
        if self.train:
            S['xs'], S['ys'] = utils.shuffle(S['xs'], S['ys'])
            ns = random.randint(self.min_ns,max_ns)
            S['xs'] = S['xs'][:ns]
            S['ys'] = S['ys'][:ns]
        if self.p_noise > 0: # emission noise
            for i in range(len(S['yq'])):
                S['yq'][i] = add_response_noise(S['yq'][i],self.p_noise,self.langs)
        if self.inc_support_in_query:
            S['xq'] = S['xs'] + S['xq']
            S['yq'] = S['ys'] + S['yq']
        myhash = make_hashable(S['grammar_str'])
        aux = {'grammar_str':S['grammar_str']}        
        return bundle_biml_episode(S['xs'],S['ys'],S['xq'],S['yq'],myhash,aux=aux)

def add_response_noise(yq_item, p_noise, langs):
    # Flip coin with weight p_noise of heads. If heads, replace output symbol with an arbitrary one.
    #   This symbol could be EOS token, in which case, we end the example.
    #
    # Input
    #  yq_item : list of output symbols for the response to a single command (excluding EOS)
    #  p_noise : probability of lapse (uniform draw) for a particular emission
    #
    assert(isinstance(yq_item,list))
    assert(isinstance(yq_item[0],str))
    choice_set_wo_eos = langs['output'].symbols
    choice_set = langs['output'].symbols + [EOS_token]
    yq_item = deepcopy(yq_item)
    for i in range(len(yq_item)): # for each token
        if flip(p_noise):
            new_token = random.choice(choice_set)
            if i==0: new_token = random.choice(choice_set_wo_eos)
            if new_token == EOS_token:
                return yq_item[:i]
            yq_item[i] = new_token
    return yq_item

class DataRetrieve(DataAlg):
    # Copy task of retrieving study examples

    def __init__(self, mode, mydir, min_ns=2, max_ns=2):
        # Each episode has a set of support strings, identical to DataAlg class
        #  Number of support items picked uniformly from min_ns...max_ns
        #
        # Input
        #  mode: 'train' or 'val'
        #  mydir : directory where data is stored
        super().__init__(mode, mydir, inc_support_in_query=True, min_ns=min_ns)
        self.my_max_ns = max_ns

    def __getitem__(self, idx):
        S = readfile(self.list_items[idx])
        assert(len(S['xs'])>=self.my_max_ns)
        if self.train:
            S['xs'], S['ys'] = utils.shuffle(S['xs'], S['ys'])
        ns = random.randint(self.min_ns,self.my_max_ns)
        S['xs'] = S['xs'][:ns]
        S['ys'] = S['ys'][:ns]        
        S['xq'] = S['xs']
        S['yq'] = S['ys']
        myhash = make_hashable(S['grammar_str'])
        aux = {'grammar_str':S['grammar_str']}
        return bundle_biml_episode(S['xs'],S['ys'],S['xq'],S['yq'],myhash,aux=aux)

class DataAlgAndBias(DataAlg):
    # Meta-training for full BIML model

    def __init__(self, mode, mydir, p_bias=0.2, p_noise=0., inc_support_in_query=True, min_ns=0):
        # Optimize for a combination of algebraic and bias-based outputs.
        #  Each episode has different latent (algebraic) grammar. 
        #  The number of support items picked uniformly from min_ns...max_ns.
        #  With probability=p_bias, a query uses a heuristic output rather than an algebraic one.
        #
        # Input
        #  mode: 'train' or 'val'
        #  mydir : directory where data is stored
        #  p_bias : for a given query, probability of producing a heuristic rather than algebraic output
        #  p_noise : for a given symbol emission, probability that it will be from uniform distribution
        #  inc_support_in_query : should support items also be query items?
        #  min_ns : minimum number of support items
        super().__init__(mode, mydir, p_noise=p_noise, inc_support_in_query=inc_support_in_query, min_ns=min_ns)
        self.p_bias = p_bias
        self.maxlen = 8 # maximum length of input and output sequences 

    def __getitem__(self, idx):
        S = readfile(self.list_items[idx])
        max_ns = len(S['xs'])
        if self.train:            
            S['xs'], S['ys'] = utils.shuffle(S['xs'], S['ys'])
            ns = random.randint(self.min_ns,max_ns)
            S['xs'] = S['xs'][:ns]
            S['ys'] = S['ys'][:ns]
        prims = get_prims(S['xs'],S['ys'])
        for i in range(len(S['xq'])):
            if flip(self.p_bias): # if true, then make heuristic trial                
                if flip(): # one-to-one, left-to-right translation
                    S['yq'][i] = use_one2one_heuristic(S['xq'][i],prims,self.langs)
                else: # apply rules with some of the rules flipped                
                    S['yq'][i] = use_flipped_grammar(S['xq'][i],S['grammar_str'],self.maxlen)
        if self.p_noise > 0: # emission noise
            for i in range(len(S['yq'])):
                S['yq'][i] = add_response_noise(S['yq'][i],self.p_noise,self.langs)
        if self.inc_support_in_query:
            S['xq'] = S['xs'] + S['xq']
            S['yq'] = S['ys'] + S['yq']
        myhash = make_hashable(S['grammar_str'])
        aux = {'grammar_str':S['grammar_str']}
        return bundle_biml_episode(S['xs'],S['ys'],S['xq'],S['yq'],myhash,aux=aux)


def get_prims(xs,ys):
    # Find each primitive token (in support set) and what output symbols it maps to
    #
    # Input
    #  xs: list of support inputs (list of list of symbols)
    #  ys: list of support outputs (list of list of symbols)
    #
    # Output
    #  prim : dictionary of primitive mappings
    ns = len(xs)
    prims = {}
    for i in range(ns):
        if len(xs[i])==1:
            assert len(ys[i])==1
            x_sym = xs[i][0]
            y_sym = ys[i][0]
            prims[x_sym] = y_sym
    return prims

def use_one2one_heuristic(xq_item,prims,langs):
    # Answer query using a one-to-one, left-to-right translation
    # 
    # input
    #  xq_item : Single query command (list of symbols)
    #  prims : dict for direct input/output mappings (primitives)
    #
    # output
    #  yq_item : Single query output (list of symbols)
    assert(isinstance(xq_item,list))
    assert(isinstance(xq_item[0],str))
    yq_item = []
    for x_sym in xq_item:
        if x_sym in prims:
            yq_item.append(prims[x_sym])
        else:
            yq_item.append(random.choice(langs['output'].symbols)) # pick from non-special symbols
    return yq_item

def flip(p_head=0.5):
    # return True with probability p_head
    return random.random() < p_head

def flip_RHS(RHS_list):
    # For a two-argument rule like [u1] dax [u2] -> [u2] [u1] [u2], flip the order of the
    #  arguments on the output side, e.g., [u1] dax [u2] -> [u1] [u2] [u1]
    # 
    # Input
    #   RHS_list : list of right-hand-side variables like '[u2] [u2] [x1]'
    # Output
    #   same list, but with roles of variables switched
    list_var = sorted(set(RHS_list))
    assert(len(list_var)==2)
    dmap = {list_var[0] : list_var[1], list_var[1] : list_var[0]}
    return [dmap[v] for v in RHS_list]

def use_flipped_grammar(xq_item,grammar_str,maxlen,input_symbols=input_symbols_list_default):
    # Apply mutated grammar to query (xq_item), where 
    #  each of the two-argument rules has a chance of 
    #  swapping the arguments when applied
    #
    # Input
    #  xq_item : Single query command (list of token strings)
    #  grammar_str : (string) gold grammar for episode
    #  maxlen : maximum output length
    #  input_symbols : different possible input tokens(for sake of constructing the grammar)
    #
    # Output
    #  yq_item : Single query output (list of symbols)
    assert(isinstance(xq_item,list))
    assert(isinstance(xq_item[0],str))
    xq = ' '.join(xq_item)
    G = str_to_grammar(grammar_str, input_symbols)
    yq_prev = G.apply(xq)
    rules_update = G.rules
    nr = len(rules_update)
    for j in range(nr):
        myrule = rules_update[j]
        if flip() and len(myrule.LHS_list) == 3: # only consider two-argument rules
            new_RHS = flip_RHS(myrule.RHS_list)
            rules_update[j] = Rule(myrule.LHS_str, ' '.join(new_RHS))
    G = Grammar(rules_update, input_symbols)
    yq = G.apply(xq)
    yq_item = yq.split()
    if len(yq_item) > maxlen:
        yq_item = yq_prev.split()
    return yq_item

class DataHumanFewShot(Dataset):
    # For evaluating predictions on human few-shot learning behavior
    
    def __init__(self, mode, inc_support_in_query=False, do_remap=True, data_mult=1, mydir='data_human/few_shot'):
        # Input
        #   mode : 'gold' for ground-truth responses for each command
        #          'behavior' for actual human responses
        #  inc_support_in_query : (default=False) include all support examples as queries? If True, it messes up the log-likelihood calculations.
        #  do_remap : [default=True] Randomly remap the input and output symbols to standard pseudowords and colors 
        #  data_mult : [default=1] replicate the queries this many times (useful if we want multiple samples from models)
        assert mode in ['gold','behavior']
        self.placeholder_length = 100 # placeholder number of episodes (if 'gold' mode)
        self.mode = mode
        self.is_gold = mode == 'gold'
        self.randomize_order = True
        self.do_remap = do_remap
        self.data_mult = data_mult
        self.inc_support_in_query = inc_support_in_query
        subdir = 'val_' + self.mode
        self.mydir_items = os.path.join(mydir,subdir)    
        self.fn_gold = os.path.join(mydir,'val_gold','mini_scan.txt') #gold filename
        self.list_items = glob.glob(self.mydir_items+'/*.txt') # all episode files
        if do_remap:
            self.input_symbols = input_symbols_list_default
            self.output_symbols = output_symbols_list_default
        else:
            self.input_symbols = ['1','2','3','DAX','thrice','surround','after']
            self.output_symbols = ['1','2','3','DAX']
        comb = combine_input_output_symb(self.input_symbols,self.output_symbols)
        self.langs = {'input' : Lang(comb), 'output': Lang(self.output_symbols)}

    def __len__(self):
        if self.is_gold:
            return self.placeholder_length
        else:
            return len(self.list_items)

    def __getitem__(self, idx):
        if self.is_gold: # gold output strings
            S = readfile(self.fn_gold)
        else: # behavioral data
            S = readfile(self.list_items[idx])
        if self.randomize_order:
            S['xs'], S['ys'] = utils.shuffle(S['xs'], S['ys'])
        S['aux'] = {}
        if self.do_remap: S = assign_random_map(S, self.input_symbols, self.output_symbols)
        if self.inc_support_in_query:
            S['xq'] = S['xs'] + S['xq']
            S['yq'] = S['ys'] + S['yq']
        if self.data_mult>1:
            S['xq'] = S['xq']*self.data_mult
            S['yq'] = S['yq']*self.data_mult
        if self.is_gold and not self.do_remap:
            S['aux']['grammar_str'] = S['grammar_str']
        return bundle_biml_episode(S['xs'],S['ys'],S['xq'],S['yq'],'',aux=S['aux'])

    def get_raw_item(self, idx):
        # Get raw-text version of an episode
        assert not self.is_gold, "only applicable for behavioral data"
        S = readfile(self.list_items[idx])
        return S

class DataHumanProbe(Dataset):
    # Predicting human responses on "additional inductive bias" probe experiments

    def __init__(self, mode, data_mult=1, inc_pool=False, mydir='data_human/data_behavior_probes'):
        # Input
        #  mode : train vs. val
        #  data_mult : copy the examples this many times
        #  inc_pool : if True, use string to list pool of possible options
        assert mode in ['train','val']
        self.mode = mode
        self.placeholder_length = 100000 # placeholder number of episodes (if 'train' mode)
        self.is_train = mode == 'train'
        self.randomize_order = True
        self.do_remap = True
        self.inc_support_in_query = True
        self.data_mult = data_mult      
        self.inc_pool = inc_pool  
        self.mydir_items = os.path.join(mydir,'val')    
        self.list_items = glob.glob(self.mydir_items+'/*.txt') # all episode files        
        self.input_symbols = input_symbols_list_default
        self.output_symbols = output_symbols_list_default
        comb = combine_input_output_symb(self.input_symbols,self.output_symbols)
        self.langs = {'input' : Lang(comb), 'output': Lang(self.output_symbols)}

    def __len__(self):
        if self.is_train:
            return self.placeholder_length
        else:
            return len(self.list_items)

    def __getitem__(self, idx):
        if self.is_train:
            fn = random.choice(self.list_items)
            S = readfile(fn)
        else:
            fn = self.list_items[idx] ## if we truly want to iterate over files
            S = readfile(fn) 
        output_pool = S['grammar_str'].lstrip('allowed outputs:\n').split()
        if self.randomize_order:
            S['xs'], S['ys'] = utils.shuffle(S['xs'], S['ys'])
        if self.do_remap: S = assign_random_map(S, self.input_symbols, self.output_symbols, output_pool)
        if self.inc_support_in_query:
            S['xq'] = S['xs'] + S['xq']
            S['yq'] = S['ys'] + S['yq']
        if self.inc_pool:
            S['xs'] = [[]] + S['xs']
            S['ys'] = [S['aux']['output_pool']] + S['ys']
        if self.data_mult>1:
            S['xq'] = S['xq']*self.data_mult
            S['yq'] = S['yq']*self.data_mult
        S['aux']['filename'] = fn
        return bundle_biml_episode(S['xs'],S['ys'],S['xq'],S['yq'],'',aux=S['aux'])

    def get_raw_item(self, idx):
        # Get raw-text version of an episode
        S = readfile(self.list_items[idx])
        output_pool = S['grammar_str'].lstrip('allowed outputs:\n').split()
        S['aux'] = {'output_pool' : output_pool, 'filename':self.list_items[idx] }
        return S

class DataFewShotVanilla(Dataset):
    # Train vanilla seq2seq model directly on few-shot learning task

    def __init__(self, mode, mydir='data_human/few_shot'):
        # Input
        assert mode in ['train','val']
        self.placeholder_length = 200 # placeholder number of episodes (if 'gold' mode)
        self.mode = mode
        self.is_train = self.mode =='train'
        self.randomize_order = True
        self.fn_gold = os.path.join(mydir,'val_gold','mini_scan.txt') #gold filename
        self.input_symbols = ['1','2','3','DAX','thrice','surround','after']
        self.output_symbols = ['1','2','3','DAX']
        comb = combine_input_output_symb(self.input_symbols,[])
        self.langs = {'input' : Lang(comb), 'output': Lang(self.output_symbols)}

    def __len__(self):
        if self.is_train:
            return self.placeholder_length
        else:
            return 1

    def __getitem__(self, idx):
        S = readfile(self.fn_gold)
        if self.is_train:
            S['xs'], S['ys'] = utils.shuffle(S['xs'], S['ys'])
            S['xq'] = S['xs']
            S['yq'] = S['ys']
        S['xs'] = [] # we have no support items when using vanilla seq2seq
        S['ys'] = []
        return bundle_biml_episode(S['xs'],S['ys'],S['xq'],S['yq'],'',aux={'none':[]})

class DataHumanVanilla(DataHumanFewShot):
    # Human behavior in format accessible to vanilla seq2seq

    def __init__(self, mode, inc_support_in_query=False, do_remap=True, data_mult=1, mydir='data_human/few_shot'):
        # Input
        #   mode : 'gold' for ground-truth responses for each command
        #          'behavior' for actual human responses
        #  inc_support_in_query : (default=False) include all support examples as queries? If True, it messes up the log-likelihood calculations.
        #  do_remap : [default=True] Randomly remap the input and output symbols to standard pseudowords and colors? 
        #  data_mult : [default=1] replicate the queries this many times (useful if we want multiple samples from models)
        assert mode in ['behavior']
        assert not do_remap
        assert not inc_support_in_query
        super().__init__(mode, inc_support_in_query, do_remap, data_mult, mydir)

    def __getitem__(self, idx):
        # Get raw-text version of an episode        
        S = readfile(self.list_items[idx])
        S['xs'] = [] # we don't need the support set
        S['ys'] = []
        xq = []
        yq = []
        nq = len(S['xq'])
        for i in range(nq): # skip human examples that have novel color not in support
            if 'undefined_action' not in S['yq'][i]:
                xq.append(S['xq'][i])
                yq.append(S['yq'][i])
            else:
                print('  skipped',' '.join(S['xq'][i]),'->',' '.join(S['yq'][i]))
        S['xq'] = xq
        S['yq'] = yq
        S['aux'] = {'unmap_output': lambda x:x, 'unmap_input': lambda x:x}
        return bundle_biml_episode(S['xs'],S['ys'],S['xq'],S['yq'],'',aux=S['aux'])

class DataHumanOpenEnded(Dataset):
    # Human data for open-ended task with arbitrary query/support split for each participant
    
    def __init__(self, mode, mydir, inc_support_in_query=False):
        # Input
        #  mode : 'train' or 'val' data
        #  inc_support_in_query : (default=False) include all support examples as queries?
        #        If True, it messes up the log-likelihood calculations.
        #  mydir : directory where data is stored
        assert mode in ['train','val']        
        self.placeholder_length = 100000 # placeholder number of episodes in epoch
        self.mode = mode
        self.train = mode == 'train'
        self.randomize_order = True
        self.inc_support_in_query = inc_support_in_query
        self.mydir_items = os.path.join(mydir,self.mode)
        self.list_items = glob.glob(self.mydir_items+"/*.txt") # all episode files
        self.input_symbols = input_symbols_list_default
        self.output_symbols = output_symbols_list_default
        comb = combine_input_output_symb(self.input_symbols,self.output_symbols)
        self.langs = {'input' : Lang(comb), 'output': Lang(self.output_symbols)}

    def __len__(self):
        if self.train:
            return self.placeholder_length
        else:
            return len(self.list_items)

    def __getitem__(self, idx):
        if self.train:
            S = readfile(random.choice(self.list_items))
        else:
            S = readfile(self.list_items[idx]) ## if we truly want to iterate over files
        assert(len(S['xs'])==0) # in the text files, all items should be listed as queries
        if self.randomize_order:
            S['xq'], S['yq'] = utils.shuffle(S['xq'], S['yq'])
        S = assign_random_map(S, self.input_symbols, self.output_symbols)
        S = self.__random_query_split_uniform(S)
        if self.inc_support_in_query:
            S['xq'] = S['xs'] + S['xq']
            S['yq'] = S['ys'] + S['yq']  
        return bundle_biml_episode(S['xs'],S['ys'],S['xq'],S['yq'],'',aux=S['aux'])

    def get_raw_item(self, idx):
        # Get raw-text version of an episode
        S = readfile(self.list_items[idx]) ## if we truly want to iterate over files
        assert(len(S['xs'])==0) # in the text files, all items should be listed as queries
        return S

    def __random_query_split_uniform(self, S):
        # Randomly split the examples between query and support, uniformly over support size
        x = S['xs'] + S['xq']
        y = S['ys'] + S['yq']
        x, y = utils.shuffle(x,y)
        n = len(x)
        ns = random.randint(0,n-1)
        S['xs'] = x[:ns]
        S['ys'] = y[:ns]
        S['xq'] = x[ns:]
        S['yq'] = y[ns:]
        return S

class DataHumanOpenEndedIterative(Dataset):
    # Human validation data for open-ended task with iterative prediction of queries.
    #  For each query, all previous queries are the study examples.
    
    def __init__(self, mode, mydir):
        # Input
        #  mode: must be validation ('val') data
        #  mydir : directory where data is stored
        assert(mode=='val')
        self.mode = mode
        self.mydir_items = os.path.join(mydir,mode)
        self.list_items = glob.glob(self.mydir_items+"/*.txt") # all episode files
        self.input_symbols = input_symbols_list_default
        self.output_symbols = output_symbols_list_default
        comb = combine_input_output_symb(self.input_symbols,self.output_symbols)
        self.langs = {'input' : Lang(comb), 'output': Lang(self.output_symbols)}

    def __len__(self):
        return len(self.list_items)

    def __getitem__(self, idx):        
        S = readfile(self.list_items[idx])
        assert(len(S['xs'])==0) # in the text files, all items should be listed as queries
        S['xq'], S['yq'] = utils.shuffle(S['xq'], S['yq']) # randomize order of queries        
        S = assign_random_map(S, self.input_symbols, self.output_symbols)
        return self.__bundle_episode_custom(S['xq'],S['yq'])

    def __bundle_episode_custom(self,x_query,y_query):
        # Provide all previous queries as support for the current query.
        # 
        # Input
        #  x_query [length nq list of lists] : input sequences (each a python list of words/symbols)
        #  x_query [length nq list of lists] : output sequences (each a python list of words/symbols)
        #
        # Output
        #  sample : dict that stores episode information
        nq = len(x_query)
        x_query_context = []
        for qi in range(nq):
            xy_support = [ITEM_SEP]
            for si in range(qi):
                xy_support += x_query[si] + [IO_SEP] + y_query[si] + [ITEM_SEP]
            x_query_context.append(x_query[qi] + xy_support)
        sample = {}
        sample['identifier'] = ''
        sample['xs'] = []
        sample['ys'] = []
        sample['xq'] = x_query
        sample['yq'] = y_query
        sample['xq_context'] = x_query_context
        return sample

class DataFreeformOpenEnded(Dataset):
    # Dataset for iteratively filling out open-ended task, one query at a time
    
    def __init__(self, mydir):
        # Input
        #  mydir : directory where data is stored
        self.placeholder_length = 100 # placeholder number of episodes in epoch
        self.randomize_order = True
        self.fn = os.path.join(mydir,'000000.txt')
        assert(os.path.exists(self.fn)) # check that file exists
        self.input_symbols = input_symbols_list_default
        self.output_symbols = output_symbols_list_default
        comb = combine_input_output_symb(self.input_symbols,self.output_symbols)
        self.langs = {'input' : Lang(comb), 'output': Lang(self.output_symbols)}

    def __len__(self):
        return self.placeholder_length

    def __getitem__(self, idx):        
        S = readfile(self.fn)
        assert(len(S['xs'])==0) # in the text files, all items should be listed as queries
        if self.randomize_order:
            S['xq'], S['yq'] = utils.shuffle(S['xq'], S['yq'])
        S = assign_random_map(S, self.input_symbols, self.output_symbols)
        assert(len(S['xs'])==0 and len(S['ys'])==0)
        return bundle_biml_episode(S['xs'],S['ys'],S['xq'],S['yq'],'',aux=S['aux'])

class MixDataset(Dataset):
    # Dataset that chooses episodes from a mixture of other datasets

    def __init__(self, list_datasets):
        # list_datasets: list of Dataset objects
        self.list_datasets = list_datasets
        self.mylens = np.array([len(D) for D in self.list_datasets])
        self.cumsum = np.cumsum(self.mylens)

        # check that all of the languages are the same for the datasets
        self.langs = self.list_datasets[0].langs
        for D in self.list_datasets:
            assert(self.langs['input'].symbol2index == D.langs['input'].symbol2index), "input libraries must match"            
            assert(self.langs['output'].symbol2index == D.langs['output'].symbol2index), "output libraries must match"
            D.langs = self.langs

    def __len__(self):
        return np.sum(self.mylens)

    def __getitem__(self, idx):
        j, new_idx = self.idx_to_D(idx)
        D = self.list_datasets[j]
        return D.__getitem__(new_idx)

    def idx_to_D(self, idx):
        # which dataset given the current index in epoch?
        #
        # output
        #  j: index of the right dataset
        #  new_idx: index into that dataset
        assert(idx < len(self)), "index must be less than the overall length"
        lt = idx < self.cumsum
        j = np.min(np.nonzero(lt))
        new_idx = idx
        if j > 0:
            new_idx = new_idx - self.cumsum[j-1]
        return j, new_idx # j is index of the right dataset, new_idx is index into that dataset

def assign_random_map(S, new_input_symbols, new_output_symbols, output_pool=[]):
    # Each input and output symbol in the episode file are re-assigned based on 
    #   the standard set of input/output symbols
    #
    # Input
    #  S : dict with 'xs', 'ys', 'xq', 'yq' input/output examples in old spaces
    #  new_input_symbols : list of new input symbols we want to use
    #  new_output_symbols : list of new output symbols we want to use
    #  output_pool : (option) list of output symbols that are allowed for this episode. 
    #     Otherwise, default to all output symbols used in support and query (undefined_action for others)
    #
    # Output
    #   S : updated dict with remapped input/output patterns
    S_raw = S
    S = deepcopy(S)
    ns = len(S['xs'])
    nq = len(S['xq'])
    list_x = S['xs'] + S['xq']
    list_x = sum(list_x, [])
    unique_in_symb = sorted(set(list_x)) # all original input symbols in episode
    if len(output_pool)>0:
        unique_out_symb = output_pool
    else:
        list_y = S['ys'] + S['yq']
        list_y = sum(list_y, [])
        unique_out_symb = sorted(set(list_y)) # all original output symbols in episode

    # construct mappings
    lang_in_symb = deepcopy(new_input_symbols)
    lang_out_symb = deepcopy(new_output_symbols)
    random.shuffle(lang_in_symb)
    random.shuffle(lang_out_symb)
    lang_in_symb_trunc = lang_in_symb[:len(unique_in_symb)] # truncate if all symbols aren't needed
    lang_out_symb_trunc = lang_out_symb[:len(unique_out_symb)] # truncate if all symbols aren't needed
    map_input = lambda command: list_remap(command,unique_in_symb,lang_in_symb_trunc)
    map_output = lambda actions: list_remap(actions,unique_out_symb,lang_out_symb_trunc)

    # functions to invert mappings (emissions not used in query/support will be mapped to 'undefined_action')
    uo_lhs = lang_out_symb
    uo_rhs = unique_out_symb + ['undefined_action']*(len(uo_lhs)-len(unique_out_symb))
    unmap_input = lambda command: list_remap(command,lang_in_symb_trunc,unique_in_symb)
    unmap_output = lambda actions: list_remap(actions,uo_lhs,uo_rhs)

    S['aux'] = {}
    if output_pool: S['aux']['output_pool'] = map_output(output_pool)
    S['aux']['unmap_input'] = unmap_input
    S['aux']['unmap_output'] = unmap_output
    S['xs'] = list(map(map_input,S['xs']))
    S['xq'] = list(map(map_input,S['xq']))
    S['ys'] = list(map(map_output,S['ys']))
    S['yq'] = list(map(map_output,S['yq']))
    return S

if __name__ == "__main__":
    
    # Example episode for meta-training with full BIML model
    D_train, D_val = get_dataset('algebraic+biases')
    sample = D_train[0]
    print("")
    print('Example episode')
    print("")
    print('*Study examples*')
    display_input_output(sample['xs'],sample['ys'],sample['ys'])
    print("")
    print('*Query examples*') # includes 20 study examples and 20 nove queries
    display_input_output(sample['xq'],sample['yq'],sample['yq'])