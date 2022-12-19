import os
import shutil
import glob
import random
import numpy as np
import filecmp
import re
from copy import deepcopy, copy
from datasets import Lang, flip, make_hashable, bundle_biml_episode, input_symbols_list_default, output_symbols_list_default, equiv_to_miniscan
from train_lib import seed_all
from interpret_grammar import str_to_grammar, Grammar, Rule, get_grammar_miniscan

# For generating a dataset of meta-training episodes, for teaching BIML to do few-shot learning of algebraic systems.
# This is not needed if you are okay training on the existing 100K episodes which can be downloaded.

p_lhs_onearg = 0.5 # Probability that we have a single argument function in grammar. Otherwise, we have two arguments
p_stop_rhs = 0.6 # Probability of stopping instead of generating more variables in the right hand side of a rule
vars_atom = ['u1','u2'] # special tokens indicating a variable that can be filled by a single token
vars_string = ['x1','x2'] # special tokens indicating a variable that can be filled by arbitrary string
icon_concat_rule = Rule('u1 x1','[u1] [x1]')

def generate_rules_dataset(nsamp_train,nsamp_val,mydir='data_algebraic',episode_type='algebraic'):
    # Generate and write dataset to file
    # 
    # Input
    #  nsamp_train : number of training episodes
    #  nsamp_val : number of validation episodes
    #  mydir : output folder
    #
    # Output
    #   writes text files for each of the episodes
    #
    seed_all() 
    if os.path.exists(mydir):
        raise ValueError('Attempting to create dataset in existing directory: ' + mydir)
    os.mkdir(mydir)
    os.mkdir(mydir+'/train')
    os.mkdir(mydir+'/val')
    generate_episode_train, _ = get_episode_generator(episode_type)
    
    print( "Generating validation episodes")
    tabu_list = []
    for s in range(1,nsamp_val+1):
        if (s % 50==0): print('  episode ' + num2str(s))
        sample = generate_episode_train([])
        episode_to_file(os.path.join(mydir,'val',num2str(s)+'.txt'),sample)
        tabu_list.append(sample['identifier'])

    print( "Generating training episodes")
    for s in range(1,nsamp_train+1):
        if (s % 50==0): print(' generating episode ' + num2str(s))
        sample = generate_episode_train(tabu_list)
        episode_to_file(os.path.join(mydir,'train',num2str(s)+'.txt'),sample)

def episode_to_file(fn_out,sample):

    assert not os.path.isfile(fn_out) # file must not already exist

    fid = open(fn_out,'w')

    ns = len(sample['xs'])
    fid.write('*SUPPORT*\n')    
    for s in range(ns):
        fid.write('IN: ')
        fid.write(' '.join(sample['xs'][s]))
        fid.write(' OUT: ')
        fid.write(' '.join(sample['ys'][s]))
        fid.write('\n')
    fid.write('\n')

    nq = len(sample['xq'])
    fid.write('*QUERY*\n')    
    for q in range(nq):
        fid.write('IN: ')
        fid.write(' '.join(sample['xq'][q]))
        fid.write(' OUT: ')
        fid.write(' '.join(sample['yq'][q]))
        fid.write('\n')
    fid.write('\n')

    fid.write('*GRAMMAR*\n')
    if 'grammar_str' in sample['aux']:
        grammar_str = str(sample['aux']['grammar_str'])
        fid.write(grammar_str)
    fid.close()

def num2str(x,mylen=6):
    # convert number x to string and add preceding zeros
    s = str(x)
    add = mylen -len(s)
    s = add*'0' + s
    return s

def get_episode_generator(episode_type):
    #  Returns function that generates episodes
    #
    # Input
    #  episode_type : string specifying type of episode
    #
    # Output
    #  generate_episode_train: function handle for generating episodes
    #  generate_episode_test: function handle for generating episodes
    if episode_type == 'algebraic':
        generate_episode_train = lambda tabu_episodes : generate_rules_episode(nsupport=14,nquery=10,nprims=4,nrules=3,tabu_list=tabu_episodes,maxlen=8)
        generate_episode_test = generate_episode_train
    else:
        raise Exception("episode_type is not valid")
    return generate_episode_train, generate_episode_test

def generate_rules_episode(nsupport,nquery,nprims,nrules,maxlen,tabu_list=[],max_try_novel=100):
    # Generate episode based on set of algebraic rules.
    #   ... first, sample algebraic rules of a meta-gramamr
    #   ... then, sample examples from this grammar
    #   ... then, randomly split examples into support and query sets
    # 
    # Input
    #  nsupport : number of support items
    #  nquery : number of query items
    #  nprims : number of unique primitives in each episode
    #  nrules : number of rules
    #  maxlen : Maximum number of input and output elements in a command/output. Also, maximum length of a rule RHS
    #  max_try_novel : number of attempts to find a novel episode (not in tabu list) before throwing an error
    if tabu_list: assert(isinstance(tabu_list[0],str)) # tabu list must be string "identifiers"
    ntotal = nsupport+nquery
    count = 0
    input_symbols = input_symbols_list_default
    output_symbols = output_symbols_list_default
    while True:
        G = generate_random_rules(nprims,nrules,input_symbols,output_symbols,max_len=maxlen)
        myhash = make_hashable(G)
        D = sample_examples(ntotal,G,maxlen_input=maxlen,maxlen_output=maxlen)
        np.random.shuffle(D)
        x_total = [d[0].split(' ') for d in D]
        y_total = [d[1].split(' ') for d in D]
        x_support = x_total[:nsupport]
        y_support = y_total[:nsupport]
        x_query = x_total[nsupport:]
        y_query = y_total[nsupport:]
        assert(len(x_query)==nquery)
        if (myhash not in tabu_list) and (not equiv_to_miniscan(str(G))):
            break
        count += 1
        if equiv_to_miniscan(str(G)): print('* Grammar equivalent to MiniSCAN * ')
        if count > max_try_novel:
            raise Exception('We were unable to generate an episode that is not on the tabu list')
    return bundle_biml_episode(x_support,y_support,x_query,y_query,myhash,aux={'grammar_str':str(G)})	

def generate_random_rules(nprims,nrules,input_symbols,output_symbols,max_len):
	#
	# Input
	#   nprims : number of primitives
	#   nrules : number of rules
	#   input_symbols : available input symbols for grammar (LHS of rules)
	#   output_symbols : available output symbols for mapping input primitives to output primitives
	#   max_len : maximum length of RHS of rules
	#
	assert(nprims+nrules <= len(input_symbols))
	input_symbol_options = deepcopy(input_symbols) 
	output_symbol_options = deepcopy(output_symbols)
	np.random.shuffle(input_symbol_options)
	np.random.shuffle(output_symbol_options)
	rules, input_symbol_options = generate_prims(nprims, input_symbol_options, output_symbol_options)
	for i in range(nrules):
		input_symbol = input_symbol_options[i]
		LHS,used_vars_lhs = sample_LHS(input_symbol)		
		RHS = sample_RHS(used_vars_lhs,max_len)
		rules.append(Rule(LHS,RHS))
	# once we have all the sampled rules, add a default concatenation rule
	rules.append(icon_concat_rule)
	return Grammar(rules,input_symbols)

def generate_prims(nprims,input_symbol_options,output_symbol_options):
	# Generate the rules for the primitive mappings
	# input
	#   nprims : number of primitives
	#   *_symbol_options : available input and output symbols
	#   
	# return :  rules and updated list of leftover input symbols
	rules = []
	for i in range(nprims):
		rules.append( Rule(input_symbol_options[i],output_symbol_options[i]) )
	return rules, input_symbol_options[nprims:]

def sample_LHS(func_name):
	# Sample a left-hand-side (LHS) that is either a one (x func_name) or two (x func_name x) argument rule
	# Return as string
	assert(isinstance(func_name,str))
	vars_options = random_stitch(vars_atom,vars_string)
	if flip(p_lhs_onearg):
		arr = [vars_options.pop(0), func_name]
	else:
		arr = [vars_options.pop(0), func_name, vars_options.pop(0)]	
	used_vars = [a for a in arr if a in vars_atom+vars_string]
	used_vars = ['['+u+']' for u in used_vars] # variables we used, with [ ] pre and postfix
	return ' '.join(arr), used_vars

def sample_RHS(vars_in_lhs, max_len, min_len=2):
	#  Sample a right-hand-side (RHS) for some arbitrary mix of the LHS variables.
	#    Note that the minimum length of the RHS must be two variables.
	#
	# Input
	#   vars_in_lhs : variables that were used to construct the left hand side
	#   max_len : maximum number of variables on the RHS
	#   min_len : minimum number of variables on the RHS
	#
	# Output: return as string
	arr = vars_in_lhs.copy()
	while True:
		if len(arr) >= max_len:
			break
		if len(arr) >= min_len and flip(p_stop_rhs):
			break
		item = random.choice(vars_in_lhs)
		arr.append(item)
	np.random.shuffle(arr) # randomize RHS order
	return ' '.join(arr)

def random_stitch(list1,list2):
	# Combine two lists by randomly picking an element from one, then from another, in order.
	new_list = []
	list1 = list1.copy()
	list2 = list2.copy()
	while len(list1)>0 and len(list2)>0:
		if flip():
			new_list.append(list1.pop(0))
		else:
			new_list.append(list2.pop(0))
	while len(list1)>0:
		new_list.append(list1.pop(0))
	while len(list2)>0:
		new_list.append(list2.pop(0))
	return new_list	

def sample_examples(nexamples,G,maxlen_input,maxlen_output,maxntry=10000):
	# Input
	#  nexamples : number of input sequences
	#  G : grammar
	#  maxlen_* : maximum length for input or output sequence
	#
	# Output
	#  D : list of examples (input/output pairs)
	CFG = make_pcfg_for_data_gen(G)
	D = set([])
	ntry = 0
	while len(D)<nexamples:
		dat_in = sample_from_pcfg(CFG, maxlen_input)
		dat_out = G.apply(dat_in)
		ntry += 1
		if (dat_in != '') and (len(dat_in.split()) <= maxlen_input) and (len(dat_out.split()) <= maxlen_output):
			D.add((dat_in, dat_out))
			ntry = 0
		if ntry > maxntry:
			raise Exception('Maximum number of tries when generating an episode')			
	return list(D)

def make_pcfg_for_data_gen(G):
	# Transform the rules into a PCFG that defines a dist. over valid input strings to create the data set

	LHS_list = [r.LHS_str for r in G.rules]
	LHS_list = [re.sub(r"\bu1\b","U",s) for s in LHS_list]
	LHS_list = [re.sub(r"\bu2\b","U",s) for s in LHS_list]
	LHS_list = [re.sub(r"\bx1\b","X",s) for s in LHS_list]
	LHS_list = [re.sub(r"\bx2\b","X",s) for s in LHS_list]
	CFG = {}
	CFG['U'] = [s for s in LHS_list if len(s.split())==1]
	CFG['X'] = ['U'] + [s for s in LHS_list if len(s.split())>1]
	return CFG

def sample_from_pcfg(CFG,maxlen): 
	#  CFG : context-free grammar we want to sample from
	#  maxlen : maximum length of sampled string
	# 
	# If we sample a string that is too long, we return an empty string
	#
	mystr = 'X' # starting symbol
	while True: 
		list_expand = [] # expansion of each current symbol
		all_term = True # all terminals?
		for symb in mystr.split():						
			if symb in CFG:
				all_term = False # we made an expansion
				options = CFG[symb]
				symb = random.choice(options)
			list_expand.append(symb)

		# if we are over the allowed length
		if len(list_expand) > maxlen:
			return ''

		mystr = ' '.join(list_expand)
		if all_term:
			break
	return mystr

if __name__ == "__main__":

	# For generating the entire meta-training dataset in the paper
	seed_all()
	generate_rules_dataset(100000,200,mydir='data_algebraic',episode_type='algebraic')