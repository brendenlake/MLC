import sys
sys.path.append('..')
sys.path.append('../..')
import os
import datasets as dat
from analysis_lib import *
import numpy as np
from copy import deepcopy

# Analyze freeform task, and produce HTML file with model outputs.
# Compare human and machine samples.

def analyze_mapping(sample):
	# augment with annotations	
	xs = sample['xs']
	ys = sample['ys']	
	sample['perfect'] = is_perfect_map(xs,ys)
	sample['one2one'] = is_one2one(xs,ys)
	sample['iconic'] = is_iconic_concat(xs,ys)
	sample['ME'] = is_ME(xs,ys)

	# samples sorted in canonical order
	xs_abs = reconstruct_abs_input(xs)
	tuples = zip(xs,ys,xs_abs)
	tuples = sorted(tuples, key=lambda x : x[2])
	sample['xs_sorted'],sample['ys_sorted'],_ = zip(*tuples)
	return sample

def is_perfect_map(xs,ys):
	# See if input/output mapping follows all three inductive biases
	# Input
	#  xs : list of input commands (strings)
	#  ys : list of outputs (strings)
	# Output
	#  true/false
	n = len(xs)
	mydict = {}
	for i in range(n): # for each pattern
		in_list = xs[i].split()
		out_list = ys[i].split()
		if len(in_list) != len(out_list): # violation of one-to-one
			return False
		for idx,item in enumerate(in_list):
			if item not in mydict: # if first encounter, record input-to-output mapping for a symbol
				mydict[item] = out_list[idx] 
			if mydict[item] != out_list[idx]: # on next encounter, check input-to-output mapping
				return False # violation of consistency
	v = mydict.values()	
	if len(v) > len(set(v)):
		return False # violation of mutual exclusivity.. two words map to the same symbol
	return True

def is_one2one(xs,ys):
	# Verify each input/output pair has the same number of symbols.
	# Input
	#  xs : list of input commands (strings)
	#  ys : list of outputs (strings)
	# Output
	#  true/false
	n = len(xs)
	for i in range(n):
		if len(xs[i].split()) != len(ys[i].split()):
			return False
	return True

def is_ME(xs,ys):
	# Verify each command maps to a unique output
	# Input
	#  xs : list of input commands (strings)
	#  ys : list of outputs (strings)
	# Output
	#  true/false
	return len(ys)==len(set(ys))

def is_iconic_concat(xs,ys):
	# Verify the mapping is consistent and left-to-right. 
	# We can check this by trying to infer a dictionary.
	# Input
	#  xs : list of input commands (strings)
	#  ys : list of outputs (strings)
	# Output
	#  true/false
	xs_abs = reconstruct_abs_input(xs)
	mydict = infer_dict(xs_abs, ys)
	return len(mydict)>0

## possible items:
# 'p1',
# 'p1 p1',
# 'p1 p2',
# 'p3 p1',
# 'p1 p4 p5',
# 'p1 p4 p1',
# 'p5 p4 p1',
def reconstruct_abs_input(raw_input):
	# Translate raw inputs ('blicket dax') to abstract form (e.g., "p3 p1")
	abs_input = ['' for r in raw_input]
	mylens = [len(s.split(' ')) for s in raw_input]
	raw_input_list = [s.split(' ') for s in raw_input]

	# find the item which is a single word
	idx_p1 = mylens.index(1)
	abs_input[idx_p1] = 'p1'
	raw_prim = raw_input[idx_p1]

	# find 'p1 p1', 'p1 p2', and 'p3 p1'
	for idx,r in enumerate(raw_input_list):
		if len(r) == 2:
			if (r[0] == raw_prim and r[1] == raw_prim):
				abs_input[idx] = 'p1 p1'
			elif (r[0] == raw_prim):
				abs_input[idx] = 'p1 p2'
			elif (r[1] == raw_prim):
				abs_input[idx] = 'p3 p1'
			else:
				assert False, "reconstruction of abs_input failed"

	# find 'p1 p4 p5', 'p1 p4 p1', 'p5 p4 p1',
	for idx,r in enumerate(raw_input_list):
		if len(r) == 3:
			if (r[0] == raw_prim and r[2] == raw_prim):
				abs_input[idx] = 'p1 p4 p1'
			elif (r[0] == raw_prim):
				abs_input[idx] = 'p1 p4 p5'
			elif (r[2] == raw_prim):
				abs_input[idx] = 'p5 p4 p1'
			else:
				assert False, "reconstruction of abs_input failed"

	is_filled = [a != '' for a in abs_input]
	if (not all(is_filled)):
		assert False, "reconstruction of abs_input failed"

	return abs_input

## possible items:
# 'p1',
# 'p1 p1',
# 'p1 p2',
# 'p3 p1',
# 'p1 p4 p5',
# 'p1 p4 p1',
# 'p5 p4 p1',
def infer_dict(xs_abs, ys):
	# Assumes that participants are following concatenation assumption
	# If they are, return a dictionary of the strings for each response
	# If not, return an empty dictionary
	#
	#   xs_abs : list of abstract commands (strings)
	#   ys: list of responses (strings)
	mydict = {}

	# extract p1
	idx_p1 = xs_abs.index('p1')
	mydict['p1'] = ys[idx_p1]
	myprim = mydict['p1']

	idx_p1p1 = xs_abs.index('p1 p1')
	if (ys[idx_p1p1] != ' '.join([myprim, myprim])):
		return {}

	# extract p2
	idx_p1p2 = xs_abs.index('p1 p2')
	if (ys[idx_p1p2][:len(myprim)] != myprim):
		return {}
	mydict['p2'] = ys[idx_p1p2].replace(myprim,'').strip()

	# extract p3
	idx_p3p1 = xs_abs.index('p3 p1')
	if (ys[idx_p3p1][-len(myprim):] != myprim):
		return {}
	mydict['p3'] = ys[idx_p3p1].replace(myprim,'').strip()

	# extract p4
	idx_p1p4p1 = xs_abs.index('p1 p4 p1')
	if (ys[idx_p1p4p1][:len(myprim)] != myprim or ys[idx_p1p4p1][-len(myprim):] != myprim):
		return {}
	mydict['p4'] = ys[idx_p1p4p1].replace(myprim,'').strip()

	# extract p5
	idx_p1p4p5 = xs_abs.index('p1 p4 p5')
	tag = ' '.join([myprim, mydict['p4']])
	if (ys[idx_p1p4p5][:len(tag)] != tag):
		return {}
	mydict['p5'] = ys[idx_p1p4p5].replace(tag,'').strip()

	# confirm p5
	idx_p5p4p1 = xs_abs.index('p5 p4 p1')
	tag = ' '.join([mydict['p5'], mydict['p4'], myprim])
	if (ys[idx_p5p4p1] != tag):
		return {}

	return mydict

def print_pairs(x,y,annotation,fid):
	# Input
	#  x: list of input commands (strings)
	#  y: list of outputs (strings)
	#  annotation: text to include as a note
	#  fid : file handle for writing
	fid.write('[\n')
	fid.write(" '"+annotation+"',\n")
	m = len(x)
	for j in range(m):
		fid.write('  [')
		fid.write("'" + x[j] + "', ")
		fid.write("'" + y[j] + "'")
		fid.write('],\n')
	fid.write('],\n')

def write_to_js(samples,fid):
	# Help create HTML file
	#  samples : list of dicts (outputs of process_episode)
	#  fid : file handle
	n = len(samples)
	fid.write('var all_data = [')
	for i in range(n):
		sample = samples[i]
		annotation = ''
		if sample['one2one']: annotation += 'one2one; '
		if sample['iconic'] : annotation += 'iconic; '
		if sample['ME']     : annotation += 'ME'
		print_pairs(sample['xs_sorted'],sample['ys_sorted'],'support',fid)
		# print_pairs(sample['xs'],sample['ys'],'support',fid)
		print_pairs([],[],annotation,fid)
	fid.write('];\n')

def get_human_samples():
	# Load human data for open ended task
	prevdir = os.getcwd()
	os.chdir('../..')
	D_train, _ = dat.get_dataset('open_end_human_all')
	n = len(D_train.list_items)
	samples = [D_train.get_raw_item(i) for i in range(n)]
	os.chdir(prevdir)
	for sample in samples:
		sample['xs'] = list(map(lambda x: ' '.join(x), sample['xq']))
		sample['ys'] = list(map(lambda x: ' '.join(x), sample['yq']))
		sample['xq'] = [] # place data in support rather than query
		sample['yq'] = []
	return samples

if __name__ == "__main__":

	fn_in = 'open_end_freeform_net-BIML-open-ended-top.txt'
	model_tag = fn_in.replace('.txt','')
	fn_out_html = model_tag + '.html'
	fn_out_human = 'human_open_end_freeform.html'

	model_samples = get_all_episodes(fn_in)
	model_samples = [analyze_mapping(sample) for sample in model_samples]
	n_model = len(model_samples)
	get_mean_model = lambda field : round(100.*np.mean([s[field] for s in model_samples]),3)
	get_count_model = lambda field : np.sum([s[field] for s in model_samples])
	
	human_samples = get_human_samples()
	human_samples = [analyze_mapping(sample) for sample in human_samples]
	n_human = len(human_samples)
	get_mean_human = lambda field : round(100.*np.mean([s[field] for s in human_samples]),3)
	get_count_human = lambda field : np.sum([s[field] for s in human_samples])

	print('Human:')
	print('   Processing',n_human,'human participants.')
	print('   Percent with perfect maps (consistent with 3 inductive biases):',
		get_mean_human('perfect'),'; N=',get_count_human('perfect'),'of',n_human)
	print('   Percent with one2one maps:',
		get_mean_human('one2one'),'; N=',get_count_human('one2one'),'of',n_human)
	print('   Percent with iconic concatenation:',
		get_mean_human('iconic'),'; N=',get_count_human('iconic'),'of',n_human)
	print('   Percent with ME maps:',
		get_mean_human('ME'),'; N=',get_count_human('ME'),'of',n_human)

	print('Model:')
	print('   Processing',n_model,'model samples.')
	print('   Percent with perfect maps (consistent with 3 inductive biases):',
		get_mean_model('perfect'),'; N=',get_count_model('perfect'),'of',n_model)
	print('   Percent with one2one maps:',
		get_mean_model('one2one'),'; N=',get_count_model('one2one'),'of',n_model)
	print('   Percent with iconic concatenation:',
		get_mean_model('iconic'),'; N=',get_count_model('iconic'),'of',n_model)
	print('   Percent with ME maps:',
		get_mean_model('ME'),'; N=',get_count_model('ME'),'of',n_model)
		
	# Create human HTML file
	print('Generating HTML file:',fn_out_human)
	with open('template.html','r') as fid_in:
		mylines = fid_in.readlines()
	with open(fn_out_human,'w') as fid_out:		
		for l in mylines:
			fid_out.write(l)
			if l.strip() == '// PLACEHOLDER':
				fid_out.write('var title="Human participants"; \n')          
				write_to_js(human_samples,fid_out)	

	# Create model HTML file
	with open('template.html','r') as fid_in:
		mylines = fid_in.readlines()
	with open(fn_out_html,'w') as fid_out:
		print('Generating HTML file:',fn_out_html)		
		for l in mylines:
			fid_out.write(l)
			if l.strip() == '// PLACEHOLDER':
				fid_out.write('var title="'+model_tag+'"; \n')          
				write_to_js(model_samples,fid_out)