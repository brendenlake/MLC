import sys
sys.path.append('..')
sys.path.append('../..')
import os
import datasets as dat
from analysis_lib import *
import numpy as np
from copy import deepcopy
import scipy

# Analyze few-shot learning task. Compare human and machine samples.

map_gold_simple = { # simple queries
			"3 after DAX" : "DAX 3",
			"DAX after 1" : "1 DAX",
			"DAX thrice" : "DAX DAX DAX",
			"1 surround DAX" : "1 DAX 1",
			"DAX surround 2" : "DAX 2 DAX"}

map_gold_complex = {# complex queries
			"2 after 3 surround DAX" : "3 DAX 3 2",
			"DAX thrice after 2" : "2 DAX DAX DAX",
			"3 after DAX thrice" : "DAX DAX DAX 3",
			"DAX surround DAX after DAX thrice" : "DAX DAX DAX DAX DAX DAX",
			"DAX surround 3 after 1 thrice" : "1 1 1 DAX 3 DAX"}

map_gold_long = {# 3 function compositions
			"DAX surround DAX after DAX thrice" : "DAX DAX DAX DAX DAX DAX",
			"DAX surround 3 after 1 thrice" : "1 1 1 DAX 3 DAX"}

map_prims = {'1':'1', '2':'2', '3':'3', 'DAX':'DAX'}

map_gold = {**map_gold_simple, **map_gold_complex}
simple_qs = map_gold_simple.keys()
complex_qs = map_gold_complex.keys()
long_qs = map_gold_long.keys()
list_prims = map_prims.keys()

def analyze_errors(my_samples):
	# Percent of errors that are one2one and iconic in nature
	all_errors = [sample['errors'] for sample in my_samples]
	is_one2one = [sample['errors_is_one2one'] for sample in my_samples]
	is_iconic = [sample['errors_is_iconic'] for sample in my_samples if 'errors_is_iconic' in sample]
	all_errors = sum(all_errors,[])
	is_one2one = sum(is_one2one,[])
	is_iconic = sum(is_iconic,[])
	print('  perc. of errors that are one2one: ', round(100.*np.mean(is_one2one),3),';',np.sum(is_one2one),'of',len(is_one2one))
	print('  perc. of errors (involving "after") that are iconic : ', round(100.*np.mean(is_iconic),3),';',np.sum(is_iconic),'of',len(is_iconic))

def analyze_mapping(sample):
	# Each episode (sample) should contain only a single query type
	command = sample['command']
	yq = sample['yq']
	target = sample['target']
	nq = len(yq)
	
	is_correct = []
	list_errors = []
	for j in range(nq): # all predictions
		prediction = yq[j]
		my_correct = prediction==target
		is_correct.append(my_correct)
		if not my_correct: list_errors.append(prediction)

	is_one2one = []
	is_iconic = []
	for prediction_err in list_errors: # all errors
		is_one2one.append(is_one2one_error(command,prediction_err))
		if sample['involves_after']:
			is_iconic.append(is_iconic_error(command,prediction_err,target))

	sample['is_correct'] = is_correct
	sample['acc'] = 100.*np.mean(is_correct)
	sample['errors'] = list_errors
	sample['errors_is_one2one'] = is_one2one
	if sample['involves_after']: sample['errors_is_iconic'] = is_iconic
	return sample

def is_one2one_error(command,prediction):
	# These errors are defined as responses such that the input and
	# output sequence have the same length, and each input primitive is replaced
	# with its provided output symbol. Function words are replaced with an arbitrary output symbol.
	# 
	# Input
	#  command : string
	#  prediction : string
	command_list = command.split()
	pred_list = prediction.split()
	if len(command_list)!=len(pred_list): return False
	m = len(command_list)
	for i in range(m):
		c = command_list[i]
		p = pred_list[i]
		if c in list_prims:
			if map_prims[c] != p:
				return False
	return True

def is_iconic_error(command,prediction,target):
	# Did participant apply the 'after' command without reversing order of arguments?
	#
	# Input
	#  command : string
	#  prediction : string
	command_list = command.split()
	pred_list = prediction.split()
	target_list = target.split()
	assert('after' in command_list), "iconic errors can only relate to the function 'after'"
	return pred_list[::-1] == target_list # note that simple list reverse works for MiniSCAN,
										  # because all sub-commands of complex commands are symmetric.
										  # this does not work in general.

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
	for sample in samples:
		myresponses = sample['yq']
		assert(isinstance(myresponses[0],str))
		unique_responses = sorted(set(myresponses))
		count_unique = []
		for u in unique_responses:
		    count_unique.append(np.sum(np.array([u == rr for rr in myresponses],dtype=int)))
		unique_responses = [x for x in sorted(zip(unique_responses,count_unique), key=lambda pair: pair[1], reverse=True)]
		fid.write('[ \n')
		for u in unique_responses:
			fid.write("  ['%s', '%s', %d], \n" % (sample['command'], u[0], u[1]) )
		fid.write('], \n')
	fid.write('];\n')

def get_human_responses():
	# Load human data for few-shot learning task
	prevdir = os.getcwd()
	os.chdir('../..')
	_, D_val = dat.get_dataset('few_shot_human')
	n = len(D_val.list_items)
	samples = [D_val.get_raw_item(i) for i in range(n)]
	os.chdir(prevdir)
	for sample in samples:
		sample['xs'] = []
		sample['ys'] = []
		sample['xq'] = list(map(lambda x: ' '.join(x), sample['xq']))
		sample['yq'] = list(map(lambda x: ' '.join(x), sample['yq']))
	return samples

def process_episode(sample):
	del sample['xs'] # clear support, since it's not analyzed
	del sample['ys']
	xq = sample['xq']
	assert(len(set(xq))==1), "check that the episode only contains one query type"
	sample['command'] = xq[0]
	sample['target'] = map_gold[sample['command']]
	sample['involves_after'] = 'after' in sample['command'].split() # command uses 'after'
	del sample['xq']
	return sample

def debug(): # unit tests
	assert not is_iconic_error("2 after 3 surround DAX" ,"3 DAX 3 2","3 DAX 3 2")
	assert is_iconic_error("2 after 3 surround DAX" ,"2 3 DAX 3","3 DAX 3 2")
	assert is_iconic_error("DAX surround 3 after 1 thrice","DAX 3 DAX 1 1 1","1 1 1 DAX 3 DAX")
	assert is_one2one_error("2 after 3 surround DAX", "2 1 3 2 DAX")
	assert not is_one2one_error("2 after 3 surround DAX", "2 1 3 2 2")
	assert is_one2one_error("DAX surround 3 after 1 thrice","DAX Q 3 Q 1 Q")
	print('*PASSED* tests for iconic and one2one')

if __name__ == "__main__":
	fn_in = 'few_shot_human_mult10_net-BIML-top.txt' # input text file with  model samples
	fn_out_human = 'human_few_shot_behavior.html' # output html file with human responses

	human_responses = get_human_responses()
	human_responses = [process_episode(sample) for sample in human_responses]
	human_responses = [analyze_mapping(sample) for sample in human_responses]	

	model_responses = get_all_episodes(fn_in)
	model_responses = [process_episode(sample) for sample in model_responses]
	model_responses = [analyze_mapping(sample) for sample in model_responses]

	print('Human responses (item accuracy):')
	h_commands, h_acc = [],[]
	for sample in human_responses:		
		print('  ',sample['command'],':',round(sample['acc'],3))
		h_commands.append(sample['command'])
		h_acc.append(sample['acc'])
	print('  ---')
	print('  mean overall acc.:',round(np.mean(h_acc),3))
	print('  mean acc. on simple queries:',round(np.mean([h_acc[i] for i,command in enumerate(h_commands) if command in simple_qs]),3))
	print('  mean acc. on complex queries:',round(np.mean([h_acc[i] for i,command in enumerate(h_commands) if command in complex_qs]),3))
	print('  mean acc. on len=6 queries:',round(np.mean([h_acc[i] for i,command in enumerate(h_commands) if command in long_qs]),3))
	analyze_errors(human_responses)

	print('Model responses (item accuracy):')
	m_commands, m_acc = [],[]
	for sample in model_responses:
		print('  ',sample['command'],':',round(sample['acc'],3))
		m_commands.append(sample['command'])
		m_acc.append(sample['acc'])
	print('  ---')
	print('  mean overall acc.:',round(np.mean(m_acc),3))
	print('  mean acc. on simple queries:',round(np.mean([m_acc[i] for i,command in enumerate(m_commands) if command in simple_qs]),3))
	print('  mean acc. on complex queries:',round(np.mean([m_acc[i] for i,command in enumerate(m_commands) if command in complex_qs]),3))
	print('  mean acc. on len=6 queries:',round(np.mean([m_acc[i] for i,command in enumerate(m_commands) if command in long_qs]),3))
	analyze_errors(model_responses)

	assert(h_commands==m_commands)
	r,p = scipy.stats.pearsonr(h_acc,m_acc)
	print('\nCorrelation for item accuracies: r=',round(r,3),'; p=',round(p,3))
	r,p = scipy.stats.pearsonr(h_acc,[len(map_gold[c].split()) for c in h_commands])

	# Create human HTML file
	print('Generating HTML file:',fn_out_human)
	with open('template.html','r') as fid_in:
		mylines = fid_in.readlines()
	with open(fn_out_human,'w') as fid_out:		
		for l in mylines:
			fid_out.write(l)
			if l.strip() == '// PLACEHOLDER':
				fid_out.write('var title="Human participants"; \n')          
				write_to_js(human_responses,fid_out)