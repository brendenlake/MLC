def get_all_episodes(fn_in):
	# Input
	#   fn_in : input file
	#
	# Output
	#   samples : list of episodes (each has 'xs','ys','xq','yq' structure)

	# Find range where episodes are present
	with open(fn_in,'r') as fid:
		lines = fid.readlines()
		lines = [l.strip() for l in lines]
		assert('Evaluation episode 0' in lines[0]), "first line should 'Evaluation episode 0'"

	# Find breaks between different episodes
	idx_eval_break = []
	for i,l in enumerate(lines):
		if 'Evaluation episode' in l:
			idx_eval_break.append(i)

	# Group file by episode
	episodes = []
	for i in range(len(idx_eval_break)-1):
		idx_start = idx_eval_break[i]
		idx_end = idx_eval_break[i+1]
		episodes.append(lines[idx_start:idx_end])		
	episodes.append(lines[idx_eval_break[-1]:]) # get the last episodes
	n_episodes = len(episodes)
	
	# Process the episodes
	samples = [process_episode(sample) for sample in episodes]

	# Get filenames for each episode (if applicable)
	filenames = [lines[idx] for idx in idx_eval_break]
	mytag = 'filename:'
	if all([mytag in f for f in filenames]):
		filenames = [f[f.index(mytag)+len(mytag):].strip() for f in filenames]
		for idx,s in enumerate(samples):
			s['aux'] = {}
			s['aux']['filename'] = filenames[idx]
	return samples

def get_solo_index(mylines,flag):
	# Find a string that contains flag as a substring
	# (there should only be one, otherwise throw an error)
	#
	# Input:
	#  mylines: list of strings
	#  flag : string
	#
	# Output
	#  index in mylines
	assert(all([isinstance(l,str) for l in mylines]))
	idx = [i for i,l in enumerate(mylines) if flag in l]
	assert(len(idx)==1) # failure to find the flag in mylines
	return idx[0]

def filter_blanks(mylines):
	# Filter out any empty strings
	# Input
	#   mylines: list of strings
	assert(all([isinstance(l,str) for l in mylines]))
	return [l for l in mylines if (len(l) > 0 and 'no patterns' not in l)]

def filter_targets(mylines):
	# Filter right-hand-side of any string that includes '(** target:'...
	# Input
	#   mylines: list of strings
	for i in range(len(mylines)):
		if '(** target:' in mylines[i]:
			idx_cut = mylines[i].find('(** target:')
			mylines[i] = mylines[i][:idx_cut].strip()
	return mylines

def process_episode(mylines):
	# Break down an episode into support, retrieval, and generalization examples
	idx_episode_start = get_solo_index(mylines,'Evaluation episode')
	idx_support = get_solo_index(mylines,'support items;')
	idx_retrieval = get_solo_index(mylines,'retrieval items;')
	idx_gen = get_solo_index(mylines,'generalization items;')
	lines_support = filter_blanks(mylines[idx_support+1:idx_retrieval])
	lines_gen = filter_blanks(mylines[idx_gen+1:])
	lines_support = filter_targets(lines_support)
	lines_gen = filter_targets(lines_gen)
	xs,ys = get_pairs(lines_support)
	xq,yq = get_pairs(lines_gen)	
	return {'xs':xs, 'ys':ys, 'xq':xq, 'yq':yq}

def get_pairs(mylines):
	#  Break each line into "command" and "output",
	#   assuming each has format "command -> output"
	# Output
	#   x: list of commands (as str)
	#   y: list of outputs (as str)	
	assert all(['->' in l for l in mylines])
	pairs = [l.split('->') for l in mylines]
	pairs = [[p[0].strip(), p[1].strip()] for p in pairs]
	x = y = []
	if len(pairs)>0:
		x,y = zip(*pairs)
	return x,y