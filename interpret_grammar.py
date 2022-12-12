from copy import deepcopy
from re import fullmatch

# Interpretation grammar for re-writing input sequences as output sequences.
#   - Each grammar is a list of algebraic re-write rules.
#   - Rules are applied in sequential order (first come, first serve)

def is_prim_var(s):
	# detect if string is a primitive name, such as 'u12' (u followed by optional number)
	# ignore the interpretation symbols	
	s = int_strip(s)
	pattern = fullmatch('u[0-9]*',s)
	return bool(pattern)

def is_var(s):
	# detect if string is a variable name, such as 'x12' (x followed by optional number)
	# ignore the interpretation symbols	
	s = int_strip(s)
	pattern = fullmatch('x[0-9]*',s)
	return bool(pattern)

def int_strip(s):
	# Strip the interpretation symbols, [ and ],
	# from the beginning and end of strings, respectively
	s = s.lstrip('[')
	s = s.rstrip(']')
	return s

def to_interpet(s):
	# Does string s still need interpretation?
	#  i.e., does it start with [ and end with ]?
	return s[0] == '[' and s[-1] == ']'

def str_to_grammar(s, input_symbols):
	# Interpret string specification as a grammar object
	# 
	# e.g., 
	#   lug -> YELLOW \n x2 lug -> [x2] \n ...
	#
	# input
	#   s : string description of grammar
	#   input_symbols : possible input primitives (could include more than grammar does)
	#
	# return
	#   Grammar object, or [] if we cannot create a valid grammar with this specification
	lines = s.split('\n')
	lines = [l.strip() for l in lines]
	lines = [l for l in lines if len(l) > 0]
	rules = []
	for l in lines:
		if "->" not in l:
			return []
		sides = l.split("->")
		if len(sides) != 2:
			return []
		LHS = sides[0].strip()
		RHS = sides[1].strip()
		if len(LHS)==0 or len(RHS)==0:
			return []
		R = Rule(LHS,RHS)
		rules.append(R)
	G = Grammar(rules, input_symbols)
	if not G.var_match(): # make sure none of the rules drop a variable..
		# thus variables consistent on LHS and RHS
		return []
	return G

class Grammar():

	max_recursion = 50 # maximum number of recursive calls
	count_recursion = 0 # recursive counters
	rules = []

	def __init__(self,rules, list_prims):
		#
		# Input
		#  rules is a list of Rule objects
		#  list_prims : list of input symbols (cannot contain numbers)
		self.rules = deepcopy(rules)
		self.list_prims = list_prims
		for r in self.rules:
			r.set_primitives(list_prims)

	def var_match(self):
		# Check if all the variables match, such that those in RHS are also in LHS
		return all([r.valid_var_match for r in self.rules])

	def apply(self,s):
		# apply re-write rules recursively to string s
		self.count_recursion = 0
		return self.__apply_helper(s)

	def __apply_helper(self,s):
		self.count_recursion += 1
		valid = []
		myrule = None
		for r in self.rules:
			valid.append(r.applies(s))
		if not any(valid):
			return s
		myrule = self.rules[valid.index(True)] # pick first rule

		# run 'apply' recursively
		out = myrule.apply(s) # out is a list of partially-processed sub-strings
		for idx,o in enumerate(out):
			if to_interpet(o) and self.count_recursion < self.max_recursion: # non-terminals
				out[idx] = self.__apply_helper(int_strip(o))
		return ' '.join(out)

	def __str__(self):
		s = ''
		for r in self.rules:
			s += str(r)+' \n '
		return s

class Rule():

	# left-hand-side
	LHS_str = ''
	LHS_list = []
	LHS_regexp = ''

	# right-hand-side
	RHS_str = ''
	RHS_list = []

	# 
	valid_var_match = False # verifies that every variable in RHS is also in the LHS
	var_regexp = '([ a-zA-Z0-9]+)' # define acceptable string for a variable to hold
	
	def __init__(self,LHS,RHS):
		# LHS : string with variables (no interpretation symbols [ or ] )
		# RHS : string with variables (can have interpretation symbols for recursive computation)
		assert(LHS.strip()==LHS)
		assert(RHS.strip()==RHS)
		self.LHS_str = LHS
		self.LHS_list = LHS.split()
		self.RHS_str = RHS
		self.RHS_list = RHS.split()

	def set_primitives(self,list_prims):
		# Create regular expressions for detecting matches
		#  list_prims : list of the primitive input symbols
		self.list_prims = list_prims
		self.prim_regexp = '(' + '|'.join(self.list_prims) + ')'  # define acceptable string for a primitive

		# get list of all variables in LHS (and for valid rules, the same vars should be in the RHS)
		self.vars = [v for v in self.LHS_list if is_prim_var(v) or is_var(v)] # LHS vars
		rhs_vars  = [int_strip(v) for v in self.RHS_list if is_prim_var(v) or is_var(v)] # RHS vars
		self.valid_var_match = all([v in self.vars for v in rhs_vars]) # check that RHS variables are in LHS

		# Compute the regexp for checking whether the rule is active
		mylist = deepcopy(self.LHS_list)
		for i,x in enumerate(mylist):
			if is_prim_var(x):
				mylist[i] = self.prim_regexp
			elif is_var(x):
				mylist[i] = self.var_regexp
		self.LHS_regexp = ' '.join(mylist)

	def applies(self,s):
		# return True if the re-write rule applies to this string
		return self.valid_var_match and bool(fullmatch(self.LHS_regexp,s))

	def apply(self,s):
		# Apply rule to string s
		# Return resulting string as list
		assert self.applies(s)
		assert self.valid_var_match

		# extract variables from LHS
		m = fullmatch(self.LHS_regexp,s)
			# if the expression has two variables "x1 x2", it returns the first split
		mygroups = m.groups()
		assert(len(mygroups) == len(self.vars))
		vdict = dict(zip(self.vars,mygroups))

		# replace RHS with variable values
		mylist = deepcopy(self.RHS_list)
		for i,x in enumerate(mylist):
			if is_var(x) or is_prim_var(x):				
				mylist[i] = '['+vdict[int_strip(x)]+']'
		return mylist

	def __str__(self):
		if self.valid_var_match:
			val_tag = ''
		else:
			val_tag = ' (invalid)'
		return str(self.LHS_str) + ' -> ' + str(self.RHS_str) + val_tag

def get_grammar_miniscan():
	# Generates the grammar used for testing humans on few-shot learning
	S_miniscan = "1 -> 1 \n 3 -> 3 \n 2 -> 2 \n DAX -> DAX \n u1 thrice -> [u1] [u1] [u1] \n u1 surround u2 -> [u1] [u2] [u1] \n x1 after x2 -> [x2] [x1] \n u1 x1 -> [u1] [x1]"
	return str_to_grammar(S_miniscan,['1','2','3','DAX','thrice','surround','after'])


if __name__ == "__main__":
 	# Test code

	myrules = [Rule('walk','WALK'), Rule('u left','LTURN [u]'), Rule('x twice','[x] [x]')]
	G = Grammar(myrules,['walk','left'])
	mycommand = 'walk left twice'
	myoutput = G.apply(mycommand)
	mytarget = 'LTURN WALK LTURN WALK'
	print('Testing command:',mycommand)
	assert(myoutput==mytarget)
	print('*PASSED* Grammar.apply worked to produce output:',myoutput)

	print("")
	mycommand = 'walk left twice twice'
	myoutput = G.apply(mycommand)
	mytarget = 'LTURN WALK LTURN WALK LTURN WALK LTURN WALK'
	print('Testing command:',mycommand)
	assert(myoutput==mytarget)
	print('*PASSED* Grammar.apply worked to produce output:',myoutput)

	print("")
	print('Testing variable detector..')
	assert is_var('x')
	assert is_var('x0')
	assert is_var('x10')
	assert is_var('x101')
	assert not is_var('X10')
	assert not is_var('10x')
	assert not is_var('y10')
	assert not is_var(' x10')
	assert not is_var('x10 ')
	assert not is_var('x0 x10')
	assert not is_var('x0 x10')
	assert not is_var('[x0] [x10]')
	print('*PASSED* Variable detector tests.')

	print("")
	input_symbols_list_default = ['dax', 'lug', 'wif', 'zup', 'fep', 'blicket', 'kiki', 'tufa', 'gazzer']
	S_orig = "lug -> GREEN \n gazzer -> RED \n wif -> PURPLE \n fep -> BLUE \n u1 tufa x1 -> [u1] [x1] \n u1 dax -> [u1] [u1] \n u1 kiki x1 -> [u1] [u1] [x1] \n u1 x1 -> [u1] [x1] \n"
	print("Original string...")
	print(S_orig)
	print("")
	G = str_to_grammar(S_orig,input_symbols_list_default)
	S_recon = str(G)
	f_compare = lambda X : [x.strip() for x in X.split('\n') if len(x.strip())>0]
	print('Reconstructed string from grammar...')
	print(S_recon)
	assert(f_compare(S_recon)==f_compare(S_orig))
	print('*PASSED* Grammar construction from string, to grammar, back to string.')

	mycommand="lug kiki lug dax"
	mytarget="GREEN GREEN GREEN GREEN"
	myoutput = G.apply(mycommand)
	print('\nTesting command:',mycommand)
	assert(myoutput==mytarget)
	print('*PASSED* Grammar.apply worked to produce output:',myoutput)

	G_miniscan = get_grammar_miniscan()
	D_miniscan_support = {'1':'1', '3':'3', '2':'2', 'DAX':'DAX', '2 after 3':'3 2', '1 after 2':'2 1', '2 thrice':'2 2 2', '2 surround 3':'2 3 2', '1 thrice':'1 1 1', '3 surround 1':'3 1 3', '2 thrice after 3':'3 2 2 2', '3 after 1 surround 2':'1 2 1 3', '2 after 3 thrice':'3 3 3 2', '3 surround 1 after 2':'2 3 1 3'}
	D_miniscan_query = {"3 after DAX" : "DAX 3", "DAX after 1" : "1 DAX", "DAX thrice" : "DAX DAX DAX", "1 surround DAX" : "1 DAX 1", "DAX surround 2" : "DAX 2 DAX", "2 after 3 surround DAX" : "3 DAX 3 2", "DAX thrice after 2" : "2 DAX DAX DAX", "3 after DAX thrice" : "DAX DAX DAX 3", "DAX surround DAX after DAX thrice" : "DAX DAX DAX DAX DAX DAX", "DAX surround 3 after 1 thrice" : "1 1 1 DAX 3 DAX"}
	D_miniscan = {**D_miniscan_support, **D_miniscan_query}
	print('\nTesting MiniSCAN grammar...')
	for mycommand in D_miniscan.keys():
		myoutput = G_miniscan.apply(mycommand)
		mytarget = D_miniscan[mycommand]
		print('     ',mycommand,'->',myoutput,'(target:',mytarget,')')
		assert(myoutput==mytarget)
	print('*PASSED* Grammar.apply worked to produce MiniSCAN targets')