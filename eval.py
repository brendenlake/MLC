import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions.categorical import Categorical
import os
import sys
import argparse
import numpy as np
import math
from copy import deepcopy
from model import BIML, describe_model
import datasets as dat
from train_lib import seed_all, extract, display_input_output, assert_consist_langs
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Evaluate a pre-trained model

def evaluate_ll(val_dataloader, net, langs, loss_fn=[], p_lapse=0.0, verbose=False):
    # Evaluate the total (sum) log-likelihood across the entire validation set
    # 
    # Input
    #   val_dataloader : 
    #   net : BIML model
    #   langs : dict of dat.Lang classes
    #   p_lapse : (default 0.) combine decoder outputs (prob 1-p_lapse) as mixture with uniform distribution (prob p_lapse)
    net.eval()
    total_N = 0
    total_ll = 0
    if not loss_fn: loss_fn = torch.nn.CrossEntropyLoss(ignore_index=langs['output'].PAD_idx)
    for batch_idx, val_batch in enumerate(val_dataloader):
        val_batch = dat.set_batch_to_device(val_batch)
        dict_loss = batch_ll(val_batch, net, loss_fn, langs, p_lapse=p_lapse)
        total_ll += dict_loss['ll']
        total_N += dict_loss['N']
    return total_ll, total_N

def evaluate_acc(val_dataloader, net, langs, max_length, eval_type='max', verbose=False):
    # Evaluate accuracy (exact match) across entire validation set
    #
    # Input
    #   val_dataloader : 
    #   net : BIML model
    #   langs : dict of dat.Lang classes
    #   max_length : maximum length of output sequences
    #   langs : dict of dat.Lang classes
    #   eval_type : 'max' for greedy decoding, 'sample' for sample from distribution
    #   out_mask_allow : default=[]; set of emission symbols we want to allow. Default of [] allows all output emissions
    net.eval()
    samples_pred = [] # list of all episodes with model predictions
    for batch_idx, val_batch in enumerate(val_dataloader): # each batch
        val_batch = dat.set_batch_to_device(val_batch)
        scores = batch_acc(val_batch, net, langs, max_length, eval_type=eval_type,
                            out_mask_allow=dat.get_batch_output_pool(val_batch))        
        samples_batch = val_batch['list_samples']
        for sidx in range(len(samples_batch)): # for each episode of the batch
            yq_sel = val_batch['q_idx'].cpu().numpy() == sidx # select for queries in this episode
            in_support = scores['in_support'][yq_sel] #numpy array
            is_novel = np.logical_not(in_support)
            v_acc = scores['v_acc'][yq_sel] #numpy array
            samples_batch[sidx]['yq_predict'] = extract(yq_sel, scores['yq_predict'])
            samples_batch[sidx]['v_acc'] = v_acc            
            samples_batch[sidx]['in_support'] = in_support #numpy array
            samples_batch[sidx]['acc_retrieve'] = np.mean(v_acc[in_support])*100.
            samples_batch[sidx]['acc_novel'] = np.mean(v_acc[is_novel])*100.
        samples_pred.extend(samples_batch)

    # Compute mean accuracy across all val episodes
    mean_acc_retrieve = np.mean([sample['acc_retrieve'] for sample in samples_pred])
    v_acc_novel = [sample['acc_novel'] for sample in samples_pred]
    mean_acc_novel = np.mean(v_acc_novel)

    if verbose:
        display_console_pred(samples_pred)
    return {'samples_pred':samples_pred, 'mean_acc_novel':mean_acc_novel, 'mean_acc_retrieve':mean_acc_retrieve, 'v_novel':v_acc_novel}

def batch_ll(batch, net, loss_fn, langs, p_lapse=0.0):
    # Evaluate log-likelihood (average over cells, and sum total) for a given batch
    #
    # Input
    #   batch : from dat.make_biml_batch
    #   loss_fn : loss function
    #   langs : dict of dat.Lang classes
    net.eval()
    m = len(batch['yq']) # b*nq
    target_batches = batch['yq_padded'] # b*nq x max_length
    target_lengths = batch['yq_lengths'] # list of size b*nq
    target_shift = batch['yq_sos_padded'] # b*nq x max_length
        # Shifted targets with padding (added SOS symbol at beginning and removed EOS symbol) 
    decoder_output = net(target_shift, batch)
        # b*nq x max_length x output_size    

    logits_flat = decoder_output.reshape(-1, decoder_output.shape[-1]) # (batch*max_len, output_size)
    if p_lapse > 0:
        logits_flat = smooth_decoder_outputs(logits_flat,p_lapse,langs['output'].symbols+[dat.EOS_token],langs)
    loss = loss_fn(logits_flat, target_batches.reshape(-1))
    loglike = -loss.cpu().item()
    dict_loss = {}
    dict_loss['ll_by_cell'] = loglike # average over cells
    dict_loss['N'] = float(sum(target_lengths)) # total number of valid cells
    dict_loss['ll'] = dict_loss['ll_by_cell'] * dict_loss['N'] # total LL
    return dict_loss

def smooth_decoder_outputs(logits_flat,p_lapse,lapse_symb_include,langs):
    # Mix decoder outputs (logits_flat) with uniform distribution over allowed emissions (in lapse_symb_include)
    #
    # Input
    #  logits_flat : (batch*max_len, output_size) # unnomralized log-probabilities
    #  p_lapse : probability of a uniform lapse
    #  lapse_symb_include : list of tokens (strings) that we want to include in the lapse model
    #  langs : dict of dat.Lang classes
    #
    # Output
    #  log_probs_flat : (batch*max_len, output_size) normalized log-probabilities
    lapse_idx_include = [langs['output'].symbol2index[s] for s in lapse_symb_include]
    assert dat.SOS_token not in lapse_symb_include # SOS should not be an allowed output through lapse model
    sz = logits_flat.size() # get size (batch*max_len, output_size)
    probs_flat = F.softmax(logits_flat,dim=1) # (batch*max_len, output_size)
    num_classes_lapse = len(lapse_idx_include)
    probs_lapse = torch.zeros(sz, dtype=torch.float)
    probs_lapse = probs_lapse.to(device=DEVICE)
    probs_lapse[:,lapse_idx_include] = 1./float(num_classes_lapse)
    log_probs_flat = torch.log((1-p_lapse)*probs_flat + p_lapse*probs_lapse) # (batch*max_len, output_size)
    return log_probs_flat

def batch_acc(batch, net, langs, max_length, eval_type='max', out_mask_allow=[]):
    # Evaluate exact match accuracy for a given batch
    #
    #  Input
    #   batch : from dat.make_biml_batch
    #   net : BIML model
    #   max_length : maximum length of output sequences
    #   langs : dict of dat.Lang classes
    #   eval_type : 'max' for greedy decoding, 'sample' for sample from distribution
    #   out_mask_allow : default=[]; list of emission symbols (strings) we want to allow. Default of [] allows all output emissions
    assert eval_type in ['max','sample']
    net.eval()
    emission_lang = langs['output']
    use_mask = len(out_mask_allow)>0
    memory, memory_padding_mask = net.encode(batch) 
        # memory : b*nq x maxlength_src x hidden_size
        # memory_padding_mask : b*nq x maxlength_src (False means leave alone)
    m = len(batch['yq']) # b*nq
    z_padded = torch.tensor([emission_lang.symbol2index[dat.SOS_token]]*m) # b*nq length tensor
    z_padded = z_padded.unsqueeze(1) # [b*nq x 1] tensor
    z_padded = z_padded.to(device=DEVICE)
    max_length_target = batch['yq_padded'].shape[1]-1 # length without EOS
    assert max_length >= max_length_target # make sure that the net can generate targets of the proper length

    # make the output mask if certain emissions are restricted
    if use_mask:
        assert dat.EOS_token in out_mask_allow # EOS must be included as an allowed symbol
        additive_out_mask = -torch.inf * torch.ones((m,net.output_size), dtype=torch.float)
        additive_out_mask = additive_out_mask.to(device=DEVICE)
        for s in out_mask_allow:
            sidx = langs['output'].symbol2index[s]
            additive_out_mask[:,sidx] = 0.

    # Run through decoder
    all_decoder_outputs = torch.zeros((m, max_length), dtype=torch.long)
    all_decoder_outputs = all_decoder_outputs.to(device=DEVICE)
    for t in range(max_length):
        decoder_output = net.decode(z_padded, memory, memory_padding_mask)
            # decoder_output is b*nq x (t+1) x output_size
        decoder_output = decoder_output[:,-1] # get the last step's output (batch_size x output_size)
        if use_mask: decoder_output += additive_out_mask

        # Choose the symbols at next timestep
        if eval_type == 'max': # pick the most likely
            topi = torch.argmax(decoder_output,dim=1)
            emissions = topi.view(-1)
        elif eval_type == 'sample':
            emissions = Categorical(logits=decoder_output).sample()
        all_decoder_outputs[:,t] = emissions
        z_padded = torch.cat([z_padded, emissions.unsqueeze(1)], dim=1)

    # Get predictions as strings and see if they are correct
    all_decoder_outputs = all_decoder_outputs.detach()
    yq_predict = [] # list of all predicted query outputs as strings
    v_acc = np.zeros(m)
    for q in range(m):
        myseq = emission_lang.tensor_to_symbols(all_decoder_outputs[q,:].view(-1))
        yq_predict.append(myseq)
        v_acc[q] = yq_predict[q] == batch['yq'][q] # for each query, did model get it right?
    in_support = np.array(batch['in_support']) # which queries are also support items
    out = {'yq_predict':yq_predict, 'v_acc':v_acc, 'in_support':in_support}
    return out

def viz_train_dashboard(train_tracker):
    # Show loss curves
    import matplotlib.pyplot as plt
    if not train_tracker:
        print('No training stats to plot')
        return
    fv = lambda x : [t[x] for t in train_tracker]
    plt.figure(1)
    plt.subplot(2, 2, 1)
    plt.plot(fv('step'),fv('avg_train_loss'),'b',label='train')
    if 'val_loss' in train_tracker[0] : plt.plot(fv('step'),fv('val_loss'),'r',label='val')
    plt.xlabel('step')
    plt.legend()
    plt.title('Loss')
    plt.subplot(2, 2, 2)
    plt.plot(fv('step'),fv('lr'),'b')
    plt.xlabel('step')
    plt.title('Learning rate')
    plt.show()

def display_console_pred(samples_pred):
    # Print model predictions
    #
    # Input
    #  samples_pred : list of dicts from evaluate_acc, which has predicted query outputs for each episode
    for idx,sample in enumerate(samples_pred):
        print('Evaluation episode ' + str(idx))
        in_support = sample['in_support']
        is_novel = np.logical_not(in_support)
        if 'grammar' in sample:
            print("")
            print(sample['grammar'])
        print('  support items;')
        display_input_output(sample['xs'],sample['ys'],sample['ys'])
        print('  retrieval items;',round(sample['acc_retrieve'],3),'% correct')
        display_input_output(extract(in_support,sample['xq']),extract(in_support,sample['yq_predict']),extract(in_support,sample['yq']))
        print('  generalization items;',round(sample['acc_novel'],3),'% correct')
        display_input_output(extract(is_novel,sample['xq']),extract(is_novel,sample['yq_predict']),extract(is_novel,sample['yq']))

def display_console_unmap(samples_pred):
    # Print model predictions after remapping
    #  There must also be a remapping from current tokens back to canonical tokens/
    #
    # Input
    #  samples_pred : list of dicts from evaluate_acc, which has predictions for each episode
    for idx,sample in enumerate(samples_pred):
        assert('unmap_input' in sample['aux']), "there must be mapping back to canonical text form"
        ui = lambda x : list(map(sample['aux']['unmap_input'], x))
        uo = lambda x : list(map(sample['aux']['unmap_output'], x))
        if 'filename' in sample['aux']:
            print('Evaluation episode ' + str(idx) + '; filename:', sample['aux']['filename'])
        else:
            print('Evaluation episode ' + str(idx))
        in_support = sample['in_support']
        is_novel = np.logical_not(in_support)
        if 'grammar' in sample:
            print("")
            print(sample['grammar'])
        print('  support items;')
        display_input_output(ui(sample['xs']),uo(sample['ys']),uo(sample['ys']))
        print('  retrieval items;',round(sample['acc_retrieve'],3),'% correct')
        display_input_output(extract(in_support,ui(sample['xq'])), extract(in_support,uo(sample['yq_predict'])), extract(in_support,uo(sample['yq'])))
        print('  generalization items;',round(sample['acc_novel'],3),'% correct')
        display_input_output(extract(is_novel,ui(sample['xq'])),extract(is_novel,uo(sample['yq_predict'])),extract(is_novel,uo(sample['yq'])))

def display_html_unmap(samples_pred, fid, freq='percent', include_support=False):
    #  Show model predictions when sampling. 
    #   Each episode consists of just one command, repeated multiple times. 
    #   There must also be a remapping back to canonical text format.
    #
    #   Input:
    #     samples_pred : list of dicts from evaluate_acc, which has predictions for each episode
    #     fid : handle for text file we are writing to
    #     freq : [percent OR count], format for reporting frequency
    #     include_support : show support set in HTML format? used for probe task
    fid.write('var all_data = [')
    for idx,sample in enumerate(samples_pred):
        assert freq in ['percent','count']
        assert('unmap_input' in sample['aux']), "there must be mapping back to canonical text form"
        ui = lambda x : list(map(sample['aux']['unmap_input'], x))
        uo = lambda x : list(map(sample['aux']['unmap_output'], x))
        in_support = sample['in_support']
        xq_novel = extract(np.logical_not(in_support), ui(sample['xq']))
        yq_predict_novel = extract(np.logical_not(in_support),uo(sample['yq_predict']))
        xq_novel = [' '.join(x) for x in xq_novel]
        assert all(xq==xq_novel[0] for xq in xq_novel), "each episode must be repeats of the same command"
        mycommand = xq_novel[0]
        myresponses = [' '.join(y) for y in yq_predict_novel]
        unique_responses = sorted(set(myresponses))
        count_unique = []
        for u in unique_responses:
            if freq=='percent':
                count_unique.append(100.*np.mean(np.array([u == rr for rr in myresponses],dtype=float)))
            else:
                count_unique.append(np.sum(np.array([u == rr for rr in myresponses],dtype=int)))
        unique_responses = [x for x in sorted(zip(unique_responses,count_unique), key=lambda pair: pair[1], reverse=True)]
        if include_support:
            fid.write('[ \n')
            fid.write("'support', \n")
            xs = ui(sample['xs'])
            ys = uo(sample['ys'])
            for j in range(len(xs)):
                fid.write("  ['%s', '%s'], \n" % (' '.join(xs[j]), ' '.join(ys[j]) ))
            if 'output_pool' in sample['aux']:
                mypool = sample['aux']['output_pool']
                fid.write("  ['%s', '%s'], \n" % ('Pool', ' '.join(uo([mypool])[0])) )
            fid.write('], \n')
        fid.write('[ \n')
        if include_support: fid.write("'query', \n")
        for u in unique_responses:
            if freq=='percent':
                fid.write("  ['%s', '%s', '%.1f%%'], \n" % (mycommand, u[0], u[1]) )            
            else:
                fid.write("  ['%s', '%s', %d], \n" % (mycommand, u[0], u[1]) )
        fid.write('], \n')
    fid.write(']; \n')

def evaluate_ll_multi_rep(nrep, val_dataloader, net, langs, p_lapse=0.0):
    # Across multiple random runs, evaluate the total (sum) log-likelihood across the validation set.
    #   Multiple runs may be needed when certain things are randomized, like the mapping between original data files
    #   and input/output tokens.
    #
    #  Input
    #   nrep : number of replications
    #    for rest of input arguments, see def evaluate_ll
    #   net : BIML model
    #   langs : dict of dat.Lang classes
    #
    # Output
    #   ave_ll_by_cell : return average log-likelihood for each cell
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=langs['output'].PAD_idx)
    list_ll = []
    list_N = []
    for i in range(nrep):
        if nrep >= 5 and (i % int(nrep/5)==0): print('  run',i)
        total_ll, total_N = evaluate_ll(val_dataloader, net, langs, loss_fn=loss_fn, p_lapse=p_lapse)        
        list_ll.append(total_ll)
        list_N.append(total_N)
    mean_ll = np.mean(list_ll)
    std_ll = np.std(list_ll)
    mean_N = np.mean(list_N)
    std_N = np.std(list_N)
    ave_ll_by_cell = np.sum(list_ll)/float(np.sum(list_N))
    print('  loglike: M =', round(mean_ll,4),'(SD=',round(std_ll,4),', Nrep=',nrep,') for', round(mean_N,4), '(SD=',round(std_N,4),') symbol predictions on average')
    print('    ave LL by cell: ', round(ave_ll_by_cell,4) )
    return {'ave_ll_by_cell': ave_ll_by_cell, 'ave_ll' : mean_ll, 'std_ll':std_ll, 'ave_N':mean_N}

def fit_p_lapse(nrep, val_dataloader, net, langs, greedy_stop=True):
    # Fit value for p_lapse. For each value, use evaluate_ll_multi_rep
    #  greedy_stop : stop when the objective no longer improves
    #
    # Output
    #  p_lapse : best fitting lapse value
    #  score_best : average log-like of best value
    #  mean_N : average number of evaluations 
    iter_p_lapse = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
    iter_p_lapse += [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    score_best = {'ave_ll_by_cell':float(-math.inf)}
    for p_lapse in iter_p_lapse:
        print(' p_lapse',p_lapse,':')
        seed_all()
        score_curr =  evaluate_ll_multi_rep(nrep, val_dataloader, net, langs, p_lapse=p_lapse)
        if score_curr['ave_ll_by_cell'] >= score_best['ave_ll_by_cell']:
            score_best = score_curr
            p_lapse_best = p_lapse
        elif greedy_stop:
            break # 
    return p_lapse_best, score_best

def evaluate_iterative(val_dataloader, net, langs, max_length, eval_type='max', out_mask_allow=[], verbose=False):
    # Sample from model iteratively: 
    #   1) Generate output for the first query
    #   2) Add this query as a study example, using self-generated output as the target
    #   3) If there is another query, go back to step 1.
    #  
    # Input
    #   val_dataloader : PyTorch dataloader
    #   net : BIML model
    #   langs : dict of dat.Lang classes
    #   max_length : maximum length of the output sequences
    #   eval_type : 'max' for greedy decoding, 'sample' for sample from distribution
    #   out_mask_allow : default=[]; set of emission symbols we want to allow. Default of [] allows all output emissions
    net.eval()
    samples_pred = [] # list of all episodes with model predictions
    for batch_idx, val_batch in enumerate(val_dataloader):
        val_batch = dat.set_batch_to_device(val_batch)
        scores, val_batch = batch_iterative(val_batch, net, langs, max_length, eval_type=eval_type, out_mask_allow=out_mask_allow, verbose=verbose)
        samples_batch = val_batch['list_samples']
        for sidx in range(len(samples_batch)): # for each episode of the batch
            yq_sel = val_batch['q_idx'].cpu().numpy() == sidx # select for queries in this episode
            m = np.sum(yq_sel)
            samples_batch[sidx]['yq_predict'] = extract(yq_sel, scores['yq_predict'])
            samples_batch[sidx]['in_support'] = np.ones(m,dtype=bool)
            samples_batch[sidx]['acc_retrieve'] = float('nan')
            samples_batch[sidx]['acc_novel'] = float('nan')
        samples_pred.extend(samples_batch)
    return {'samples_pred' : samples_pred}

def batch_iterative(batch, net, langs, max_length, eval_type='max', out_mask_allow=[], verbose=False):
    # Helper function for evaluate_iterative. Processes a whole batch in iterative manner
    batch_next = batch
    flag_batch_changed = True
    ii=0
    while flag_batch_changed:
        
        # Model predicts outputs for each query
        scores = batch_acc(batch_next, net, langs, max_length, eval_type=eval_type, out_mask_allow=out_mask_allow)
        samples_curr = batch_next['list_samples'] # list of samples
        samples_next = samples_curr # placeholder for modified samples
        bsize = len(samples_curr) # number of episodes
        for sidx in range(bsize): # divide predictions by episodes
            yq_sel = batch_next['q_idx'].cpu().numpy() == sidx # select for queries in this episode
            samples_curr[sidx]['yq_predict'] = extract(yq_sel, scores['yq_predict'])
            samples_curr[sidx]['in_support'] = scores['in_support'][yq_sel] # numpy array

        if verbose:
            print('\n** Iteration ', ii,'**')
            samples_pred = samples_curr
            for idx,sample in enumerate(samples_pred[:1]):
                print('Evaluation episode ' + str(idx))                
                in_support = sample['in_support']
                print('  support items: ')
                display_input_output(sample['xs'],sample['ys'],sample['ys'])
                print('  retrieval items; ')
                display_input_output(extract(in_support,sample['xq']),extract(in_support,sample['yq_predict']),extract(in_support,sample['yq']))
                print('  generalization items; ')
                display_input_output(extract(np.logical_not(in_support),sample['xq']),extract(np.logical_not(in_support),sample['yq_predict']),extract(np.logical_not(in_support),sample['yq']))        

        # Add command from first query to support set, using self-generated output as target
        flag_batch_changed = False # did any episode get modified this iteration?
        samples_new_list = []
        for sidx in range(bsize): # for each episode
            S_curr = deepcopy(samples_curr[sidx]) # current episode
            is_novel = np.logical_not(S_curr['in_support']) # queries that are not yet in the support set
            if np.any(is_novel): # if there is a genuine query
                flag_batch_changed = True
                myidx = np.nonzero(is_novel)[0][0] # pick the first query
                xq_add = S_curr['xq'][myidx]
                yq_add = S_curr['yq_predict'][myidx]
                
                # add query to support set
                samples_next[sidx]['xs'].append(deepcopy(xq_add))
                samples_next[sidx]['ys'].append(deepcopy(yq_add))

                # keep query in query set, but update the target
                del samples_next[sidx]['xq'][myidx]
                del samples_next[sidx]['yq'][myidx]
                samples_next[sidx]['xq'].append(deepcopy(xq_add))
                samples_next[sidx]['yq'].append(deepcopy(yq_add))
            samples_new_list += [dat.bundle_biml_episode(samples_next[sidx]['xs'],samples_next[sidx]['ys'],
                                                         samples_next[sidx]['xq'],samples_next[sidx]['yq'],'')]

        # create the next batch
        batch_next = dat.make_biml_batch(samples_new_list,langs)
        batch_next = dat.set_batch_to_device(batch_next)
        ii+=1
    return scores, batch_next

if __name__ == "__main__":

        # Adjustable parameters
        parser = argparse.ArgumentParser()
        parser.add_argument('--fn_out_model', type=str, default='', help='*REQUIRED*. Filename for loading the model')
        parser.add_argument('--dir_model', type=str, default='out_models', help='Directory for loading the model file')
        parser.add_argument('--max_length_eval', type=int, default=10, help='Maximum generated sequence length')
        parser.add_argument('--batch_size', type=int, default=-1, help='Number of episodes in batch')                                
        parser.add_argument('--episode_type', type=str, default='', help='What type of episodes do we want? See datasets.py for options')
        parser.add_argument('--dashboard', default=False, action='store_true', help='Showing loss curves during training.')
        parser.add_argument('--ll', default=False, action='store_true', help='Evaluate log-likelihood of validation (val) set')
        parser.add_argument('--max', default=False, action='store_true', help='Find best outputs for val commands (greedy decoding)')
        parser.add_argument('--sample', default=False, action='store_true', help='Sample outputs for val commands')
        parser.add_argument('--sample_html', default=False, action='store_true', help='Sample outputs for val commands in html format (using unmap to canonical text)')
        parser.add_argument('--sample_iterative', default=False, action='store_true', help='Sample outputs for val commands iteratively. Output in html format')
        parser.add_argument('--fit_lapse', default=False, action='store_true', help='Fit the best lapse rate according to log-likelihood on validation')
        parser.add_argument('--ll_nrep', type=int, default=1, help='Evaluate each episode this many times when computing log-likelihood (needed for stochastic remappings)')
        parser.add_argument('--ll_p_lapse', type=float, default=0., help='Lapse rate when evaluating log-likelihoods')
        parser.add_argument('--verbose', default=False, action='store_true', help='Inspect outputs in more detail')

        args = parser.parse_args()
        fn_out_model = args.fn_out_model
        dir_model = args.dir_model
        max_length_eval = args.max_length_eval
        episode_type = args.episode_type
        do_dashboard = args.dashboard
        batch_size = args.batch_size
        do_ll = args.ll
        do_max_acc = args.max
        do_sample_acc = args.sample
        do_sample_html = args.sample_html
        do_sample_iterative = args.sample_iterative
        do_fit_lapse = args.fit_lapse
        ll_nrep = args.ll_nrep
        ll_p_lapse = args.ll_p_lapse
        verbose = args.verbose

        model_tag = episode_type + '_' + fn_out_model.replace('.pt','')
        fn_out_model = os.path.join(dir_model, fn_out_model)
        if not os.path.isfile(fn_out_model):
             raise Exception('filename '+fn_out_model+' not found')

        seed_all()
        print('Loading model:',fn_out_model,'on',DEVICE)
        checkpoint = torch.load(fn_out_model, map_location=DEVICE)
        if not episode_type: episode_type = checkpoint['episode_type']
        if batch_size<=0: batch_size = checkpoint['batch_size']
        nets_state_dict = checkpoint['nets_state_dict']
        if list(nets_state_dict.keys())==['net']: nets_state_dict = nets_state_dict['net'] # for compatibility with legacy code
        input_size = checkpoint['langs']['input'].n_symbols
        output_size = checkpoint['langs']['output'].n_symbols
        emb_size = checkpoint['emb_size']
        dropout_p = checkpoint['dropout']
        ff_mult = checkpoint['ff_mult']
        myact = checkpoint['activation']
        nlayers_encoder = checkpoint['nlayers_encoder']
        nlayers_decoder = checkpoint['nlayers_decoder']
        train_tracker = checkpoint['train_tracker']
        best_val_loss = -float('inf')
        if 'best_val_loss' in checkpoint: best_val_loss = checkpoint['best_val_loss']
            
        print(' Loading model that has completed (or started) ' + str(checkpoint['epoch']) + ' of ' + str(checkpoint['nepochs']) + ' epochs')
        print('  test episode_type:',episode_type)
        print('  batch size:',checkpoint['batch_size'])
        print('  max eval length:', max_length_eval)
        print('  number of steps:', checkpoint['step'])
        print('  best val loss achieved: {:.4f}'.format(best_val_loss))

        # Load validation dataset
        D_train,D_val = dat.get_dataset(episode_type)
        langs = D_val.langs
        assert_consist_langs(langs,checkpoint['langs'])
        train_dataloader = DataLoader(D_train,batch_size=batch_size,
                                    collate_fn=lambda x:dat.make_biml_batch(x,langs),shuffle=False)
        val_dataloader = DataLoader(D_val,batch_size=batch_size,
                                    collate_fn=lambda x:dat.make_biml_batch(x,langs),shuffle=False)

        # For backward compatibility with legacy code that used same EOS and PAD tokens
        add_pad = dat.PAD_token not in checkpoint['langs']['input'].symbol2index
        
        # Load model parameters         
        net = BIML(emb_size, input_size, output_size,
            langs['input'].PAD_idx, langs['output'].PAD_idx,
            nlayers_encoder=nlayers_encoder, nlayers_decoder=nlayers_decoder, 
            dropout_p=dropout_p, activation=myact, ff_mult=ff_mult)        
        net.load_state_dict(nets_state_dict)
        net = net.to(device=DEVICE)
        describe_model(net)
    
        # Perform selected evaluations
        if do_dashboard:
            print('Showing loss curves during training <close plot to continue>')
            viz_train_dashboard(train_tracker)
        if do_ll and ll_nrep==1:
            seed_all()
            print('Evaluating log-likelihood of val episodes...')
            print('  with lapse rate',ll_p_lapse)
            total_ll,total_N = evaluate_ll(val_dataloader, net, langs, p_lapse=ll_p_lapse)
            print('evaluation on',episode_type,'loglike:',round(total_ll,4),'for',int(total_N),'symbol predictions')
            print('mean loglike is',round(total_ll/total_N,5),'per symbol')
        if do_ll and ll_nrep>1:
            seed_all()
            print('Evaluating log-likelihood of val episodes...')
            print('  with lapse rate',ll_p_lapse)
            print('  replicated across',ll_nrep,'random runs/permutations')
            evaluate_ll_multi_rep(ll_nrep, val_dataloader, net, langs, p_lapse=ll_p_lapse)
        if do_max_acc:
            seed_all()
            E = evaluate_acc(val_dataloader, net, langs, max_length_eval, eval_type='max', verbose=verbose)
            print('Evaluating set of validation episodes (via greedy decoding)...')
            print(' Acc Retrieve (val):',round(E['mean_acc_retrieve'],4))
            print(' Acc Novel (val):',round(np.mean(E['v_novel']),4),
                    'SD=',round(np.std(E['v_novel']),4),'N=',len(E['v_novel']))
        if do_sample_acc:
            seed_all()
            E = evaluate_acc(val_dataloader, net, langs, max_length_eval, eval_type='sample', verbose=verbose)
            print('Evaluating set of validation episodes (via sampling)...')
            print(' Acc Retrieve (val):',round(E['mean_acc_retrieve'],4))
            print(' Acc Novel (val):',round(np.mean(E['v_novel']),4),
                    'SD=',round(np.std(E['v_novel']),4),'N=',len(E['v_novel']))
        if do_sample_html:
            seed_all()
            print('Sampling from model to produce HTML file...')
            E = evaluate_acc(val_dataloader, net, langs, max_length_eval, eval_type='sample', verbose=False)            
            episode_type_tag = episode_type
            if 'probe_human' in episode_type: episode_type_tag = 'probe_human'
            with open('analysis/'+episode_type_tag+'/'+model_tag+'.txt','w') as fid_out:
                prev = sys.stdout
                sys.stdout = fid_out # re-rout outputs to file
                display_console_unmap(E['samples_pred'])
                sys.stdout = prev
            with open('analysis/'+episode_type_tag+'/template.html','r') as fid_in:
                mylines = fid_in.readlines()
            with open('analysis/'+episode_type_tag+'/'+model_tag+'.html','w') as fid_out:
                for l in mylines:
                    fid_out.write(l)
                    if l.strip() == '// PLACEHOLDER':
                        fid_out.write('var title="'+model_tag+'"; \n')
                        if 'probe' in episode_type_tag:
                            display_html_unmap(E['samples_pred'], fid_out, freq='percent', include_support=True)
                        else:    
                            display_html_unmap(E['samples_pred'], fid_out, freq='percent')
            print('Done writing to files: analysis/'+model_tag+'.html and .txt')                        
        if do_sample_iterative:
            seed_all()
            print('Iteratively evaluate queries by adding them, one-by-one, to the support set using self-generated targets.')
            E = evaluate_iterative(val_dataloader, net, langs, max_length_eval, eval_type='sample', verbose=verbose)
            with open('analysis/'+episode_type+'/'+model_tag+'.txt','w') as fid_out:
                prev = sys.stdout
                sys.stdout = fid_out # re-rout outputs to file
                display_console_pred(E['samples_pred'])
                sys.stdout = prev
        if do_fit_lapse:
            print('Fitting for the best value of p_lapse use log-like...')
            print('  Each value is replicated across',ll_nrep,'random runs/permutations')
            assert(ll_nrep>1), "should use more than one replication for parameter fitting"
            p_lapse_best, score_best = fit_p_lapse(ll_nrep, val_dataloader, net, langs)
            print('* BEST FIT * p_lapse=',p_lapse_best,'with mean loglike score of',round(score_best['ave_ll'],4),
                  '(or',round(score_best['ave_ll_by_cell'],4),'per cell)')