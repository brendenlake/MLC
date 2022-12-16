import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import argparse
import math
import time
from model import BIML, describe_model
from eval import evaluate_ll
import datasets as dat
from train_lib import seed_all, timeSince
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(batch, net, loss_fn, optimizer, langs):
    # Update the model for one batch (which is a set of episodes)
    #
    # Input
    #   batch : dict output from dat.make_biml_batch
    #   net : BIML model
    #   loss_fn : loss function
    #   optimizer : torch optimizer (AdamW)
    #   langs : input and output language class
    optimizer.zero_grad()
    net.train()
    m = len(batch['yq']) # effective batch size b*nq (num_episodes*num_queries)
    target_batches = batch['yq_padded'] # b*nq x max_length
    target_lengths = batch['yq_lengths'] # list of size b*nq
    target_shift = batch['yq_sos_padded'] # b*nq x max_length
        # shifted targets with padding (added SOS symbol at beginning and removed EOS symbol) 
    decoder_output = net(target_shift, batch) # b*nq x max_length x output_size
    logits_flat = decoder_output.reshape(-1, decoder_output.shape[-1]) # (b*nq*max_length, output_size)
    loss = loss_fn(logits_flat, target_batches.reshape(-1))
    assert(not torch.isinf(loss))
    assert(not torch.isnan(loss))
    loss.backward()
    optimizer.step()
    dict_loss = {}
    dict_loss['total'] = loss.cpu().item()
    return dict_loss

def save_checkpoint(fn_out_model, step, epoch, net, optimizer, scheduler_epoch, train_tracker, best_val_loss, params, is_best=False):
    # Input
    #  fn_out_model : filename for saving the model
    #  step : number of gradient steps
    # ..
    #  train_tracker : array that stores losses over training
    #  best_val_loss : best validation loss so far (if using --save_best)
    #  params : list of hyperpameters
    #  is_best : special filename if best file so far  ... 'filename_best.pt'
    if is_best:
        s = fn_out_model.rsplit('.',1) # split off extension 
        fn_out_model = s[0] + '_best.' + s[1]
        print('> Saving new *best* model as',fn_out_model, end='')
    else:
        print('> Saving model as',fn_out_model, end='')
    state = {'step' : step,
             'epoch' : epoch,
             'nets_state_dict' : net.state_dict(),
             'optimizer_state_dict' : optimizer.state_dict(),
             'scheduler_epoch_state_dict' : scheduler_epoch.state_dict(),
             'train_tracker' : train_tracker,
             'best_val_loss' : best_val_loss}
    state.update(params)
    torch.save(state, fn_out_model)
    print(' < Done. >')

def load_checkpoint(fn_out_model, net, optimizer, scheduler_epoch, params):
    # Note that the command line args must be the same now as when model was saved.
    #  (Except for 'resume' parameter)
    #
    # Input
    #  fn_out_model : filename for model to resume
    # ...
    #  params : list of hyperpameters
    # 
    # Output
    #  step : number of gradient steps at which we resume
    #  epoch_resume_start : number of completed epochs + 1
    #  train_tracker : array that stores losses over training
    #  best_val_loss : best validation loss so far (if using --save_best)
    checkpoint = torch.load(fn_out_model, map_location=DEVICE)
    curr_args = vars(args)
    prev_args = vars(checkpoint['args'])
    for k in prev_args.keys():
        if k!='resume': assert(prev_args[k]==curr_args[k]) # check that command line args match the checkpoint's
    for k in params.keys():
        if k not in {'langs','args'}: assert(params[k]==checkpoint[k]) # check that hyperparams match the checkpoint's    
    net.load_state_dict(checkpoint['nets_state_dict'])
    net = net.to(device=DEVICE)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler_epoch.load_state_dict(checkpoint['scheduler_epoch_state_dict'])
    step = checkpoint['step']
    epoch = checkpoint['epoch']
    epoch_resume_start = epoch+1
    train_tracker = checkpoint['train_tracker']
    best_val_loss = checkpoint['best_val_loss']
    return step, epoch_resume_start, train_tracker, best_val_loss

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--fn_out_model', type=str, default='', help='*REQUIRED* Filename for saving model checkpoints. Typically ends in .pt')
    parser.add_argument('--dir_model', type=str, default='out_models', help='Directory for saving model files')
    parser.add_argument('--episode_type', type=str, default='retrieve', help='What type of episodes do we want? See datasets.py for options')
    parser.add_argument('--batch_size', type=int, default=25, help='number of episodes per batch')
    parser.add_argument('--nepochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_end_factor', type=int, default=0.05, help='factor X for decrease learning rate linearly from 1.0*lr to X*lr across training')
    parser.add_argument('--no_lr_warmup', default=False, action='store_true', help='Turn off learning rate warm up (by default, we use 1 epoch of warm up)')
    parser.add_argument('--nlayers_encoder', type=int, default=3, help='number of layers for encoder')
    parser.add_argument('--nlayers_decoder', type=int, default=3, help='number of layers for decoder')
    parser.add_argument('--emb_size', type=int, default=128, help='size of embedding')
    parser.add_argument('--ff_mult', type=int, default=4, help='multiplier for size of the fully-connected layer in transformer')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout applied to embeddings and transformer')        
    parser.add_argument('--act', type=str, default='gelu', help='activation function in the fully-connected layer of the transformer (relu or gelu)')
    parser.add_argument('--save_best', default=False, action='store_true', help='Save the "best model" according to validation loss.')
    parser.add_argument('--save_best_skip', type=float, default=0.2, help='Do not bother saving the "best model" for this fraction of early training')
    parser.add_argument('--resume', default=False, action='store_true', help='Resume training from a previous checkpoint')

    args = parser.parse_args()
    fn_out_model = args.fn_out_model
    dir_model = args.dir_model
    episode_type = args.episode_type
    batch_size = args.batch_size
    nepochs = args.nepochs
    adamW_learning_rate = args.lr
    lr_end_factor = args.lr_end_factor
    lr_warmup = not args.no_lr_warmup
    nlayers_encoder = args.nlayers_encoder
    nlayers_decoder = args.nlayers_decoder
    emb_size = args.emb_size
    ff_mult = args.ff_mult
    dropout_p = args.dropout        
    myact = args.act
    bool_save_best = args.save_best
    save_best_skip = args.save_best_skip
    bool_resume = args.resume
    assert(myact in ['relu','gelu'])
    assert(len(fn_out_model)>0) # must have filename for saving the model checkpoints
    if not os.path.exists(dir_model):
        os.makedirs(dir_model)
    fn_out_model = os.path.join(dir_model, fn_out_model)
    if not bool_resume and os.path.isfile(fn_out_model):
        print('Filename '+fn_out_model+' already exists.')
        response = input('Do you want to OVERWRITE it? ("y" if yes): ')
        if response.strip()!='y': sys.exit()
        print("Training a new network...")
    if bool_resume:
        assert(os.path.isfile(fn_out_model)), "Filename to resume does not exist: "+fn_out_model
        print('Filename '+fn_out_model+' already exists.')
        print("Resuming a previous network run...")

    print('  File output: '+fn_out_model)
    print("  Episode type:",episode_type)
    D_train, D_val = dat.get_dataset(episode_type)        
    train_dataloader = DataLoader(D_train,batch_size=batch_size,collate_fn=lambda x:dat.make_biml_batch(x,D_train.langs),
                                    shuffle=True)
    val_dataloader = DataLoader(D_val,batch_size=batch_size,collate_fn=lambda x:dat.make_biml_batch(x,D_val.langs),
                                    shuffle=False)


    nsteps_estimate = math.ceil(nepochs*len(D_train)/batch_size)
    print('  Training on', DEVICE, end='')
    print(' for', nepochs, 'epochs with batch size', batch_size)    
    print('    for a total of',nepochs*len(D_train),'episode presentations')
    if bool_save_best: print('    with "save best" early stopping')
    langs = D_train.langs
    input_size = langs['input'].n_symbols
    output_size = langs['output'].n_symbols
    params_state = {'langs': langs, 'episode_type': episode_type, 'emb_size':emb_size, 'input_size':input_size, 'output_size':output_size,
                    'dropout':dropout_p, 'nlayers_encoder':nlayers_encoder, 'nlayers_decoder':nlayers_decoder,
                    'nepochs':nepochs, 'batch_size':batch_size, 'activation':myact, 'ff_mult':ff_mult, 'args':args}

    # setup model
    net = BIML(emb_size, input_size, output_size,
        langs['input'].PAD_idx, langs['output'].PAD_idx,
        nlayers_encoder=nlayers_encoder, nlayers_decoder=nlayers_decoder, 
        dropout_p=dropout_p, activation=myact, ff_mult=ff_mult)
    net = net.to(device=DEVICE)

    # setup loss and scheduled
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=langs['output'].PAD_idx)
    optimizer = optim.AdamW(net.parameters(),lr=adamW_learning_rate, betas=(0.9,0.95), weight_decay=0.01)
    if lr_warmup:
        print('    with LR warmup ON (1st epoch)')
        scheduler_epoch = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=lr_end_factor, total_iters=nepochs-2, verbose=False)
        nstep_epoch_estimate = math.floor(len(D_train)/batch_size)
        scheduler_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-4, end_factor=1.0, total_iters=nstep_epoch_estimate, verbose=False)
    else:            
        print('    with LR warmup OFF')
        scheduler_epoch = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=lr_end_factor, total_iters=nepochs-1, verbose=False)

    describe_model(net)        
    avg_train_loss = 0.
    best_val_loss = float('inf')
    counter = 0 # num updates since the loss was last reported
    step = 0
    train_tracker = []
    epoch_start = 1
    start = time.time()

    if bool_resume:
        print("Loading checkpoint states for net, optimizer, and scheduler.")
        step, epoch_start, train_tracker, best_val_loss = load_checkpoint(fn_out_model, net, optimizer, scheduler_epoch, params_state)

    print('Setting LR={:.7f}'.format(optimizer.param_groups[0]['lr']))
    for epoch in range(epoch_start,nepochs+1):
        print("Epoch",epoch,"\n-------------------------------")

        for batch_idx, train_batch in enumerate(train_dataloader):
            train_batch = dat.set_batch_to_device(train_batch)
            dict_loss = train(train_batch, net, loss_fn, optimizer, langs)
            avg_train_loss += dict_loss['total']
            counter += 1
            step += 1                
            if step in [1,25] or step % 100 == 0:
                mylr = optimizer.param_groups[0]['lr']
                mytracker = {'epoch':epoch, 'step':step, 'lr':mylr, 'avg_train_loss':avg_train_loss/counter}
                print('{:s} ({:d} {:.0f}% finished) LR: {:.7f}, TrainLoss: {:.4f}, '.format(timeSince(start, float(step) / float(nsteps_estimate)),
                                         step, float(step) / float(nsteps_estimate) * 100., mylr, avg_train_loss/counter), end='')
                
                # compute val loss
                total_ll, total_N = evaluate_ll(val_dataloader, net, langs, loss_fn=loss_fn)
                val_loss = -total_ll/total_N
                print('ValLoss: {:.4f}'.format(val_loss))
                mytracker['val_loss'] = val_loss
                avg_train_loss = 0.
                counter = 0
                train_tracker.append(mytracker)

                if bool_save_best and val_loss < best_val_loss and (epoch > nepochs*save_best_skip): # don't bother saving best model in early epochs
                    best_val_loss = val_loss
                    save_checkpoint(fn_out_model,step,epoch,net,optimizer,scheduler_epoch,train_tracker,best_val_loss,params_state,is_best=True)

            # if warm-up, adjust learning rate for each step of the first epoch
            if lr_warmup and epoch==1: scheduler_warmup.step()
        
        # after each epoch, adjust the general learning rate
        if epoch>1 or not lr_warmup: scheduler_epoch.step()
        save_checkpoint(fn_out_model,step,epoch,net,optimizer,scheduler_epoch,train_tracker,best_val_loss,params_state)

    print('Training complete.')