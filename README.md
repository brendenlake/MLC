# Behaviorally-Informed Meta-Learning (BIML)

BIML is a meta-learning approach for guiding neural networks to human-like systematic generalization and inductive biases, through high-level guidance or direct human examples. This code shows how to train and evaluate a sequence-to-sequence (seq2seq) transformer in PyTorch to implement BIML through memory-based meta-learning.

This code accompanies the following submitted paper:
- Lake, B. M. and Baroni, M. (submitted). Human-like systematic generalization through a meta-learning neural network. 

### Credits
This repo borrows from the excellent [PyTorch seq2seq tutorial](https://pytorch.org/tutorials/beginner/translation_transformer.html).

### Requirements
Python 3 with the following packages:
torch, sklearn, numpy, matplotlib


### Using the code

**Downloading the meta-training episodes**  
For training BIML on few-shot learning, you need to download the following [zip file](https://cims.nyu.edu/~brenden/supplemental/BIML-large-files/data_algebraic.zip) with the 100K meta-training episodes. Please extract `data_algebraic.zip` such that `data_algebraic`is a sub-directory of the main repository directory.

**Training a model**   
To train BIML on few-shot learning (BIML model in Fig. 2 and Table 4B), you can run the command:
```python
python train.py --episode_type algebraic+biases --fn_out_model net_algebraic+biases.tar
```
which will produce a file `out_models/net_algebraic+biases.tar`. 

You can also adjust the command line arguments.

Use the `-h` option in train.py to view all arguments:
```
optional arguments:
  -h, --help            show this help message and exit
  --fn_out_model FN_OUT_MODEL
                        *REQUIRED* Filename for saving model checkpoints.
                        Typically ends in .pt
  --dir_model DIR_MODEL
                        Directory for saving model files
  --episode_type EPISODE_TYPE
                        What type of episodes do we want? See datasets.py for
                        options
  --batch_size BATCH_SIZE
                        number of episodes per batch
  --nepochs NEPOCHS     number of training epochs
  --lr LR               learning rate
  --lr_end_factor LR_END_FACTOR
                        factor X for decrease learning rate linearly from
                        1.0*lr to X*lr across training
  --no_lr_warmup        Turn off learning rate warm up (by default, we use 1
                        epoch of warm up)
  --nlayers_encoder NLAYERS_ENCODER
                        number of layers for encoder
  --nlayers_decoder NLAYERS_DECODER
                        number of layers for decoder
  --emb_size EMB_SIZE   size of embedding
  --ff_mult FF_MULT     multiplier for size of the fully-connected layer in
                        transformer
  --dropout DROPOUT     dropout applied to embeddings and transformer
  --act ACT             activation function in the fully-connected layer of
                        the transformer (relu or gelu)
  --save_best           Save the "best model" according to validation loss.
  --save_best_skip SAVE_BEST_SKIP
                        Do not bother saving the "best model" for this
                        fraction of early training
  --resume              Resume training from a previous checkpoint
```                       

**Evaluating a model**  
To evaluate the accuracy of this model after training, you can use do the following:
```python
python eval.py --episode_type few_shot_gold --fn_out_model net_algebraic+biases.tar --max
```
You can also evaluate the log-likelihood (--ll) and draw samples from the distribution on outputs (--sample). The "few_shot_gold" task is the same task provided to human participants.

Use the `-h` option to view all arguments:
```
optional arguments:
  -h, --help            show this help message and exit
  --fn_out_model FN_OUT_MODEL
                        *REQUIRED*. Filename for loading the model
  --dir_model DIR_MODEL
                        Directory for loading the model file
  --max_length_eval MAX_LENGTH_EVAL
                        Maximum generated sequence length
  --batch_size BATCH_SIZE
                        Number of episodes in batch
  --episode_type EPISODE_TYPE
                        What type of episodes do we want? See datasets.py for
                        options
  --dashboard           Showing loss curves during training.
  --ll                  Evaluate log-likelihood of validation (val) set
  --max                 Find best outputs for val commands (greedy decoding)
  --sample              Sample outputs for val commands
  --sample_html         Sample outputs for val commands in html format (using
                        unmap to canonical text)
  --sample_iterative    Sample outputs for val commands iteratively
  --fit_lapse           Fit the lapse rate
  --ll_nrep LL_NREP     Evaluate each episode this many times when computing
                        log-likelihood (needed for stochastic remappings)
  --ll_p_lapse LL_P_LAPSE
                        Lapse rate when evaluating log-likelihoods
  --verbose             Inspect outputs in more detail
```

**Episode types**

See datasets.py for the full set of options. Here are a few key episode types:
- "algebraic+biases" : Corresponds to "BIML" in Table 4B and main results
- "algebraic_noise" : Corresponds to "BIML (algebraic only)" in Table 4B and main results
- "retrieve" : Correspond to "BIML (copy only)" in Table 4B and main results
- "few_shot_gold" : For evaluating BIML and people on the same few-shot learning task. This episode type provides the test set only.