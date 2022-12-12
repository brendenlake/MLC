# Behaviorally-Informed Meta-Learning (BIML)

BIML is a meta-learning approach for guiding neural networks to human-like inductive biases, through high-level guidance or direct human examples. This code shows how to train and evaluate a sequence-to-sequence (seq2seq) transformer, which implements BIML using a form of memory-based meta-learning.

### Credits
This repo borrows from the excellent [PyTorch seq2seq tutorial](https://pytorch.org/tutorials/beginner/translation_transformer.html).

### Using the code

**Training a model**   
This demos a simple retrieval task. To train a model that just retrieves a query output from the support set (which contains the query command exactly), you can type:
```python
python train.py --episode_type retrieve --nepochs 10 --fn_out_model net_retrieve.tar
```
which will produce a file `out_models/net_retrieve.tar`. 

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
To evaluate the accuracy of this model after training, you can type:
```python
python eval.py --episode_type retrieve --fn_out_model net_retrieve.tar --max
```
You can also evaluate the log-likelihood (--ll) and draw samples from the distribution on outputs (--sample).

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
  --episode_type EPISODE_TYPE
                        What type of episodes do we want? See datasets.py for
                        options
  --dashboard           Showing loss curves during training.
  --ll                  Evaluate log-likelihood of validation (val) set
  --max                 Find best outputs for val commands (greedy decoding)
  --sample              Sample outputs for val commands
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
- "algebraic+biases" : For training the full BIML model on few-shot grammar induction. Also has a validation set.
- "algebraic_noise" : For training BIML (algebraic only). Also has a validation set.
- "few_shot_gold" : For evaluating BIML and people on the same few-shot learning task. Validation set only.

