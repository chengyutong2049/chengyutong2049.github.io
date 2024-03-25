---
layout: default
title: hydra
nav_order: 3
has_children: false
parent: Coding Fundamental
permalink: /docs/coding-fundamental/hydra
math: mathjax
---

# logging
{: .fs-9 }

[Document](https://hydra.cc/).
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---
## Decorating the main function
```python
@hydra.main(config_path="config", config_name="config", version_base="1.2")
def run(config: DictConfig):
```
* This line is a decorator provided by Hydra, a framework for elegantly configuring complex applications. The decorator is applied to the run function (not shown in the snippet you provided, but it's the function immediately following this decorator in your code). Here's what each argument to the decorator specifies:
* `config_path="config"`: This tells Hydra where to look for configuration files. Hydra will search for configuration files in a directory named config relative to the location where the script is run.
* `config_name="config"`: This specifies the name of the default configuration file (without the file extension) that Hydra will use. Hydra expects this file to be in the config_path directory. So, in this case, Hydra will look for a file named config.yaml (or another supported format like .json or .yml) in the config directory.
* `version_base="1.2"`: This sets the version of the Hydra configuration. Hydra versions its configuration syntax and behavior, and specifying the version ensures that the correct parsing and validation logic is applied. Version "1.2" refers to a specific set of features and syntax supported by Hydra.
* When the script is executed, Hydra initializes the application's configuration according to the specified parameters. It loads the configuration from the specified file, potentially overriding it with command line arguments or environment variables, and then passes the resulting configuration object (DictConfig) as an argument to the decorated function (run in your case). This allows your application to be highly configurable, with the ability to easily switch between different settings or environments by simply changing the configuration file or command line arguments.
* `def run(config: DictConfig)`: This defines a function named run that takes a single parameter named config. The type of config is indicated as DictConfig, which is a special type provided by omegaconf. DictConfig is used to represent configuration data in a dictionary-like structure, allowing for easy access to configuration values.
Reply
* When the script is executed, the @hydra.main() decorator initializes Hydra with the specified configuration parameters. Hydra then loads the configuration from the specified file, potentially overriding it with command line arguments or environment variables. The resulting configuration object (DictConfig) is passed as an argument to the run function.
* This setup allows the application to be highly configurable, enabling easy switching between different settings or environments by simply changing the configuration file or command line arguments.



### Configuring the application
{: .fs-3 }
/home/yutong/GRACE/grace/config/config.yaml
{: .fs-3 .fw-100 }
```yaml
defaults:  
  - _self_
  - editor: null
  - experiment: null
  - model: null

tags: ~
batch_size: 1 # Batch size for computing TRR and ERR. Default is 1 but can be increased given larger GPUs
n_iter: 100 # Number of iterations to use per model during editing
max_n_edits: 1000 # Maximum number of edits during experiments
reinit: True # If True, download new model from huggingface always, if False huggingface tries to use an existing checkpoint
dropout: 0.0
device: cuda # Device to use. If 'cuda' but no GPU is available, all experiments default to CPU
wandb: False # Whether or not to use W&B at all
wandb_mode: online # Whether or not to push W&B results. Options: online, offline
wandb_project_name: my_project # W&B project name
wandb_run_name: null # Manual name for W&B
metric_period: 50 # How often to compute and record ERR and TRR (computing frequently is slow)
pretrain: False # (Hallucination Only) Whether or not to pre-train GPT2-XL
load_pretrained: True # Whether to try and load your own pre-trained GPT2-XL
re_init_model: False # Whether or not to initialize a new GPT2-XL model from scratch
ckpt: False # Whether or not to save your model after training
ckpt_dir: ./ckpts/
```

#### Configuring the application
```yaml
name: google/t5-small-ssm-nq
class_name: AutoModelForSeq2SeqLM
tokenizer_class: AutoTokenizer
tokenizer_name: google/t5-small-ssm-nq
inner_params:
- encoder.block[4].layer[1].DenseReluDense.wo.weight

pt: null # Path to a local model checkpoint, null means model is not pre-trained
```

```yaml
name: gpt2-xl
class_name: GPT2LMHeadModel
tokenizer_class: GPT2TokenizerFast
tokenizer_name: gpt2-xl
inner_params:
- transformer.h[35].mlp.c_fc.weight

# pt: null
pt: /data/healthy-ml/scratch/tomh/code/GRACE/ckpts/hallucination/ # set this to 'hallucination' inside your checkpoint directory
```


## hydra.utils.get_original_cwd
```python
hydra.utils.get_original_cwd()
```
* This line returns the original current working directory (cwd) before Hydra was initialized. This is useful if you need to access the original cwd in your script.

