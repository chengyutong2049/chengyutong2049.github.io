---
layout: default
title: GRACE
parent: LLM Knowledge Editing
permalink: /docs/LLM-knowledge-editing/GREACE
nav_order: 3
has_children: true
math: mathjax
---

# GRACE (General Retrieval Adaptors for Continual Editing)
{: .no_toc }

[Github repo](https://github.com/chengyutong2049/GRACE)
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## GRACE overview
![](../../assets/images/LLM-knowledge-editing/GRACE(1).png){:width="1000"}
GRACE edits a model by adding an Adaptor to a chosen layer, while never changing its weights. This Adaptor then modifies  layer-to-layer transformations for select inputs. By caching embeddings for input errors and learning valurs that decode into desired model outputs, GRACE serves as a codebook in which edits are stored, enabling longer sequences of edits than prior works.


### Contributions

<div class="code-example" markdown="1">
1. Establish key metrics and comparisons for lifelong model editing, introduce 2 benchmark for lifelong model editing: [mitigating LLM hallucination](https://arxiv.org/pdf/2303.08896.pdf) and [addressing label shift](https://arxiv.org/pdf/2203.07228.pdf)
2. Develop GRACE
3. Experiments: GRACE outperforms seven alternatives when sequentially editing T5, BERT, and GPT models for QA, document classification, language generation.  
</div>

---
## Method: GRACE
<!-- ### Problem Formulation -->
GRACE works by wrapping a chosen layer of any pre-trained model architecture with an Adaptor. A GRACE Adaptor at model $$f$$'s layer $$l$$ contains two components: (1) a codebook $$C$$ and (2) a deferral mechanism to decide whether to use $$C$$ for a given input.

### GRACE codebook
* Keys ($$K$$): Set of keys, where each key is a cached activation $$h^{l-1}$$ predicted by layer $$l$$-1
* Values ($$V$$): Set of values that are initialized randomly and are updated using the model's finetuning loss for edits. Each key maps to a single, corresponding value.
* Deferral radii ($$\mathcal{E}$$): Each key has a deferral radius $$\epsilon$$, which serves as a threshold for similarity matching. New entries have a default value $$\epsilon_{init}$$, a hyperparameter.
  
![](../../assets/images/LLM-knowledge-editing/GRACE(1-1).png){:width="300"}

### Deferral mechanism
  
![](../../assets/images/LLM-knowledge-editing/GRACE(f1).png){:width="1000"}

### Codebook maintenance
![](../../assets/images/LLM-knowledge-editing/GRACE(a1).png){:width="400"}

To make an edit, a GRACE layer can perform one of two operations. 

* If the codebook is empty or the input embedding $$h^{l-1}$$ falls outside the deferral radius of any key in the codebook, the layer adds a new key-value pair to the codebook: {($$h^{l-1}$$, $$v$$, $$\epsilon_{init}$$, $$y$$)}. 
* If a query $$h^{l-1}$$ is close enough to an existing key that adding a new entry would cause their $$\epsilon$$-balls to overlap. To avoid this, compare the edit label y to the model's prediction of for the nearest key key and distinguish two cases:
    * If the overlapping's key's label is the same as y, **Expand** that key's $$\epsilon$$ to emcompass the query.
    * If the overlapping's key's label is different from y, **Split** these keys by first decreasing the influence radius of the overlapping key, then adding a new codebook entry where the new key is simply the query $$h^{l-1}$$.
      * Set both keys' $$\epsilon$$ to be half of their distant apart.
<!-- * If $$x_t$$ is passed into f again, $$h^{l-1}$$ would activate the codebook and value $$v$$ would be passed to layer $$l+1$$. -->

{: .note }
<!-- To use the theme, you do ***not*** need to clone or fork the [Just the Docs repo]! You should do that only if you intend to browse the theme docs locally, contribute to the development of the theme, or develop a new theme based on Just the Docs. -->
<!-- Please see different blog section in the left-side navigation bar. -->
$$\epsilon_{init}$$ is the sole hyperparameter in GRACE.
Intuitively, using a larger $$\epsilon_{init}$$ will create edits with more influence, making edits more general, but increasing the interference with unrelated inputs. 

### Training GRACE values
* When making a edit, either a new key-value pair is learned or an existing key-value pair is updated. 
* Train GRACE values using backpropagation through the finetuning loss on the model's prediction given the edit. The learned value then replace $$h^l$$ for the rest of the forward pass.
* In experiments, train values using 100 gradient descent steps.

### GRACE layers with sequential inputs
* For models with different representations per input token, like transformer, we must choose:
  1. which token should be GRACE's input query
  2. which token to replace with a retrieved value in the subsequent layers

{: .note }
Upon choosing these tokens, GRACE applies to all transformer models.

### Illustrative Example
![](../../assets/images/LLM-knowledge-editing/neural-net.png){:width="400"}
![](../../assets/images/LLM-knowledge-editing/image(2).png){:width="1100"}
* Sample 100 instances from two 2D distributions corresponding to classes. (a)
* Train a three-layer binary classifier with two 100-dimensional hidden layers and ReLU activations. (b)
* Introduce edits with flipped labels (synthetic data), stimulating local label shift at test time.
* The original model misclassifies these label flipped instances. (c)
* GRACE fixes these label without impacting other inputs using a single key in layer 2. (d)
* Finetuning on these errors alone break the model.

---
## Experiment
### Baselines
* Continual learning methods
  * Continually finetune ([**FT**](https://arxiv.org/pdf/2205.02014.pdf)) on streaming errors 
  * Elastic weight consolidation ([**FT+EWC**](https://arxiv.org/pdf/1612.00796.pdf))
  * Experience replay, periodicly retrain ([**FT+Retrain**](https://arxiv.org/pdf/1811.11682.pdf)) the model on previous edits.
* Model editing methods
  * [**MEND**](https://arxiv.org/pdf/2110.11309.pdf)
  * **Defer**, inspired by [SERAC]((https://arxiv.org/pdf/2206.06520.pdf))
  * [**ROME**](https://arxiv.org/pdf/2202.05262.pdf)
* Ablation study
  * Replace GRACE's discrete search with a **Memory** network containing memory module that is indexed by a soft attention mechanism.

### Datasets and pretrained models
![](../../assets/images/LLM-knowledge-editing/table(1).png){:width="1000"}

* **Test retention data:** testing set of each model's training data.
* **N:** the number of samples
* **Pre-edit:** unedited model's performance on each dataset


<div class="code-example" markdown="1">
- **Row 1**: Edit a 60M T5 model 
  - Model goal: QA (RE)
  * Training data: Random 1k samples from NQ
  * Edit data: [**zsRE**](https://arxiv.org/pdf/1706.04115.pdf)
- **Row 2**: Edit a 110M BERT classifier 
  - Model goal: Categorize US Supreme Court documents over multiple decades into 11 topics. 
    - Over time, categorization rules change, so label distributions shift.
  - Training data: 7.4k cases from [**SCOTUS**](https://arxiv.org/pdf/2203.07228.pdf) 1946-1982
  - Edit data: 931 cases from [**SCOTUS**](https://arxiv.org/pdf/2203.07228.pdf) 1991-2009
- **Row 3**: Edit a 1.5B GPT2-XL model 
  - Model goal: Generate wikipedia-style biographies.
  <!-- - Correct a GPT language model's **Hallucination** -->
  - Training data: First 1k sentences from OpenWebText
  - Edit data: SelfCheckGPT (**Hallucination**)
    - Replace inaccurate data with corresponding sentences in the true wikipedia entries.
      - 1392 sequential edits
      <!-- - 592 already-accurate outputs -->
</div>

### Metrics
![](../../assets/images/LLM-knowledge-editing/table(2).png){:width="1000"}
1.  Edit Success(**ES**): $$m(y,\hat{y})$$, $$m(\cdot)$$ is a task-specific measure of accuracy
    - Stanford F1 for QA
    - Accuracy for classification
    - Perplexity for generation
2. Test Retention Rate (**TRR**): How well an edited model retains its performance on its original testing data. 
  
    $$\frac{1}{N}\sum_{i=1}^{N}m(f(x_i),y_i)$$, $$(x_i, y_i)\in\mathcal{D}_{test}$$
3. Edit Retention Rate (**ERR**): How well an edited model retains previous edits

    $$\frac{1}{N}\sum_{i=1}^{N}m(f(x_i),y_i)$$, $$(x_i, y_i)\in\mathcal{D}_{edits}$$

4. Number of edits (**#E**)

# Aging with GRACE: Lifelong Model Editing with Discrete Key-Value Adaptors  - Github

Official implementation of **[Aging with GRACE: Lifelong Model Editing with Discrete Key-Value Adaptors](https://arxiv.org/abs/2211.11031)** (NeurIPS 2023).

<img width="1866" alt="image" src="https://github.com/Thartvigsen/GRACE/assets/26936677/8f28ab99-2411-4fd8-949b-8373ebfff3b5">

Please feel free to email [Tom](https://www.tomhartvigsen.com) or raise an issue with this repository and we'll get back to you as soon as possible.

## Setup
1. Create a virtual environment (we use conda)
    ```
    conda env create --name grace_env --file environment.yml
    ```
2. Activate the virtual environment
    ```
    conda activate grace_env
    ```
3. Install the repository
    ```
    pip install -e .
    ```

## Data
The QA experiments use data linked by the [MEND](https://github.com/eric-mitchell/mend) repository. Per their instructions, you can download the data for NQ and zsRE from [their Google Drive link](https://drive.google.com/drive/folders/1jAqBE45jEKR-5pMkwxlVQ0V8eKxqWbxA) and unzip each sub-directory into `grace/data`. SCOTUS and Hallucination data are handled through huggingface.

## Running experiments
Experiments are run using [main.py](./grace/main.py). Experiment settings and hyperparameters are chosen using [hydra](https://github.com/facebookresearch/hydra). While more examples are available in [./scripts/main.sh](./scripts/main.sh), three representative experiments can be run as follows:

### Editing GPT2-XL on Hallucination with GRACE
```
python grace/main.py experiment=hallucination model=gpt2xl editor=grace
```

### Editing BERT on SCOTUS with GRACE
```
python grace/main.py experiment=scotus model=bert editor=grace
```

### Editing T5 on zsRE with GRACE
```
python grace/main.py experiment=qa model=t5small editor=grace
```

## Repository Roadmap
* [./scripts/](./scripts/) contains handy shell scripts for starting and running experiments in slurm.
* [./notebooks/](./notebooks/) contains a simple example of editing a model with GRACE.
* [./ckpts/](./ckpts/) will contain checkpoints of your edited models if you choose to checkpoint models.
* [./data/](./data/) will contain downloaded datasets if you choose to cache data yourself instead of relying on HuggingFace.
* [./grace/](./grace/) contains the source code to GRACE
    * [./grace/main.py](./grace/main.py) is the main file to kick off experiments.
    * [./grace/config/](./grace/config/) contains the config files for datasets, editors, and pretrained models.
    * [./grace/editors/](./grace/editors/) contains source code for each compared editor.
    * [./grace/dataset.py](./grace/dataset.py) contains source code for each compared dataset.
    * [./grace/metrics.py](./grace/metrics.py) contains source code for each compared dataset.
    * [./grace/models.py](./grace/models.py) contains source code for loading pretrained models.

## Citation
Please use the following to cite this work:
```
@inproceedings{hartvigsen2023aging,
  title={Aging with GRACE: Lifelong Model Editing with Discrete Key-Value Adaptors},
  author={Hartvigsen, Thomas and Sankaranarayanan, Swami and Palangi, Hamid and Kim, Yoon and Ghassemi, Marzyeh},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```


