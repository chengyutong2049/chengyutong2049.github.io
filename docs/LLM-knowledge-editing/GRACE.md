---
layout: default
title: GRACE
parent: LLM Knowledge Editing
permalink: /docs/LLM-knowledge-editing/GREACE
nav_order: 2
math: mathjax
---

# GRACE (General Retrieval Adaptors for Continual Editing)
{: .no_toc }

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
* If $$x_t$$ is passed into f again, $$h^{l-1}$$ would activate the codebook and value $$v$$ would be passed to layer $$l+1$$.

### $$\epsilon_{init}$$ parameter
* $$\epsilon_{init}$$ is the sole hyperparameter in GRACE.
* Intuitively, using a larger $$\epsilon_{init}$$ will create edits with more influence, making edits more general, but increasing the interference with unrelated iniputs. 

<div class="code-example" markdown="1">
[Link button](https://just-the-docs.com){: .btn }

[Link button](https://just-the-docs.com){: .btn .btn-purple }
[Link button](https://just-the-docs.com){: .btn .btn-blue }
[Link button](https://just-the-docs.com){: .btn .btn-green }

[Link button](https://just-the-docs.com){: .btn .btn-outline }
</div>
```markdown
[Link button](https://just-the-docs.com){: .btn }

[Link button](https://just-the-docs.com){: .btn .btn-purple }
[Link button](https://just-the-docs.com){: .btn .btn-blue }
[Link button](https://just-the-docs.com){: .btn .btn-green }

[Link button](https://just-the-docs.com){: .btn .btn-outline }
```

### Button element

GitHub Flavored Markdown does not support the `button` element, so you'll have to use inline HTML for this:

<div class="code-example">
<button type="button" name="button" class="btn">Button element</button>
</div>
```html
<button type="button" name="button" class="btn">Button element</button>
```

---

## Using utilities with buttons

### Button size

Wrap the button in a container that uses the [font-size utility classes]({% link docs/utilities/typography.md %}) to scale buttons:

<div class="code-example" markdown="1">
<span class="fs-6">
[Big ass button](https://just-the-docs.com){: .btn }
</span>

<span class="fs-3">
[Tiny ass button](https://just-the-docs.com){: .btn }
</span>
</div>
```markdown
<span class="fs-8">
[Link button](https://just-the-docs.com){: .btn }
</span>

<span class="fs-3">
[Tiny ass button](https://just-the-docs.com){: .btn }
</span>
```

### Spacing between buttons

Use the [margin utility classes]({% link docs/utilities/layout.md %}#spacing) to add spacing between two buttons in the same block.

<div class="code-example" markdown="1">
[Button with space](https://just-the-docs.com){: .btn .btn-purple .mr-2 }
[Button](https://just-the-docs.com){: .btn .btn-blue }

[Button with more space](https://just-the-docs.com){: .btn .btn-green .mr-4 }
[Button](https://just-the-docs.com){: .btn .btn-blue }
</div>
```markdown
[Button with space](https://just-the-docs.com){: .btn .btn-purple .mr-2 }
[Button](https://just-the-docs.com){: .btn .btn-blue }

[Button with more space](https://just-the-docs.com){: .btn .btn-green .mr-4 }
[Button](https://just-the-docs.com){: .btn .btn-blue }
```
