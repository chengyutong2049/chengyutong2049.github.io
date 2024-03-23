---
layout: default
title: GREACE
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
![](../../assets/images/LLM-knowledge-editing/greace(1).png){:width="1000"}
GRACE edits a model by adding an Adaptor to a chosen layer, while never changing its weights. This Adaptor then modifies  layer-to-layer transformations for select inputs. By caching embeddings for input errors and learning valurs that decode into desired model outputs, GRACE serves as a codebook in which edits are stored, enabling longer sequences of edits than prior works.


### Contributions

<div class="code-example" markdown="1">
1. Establish key metrics and comparisons for lifelong model editing, introduce 2 benchmark for lifelong model editing: [mitigating LLM hallucination](https://arxiv.org/pdf/2303.08896.pdf) and [addressing label shift](https://arxiv.org/pdf/2203.07228.pdf)
2. Develop GRACE
3. Experiments: GRACE outperforms seven alternatives when sequentially editing T5, BERT, and GPT models for QA, document classification, language generation.  
</div>

## Method: GRACE
<!-- ### Problem Formulation -->
GRACE works by wrapping a chosen layer of any pre-trained model architecture with an Adaptor. A GRACE Adaptor at model $$f$$'s layer $$l$$ contains two components: (1) a codebook $$C$$ and (2) a deferral mechanism to decide whether to use $$C$$ for a given input.

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
