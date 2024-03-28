---
layout: default
title: python
nav_order: 3
has_children: false
parent: Coding Fundamental
permalink: /docs/coding-fundamental/python
math: mathjax
---

# Python
{: .fs-9 }

<!-- [Document](https://docs.python.org/3/library/logging.html). -->
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## zip()
```python
for x, y in zip(questions[:1000], answers[:1000]):
            self.data.append({
                "text": x,
                "labels": y
            })
```
* The zip function is used to combine two lists into a single list of tuples.
* In this example, the `questions` and `answers` lists are combined into a single **list of tuples**, which is then used to create a new list of dictionaries. Each dictionary contains a question and its corresponding answer.

## jsonlines
```python
with jsonlines.open(data_path) as f:
            for d in f:
                ex = {k: d[k] for k in ["input", "prediction", "alternatives", "filtered_rephrases", "output"]}
                questions.append(ex["input"])
                answers.append(ex["output"][0]["answer"])
                if len(ex["filtered_rephrases"]) >= 10: # Only use samples for which there are 10 rephrasings
                    for rephrase in ex["filtered_rephrases"][:10]: # Only use the first 10 rephrasings
                        questions.append(rephrase)
                        answers.append(ex["output"][0]["answer"])
```
* The `jsonlines` library is used to read data from a JSON file line by line.
* In this example, the `jsonlines.open()` function is used to open a JSON file and read its contents line by line.
* f is an iterator that yields dictionaries, one for each line in the file.

## reference 
```python
from grace.metrics import F1, PPL, Accuracy, is_qa_error, is_acc_error
metric = F1 # Measure QA F1
```
```python
def F1(model, batch):
    try:
        preds = model.generate(batch["input_ids"], max_length=20).squeeze()
        if len(preds) > 1:
            preds = preds[preds != model.tokenizer.pad_token_id]
        gold_toks = batch["labels"][batch["labels"] != -100].cpu().squeeze() # -100 might be nonsense
        num_same = len(np.intersect1d(preds.cpu().squeeze(), gold_toks))
        if (num_same == 0) or (len(preds.squeeze()) == 0):
            return 0
        precision = num_same / len(preds.squeeze())
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    except:
        # Every once in a while, the model just returns the stop token
        return 0
```
* Assigning metric = F1 means that metric now holds a reference to the F1 function. You can call metric just like you would call F1, passing any required arguments to compute the F1 score (e.g., score = metric(y_true, y_pred)).

## [\:]
* This syntax is used for slicing in Python. It can take up to three parameters: start, stop, and step. When used as [:-1], it specifies a slice of components from the beginning (start is omitted and thus defaults to 0) up to, but not including, the last element (stop is -1, which represents the last element in Python indexing).

## getattr(...)
```python
for component in components[:-1]:#encoder.block.4.layer.1 -> encoder.block.4.layer
        if hasattr(parent, component):
            parent = getattr(parent, component)
#parent = model
```
* The `getattr()` function in Python is used to get the value of a named attribute of an object. 
* It takes two arguments: the object and the attribute name.

## isdigit()
* This is a method available on string objects in Python. 
* It returns True if all characters in the string are digits, and there is at least one character, otherwise, it returns False. 
* Digits include the characters 0 through 9.

## int()
* This is a built-in Python function that converts a string or number to an integer.

## setattr(edit_module, layer_name, ...)

```python
setattr(edit_module, layer_name, GRACEAdapter(config, original_layer, transpose=transpose).to(self.device))
```
* This function dynamically sets an attribute on edit_module. It replaces the attribute named layer_name with the new GRACEAdapter instance. This is a critical step where the actual modification of the model takes place, inserting the GRACEAdapter in place of the original layer or parameter.