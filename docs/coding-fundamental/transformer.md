---
layout: default
title: transformers
nav_order: 3
has_children: false
parent: Coding Fundamental
permalink: /docs/coding-fundamental/transformers
math: mathjax
---

# transformers
{: .fs-9 }

[Document](https://huggingface.co/docs/transformers/index).
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Introduction
Transformers provides APIs and tools to easily download and train state-of-the-art pretrained models. Using pretrained models can reduce your compute costs, carbon footprint, and save you the time and resources required to train a model from scratch. 

## transformers.AutoModelForSeq2SeqLM
<!-- [Problem of wandb in train.py](https://github.com/ultralytics/yolov5/issues/5772)   -->
```python
ModelClass = getattr(transformers, config.model.class_name)
```
* [transformers.AutoModelForSeq2SeqLM](https://huggingface.co/docs/transformers/v4.39.1/en/model_doc/auto#transformers.AutoModelForSeq2SeqLM)
* This is a generic model class that will be instantiated as one of the model classes of the library (with a sequence-to-sequence language modeling head) when created with the `from_pretrained()` class method or the `from_config()` class method.

### from_pretrained
```python
model = ModelClass.from_pretrained(config.model.name, cache_dir=ckpt_dir())
```
* This line instantiates a model from the `ModelClass` class using the pretrained weights specified by `config.model.name`.
* `cache_dir` (str or os.PathLike, optional) — Path to a directory in which a downloaded pretrained model configuration should be cached if the standard cache should not be used.
* `pretrained_model_name_or_path` (str or os.PathLike) — Can be either:
  * A string, the model id of a pretrained model hosted inside a model repo on huggingface.co.
  * A path to a directory containing model weights saved using save_pretrained(), e.g., ./my_model_directory/.
  * A path or url to a tensorflow index checkpoint file (e.g, ./tf_model/model.ckpt.index). In this case, from_tf should be set to True and a configuration object should be provided as config argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

* The model class to instantiate is selected based on the model_type property of the config object (either passed as an argument or loaded from pretrained_model_name_or_path if possible), or when it’s missing, by falling back to using pattern matching on pretrained_model_name_or_path:
  * bart — BartForConditionalGeneration (BART model)
  * bigbird_pegasus — BigBirdPegasusForConditionalGeneration (BigBird-Pegasus model)
  * blenderbot — BlenderbotForConditionalGeneration (Blenderbot model)
  * blenderbot-small — BlenderbotSmallForConditionalGeneration (BlenderbotSmall model)
  * encoder-decoder — EncoderDecoderModel (Encoder decoder model)
  * fsmt — FSMTForConditionalGeneration (FairSeq Machine-Translation model)
  * gptsan-japanese — GPTSanJapaneseForConditionalGeneration (GPTSAN-japanese model)
  * led — LEDForConditionalGeneration (LED model)
  * longt5 — LongT5ForConditionalGeneration (LongT5 model)
  * m2m_100 — M2M100ForConditionalGeneration (M2M100 model) 
  * marian — MarianMTModel (Marian model)
  * mbart — MBartForConditionalGeneration (mBART model)
  * mt5 — MT5ForConditionalGeneration (MT5 model)
  * mvp — MvpForConditionalGeneration (MVP model)
  * nllb-moe — NllbMoeForConditionalGeneration (NLLB-MOE model)
  * pegasus — PegasusForConditionalGeneration (Pegasus model)
  * pegasus_x — PegasusXForConditionalGeneration (PEGASUS-X model)
  * plbart — PLBartForConditionalGeneration (PLBart model)\
  * prophetnet — ProphetNetForConditionalGeneration (ProphetNet model)
  * seamless_m4t — SeamlessM4TForTextToText (SeamlessM4T model)
  * seamless_m4t_v2 — SeamlessM4Tv2ForTextToText (SeamlessM4Tv2 model)
  * switch_transformers — SwitchTransformersForConditionalGeneration (SwitchTransformers model)
  * t5 — T5ForConditionalGeneration (T5 model)
  * umt5 — UMT5ForConditionalGeneration (UMT5 model)
  * xlm-prophetnet — XLMProphetNetForConditionalGeneration (XLM-ProphetNet model)


### t5
[Document](https://huggingface.co/docs/transformers/v4.39.1/en/model_doc/t5#transformers.T5ForConditionalGeneration)

```python
from transformers import AutoTokenizer, T5ForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")#This model is loaded from huggingface
```
* [Model Repo](https://huggingface.co/google/t5-small-ssm-nq)


### training
```python
input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
outputs = model(input_ids=input_ids, labels=labels)
loss = outputs.loss
logits = outputs.logits
```
### inference
```python
input_ids = tokenizer(
    "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
).input_ids  # Batch size 1
outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# studies have shown that owning a dog is good for you.
```

