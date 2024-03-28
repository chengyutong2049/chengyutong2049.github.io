---
layout: default
title: zsRE
parent: LLM Knowledge Editing
permalink: /docs/LLM-knowledge-editing/zsRE
nav_order: 2
math: mathjax
---

# zsRE
{: .fs-9 }
<!-- Here we will introduce the paper: [Fast Model Editing at Scale](https://openreview.net/pdf?id=0DcZxeWfOPt). -->
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---


```json
{
    "id": "e635bbe7-c8fb-4014-99ba-a14c01093630", 
    "input": "What university did Watts Humphrey attend?", 
    "output": 
    [
        {
            "answer": "Illinois Institute of Technology", 
            "provenance": [
                {
                    "wikipedia_id": "547618", 
                    "title": "Watts Humphrey", 
                    "start_paragraph_id": 6, 
                    "start_character": 18, 
                    "end_paragraph_id": 6, 
                    "end_character": 282, 
                    "bleu_score": 0.9505786228129219, 
                    "meta": {}, 
                    "section": "Section::::Biography.\n"
                }
            ]
        }
    ], 
    "meta": {
        "template_questions": ["What university did Watts Humphrey attend?"]
    }, 
    "rephrases": ["Which university did Watts Humphrey attend?", "Which university has Watts Humphrey attended?", "Which university did Watts Humphrey go to?", "Which university has Watts Humphrey visited?", "Which university attended Watts Humphrey?", "Which university did Watts go to Humphrey?", "What university did Watts attend Humphrey at?", "Which university did Watts attend Humphrey?", "What university did Watts attend Humphrey?", "What university did Watts go to Humphrey?", "Which university did Watts Humphrey take part in?", "What university did Watts Humphrey take part in?", "Which university did Watts Humphrey participate in?", "Which university did Watts Humphrey study at?", "What university did Watts Humphrey study at?", "What university did Watts Humphrey go to?", "What university did Watts Humphrey attend?"], 
    "prediction": "Trinity College", 
    "alternatives": ["Yale University", "University of Chicago", "King's College London", "University of Michigan"], 
    "filtered_rephrases": ["Which university has Watts Humphrey attended?", "Which university did Watts Humphrey take part in?", "Which university attended Watts Humphrey?", "Which university has Watts Humphrey visited?", "Which university did Watts Humphrey go to?", "Which university did Watts Humphrey attend?", "What university did Watts Humphrey take part in?", "What university did Watts Humphrey study at?", "What university did Watts Humphrey go to?", "Which university did Watts Humphrey participate in?", "Which university did Watts Humphrey study at?"
    ]
}
```