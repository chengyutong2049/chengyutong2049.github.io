---
layout: default
title: LLM Knowledge Editing
nav_order: 3
has_children: true
permalink: /docs/LLM-knowledge-editing
math: mathjax
---

# LLM Knowledge Editing
{: .fs-9 }

<!-- To make it as easy as possible to write documentation in plain Markdown, most UI components are styled using default Markdown elements with few additional CSS classes needed.
{: .fs-6 .fw-300 } -->
## Problem Definition

* **Model Editing**: aims to adjust an intial base model's ($$f_{\theta}$$) behavior on the particular edit descriptor $$(x_e, y_e)$$ without influencing the model behavior on other samples. The ultimate goal is to create an edited model $$f_{\theta_e}$$
* The edited model $$f_{\theta_e}$$ should satisfied:
  * **Reliability**: The reliability is measured as the average accuracy on the edit case:

    ![](../../assets/images/LLM-knowledge-editing/survey(2).png){:width="400"}

  * **Generalization**: The edited model $$f_{\theta_e}$$ should also edit the equivalent neighbour $$N(x_e, y_e)$$. It is evaluated by the average accuracy of the model $$f_{\theta_e}$$ on examples drawn uniformly from the equivalence neighborhood $$N(x_e, y_e)$$:
  
    ![](../../assets/images/LLM-knowledge-editing/survey(3).png){:width="400"}

  * **Locality**: or Specificity. Edited model $$f_{\theta_e}$$ should not change the output of the irrelevant examples in the out-of-scope $$O(x_e, y_e)$$. The locality is evaluated through the rate at which the edited model $$f_{\theta_e}$$'s predictions are unchanged as the pre-edit $$f_{\theta}$$ model:
    
    ![](../../assets/images/LLM-knowledge-editing/survey(4).png){:width="400"}