---
layout: default
title: AGentV
nav_order: 3
parent: Paper Review
# has_children: true
permalink: /docs/review/Paper_1843_CCS24
math: mathjax
---

AGentV: Access Control Policy Generation and Verification Framework with Language Models
{: .fs-9 }

```markdown
==+== ACM CCS 2024 B Review Form
==-== DO NOT CHANGE LINES THAT START WITH "==+==" OR "==*==".
==-== For further guidance, or to upload this file when you are done, go to:
==-== https://ccs2024b.hotcrp.com/offline

==+== =====================================================================
==+== Begin Review #1843
==+== Reviewer: Peng Gao <penggao@vt.edu>

==+== Paper #1843
==-== Title: AGentV: Access Control Policy Generation and Verification
==-==        Framework with Language Models

==+== Review Readiness
==-== Enter "Ready" if the review is ready for others to see:

Ready

==*== Paper summary
==-== Markdown styling and LaTeX math supported.



==*== Strengths / Reasons to accept
==-==    What are the paper’s important strengths?
==-== Markdown styling and LaTeX math supported.
1. The paper proposes a novel approach to generate and verify access control policies using language models.
2. The proposed method runs effectively on resource-constrained computers and preserves the privacy of the orginazation's data, which is suitable for real-world applications.
3. The proposed method contains a verification module that can verify the generated policies, and tell the error types and highlight the error locations, which is helpful for administrators to refine the incorrectly generated policies before applying them to the system, avoiding the hallucination of the LLMs.
4. The paper conducts extensive experiments to evaluate the proposed method, and the results show that the proposed method outperforms the state-of-the-art methods.
5. The paper releases two annotated datasets and the source code of the proposed method, which is beneficial for the research community to conduct follow up research.


==*== Weaknesses / Reasons to reject
==-==    What are the paper’s important weaknesses?
==-== Markdown styling and LaTeX math supported.
1. The paper used Phi as the base model to generate the access control policies, which is not the state-of-the-art model in the field of language models. The authors claims they used Phi as it is the smallest LLM at the time of writing, but it is still not convincing enough to use Phi as the base model. 
2. The paper should explain the reason for choosing the BART model as the base model for the verification module, and compare it with other models like RoBERTa, BERT, etc. to show the effectiveness of the BART model.
3. In the table 5, the proposed method's performance on extracting the Purpose and Condition components is relatively low compared to the other components. 
4. In the Table 6, the paper only shows the access policy generation performance of AGentV, but not compare it with the state-of-the-art methods. It would be better if the paper can provide a comparison with the state-of-the-art methods to show the effectiveness of the proposed method.
5. In the Table 7, the proposed method's performance on recognizing the incorrect policies is relatively low (63%), especially for the error type of "Incorrect condition" (0%), and the error type of "Incorrect purpose" (40%), which is not convincing enough to show the effectiveness of the proposed verifier module.



==*== Constructive comments for author
==-==    Note to reviewers: please fill this field as you would like to
==-==    receive reviews on your own papers, regardless of outcome. What
==-==    did you like about the paper? How could it be improved?
==-== Markdown styling and LaTeX math supported.

The paper is well-orgainzed and easy to follow. However, the authors should explain the reason for choosing the base models in different modules, (e.g., Phi in the generation module and BART in the verification module), and compare them with other models to show the effectiveness of the chosen models. The authors should also deal with the low performance of the proposed method on extracting the "Purpose" and "Condition" components, and recognizing the incorrect policies, especially for the error type of "Incorrect condition" and "Incorrect purpose". 

==*== Questions for authors’ response
==-==    Specific questions that could affect your accept/reject decision.
==-==    Remember that the authors have limited space and must respond to
==-==    all reviewers.
==-== Markdown styling and LaTeX math supported.

1. Why did you choose Phi as the base model for the generation module? How does Phi compare with other small LLMs like Llama-3-8B, Mistral 7B, etc. in terms of the effectiveness of generating access control policies? Does the real-world application is severely constrained by the model size? What is the reference for this claim?
2. Why did you choose BART as the base model for the verification module? How does BART compare with other models like RoBERTa, BERT, etc. in terms of the effectiveness of verifying the generated policies? 



==*== Does the paper raise ethical concerns?
==-== Choices:
==-==    1. No
==-==    2. Yes
==-== Enter the number of your choice:

(Your choice here) 1. No

==*== Description of any ethical concerns
==-==    If the answer to the above was "no", please skip this question. If
==-==    the answer was "yes", please describe the ethical concerns.
==-== Markdown styling and LaTeX math supported.



==*== Concerns to be addressed during the revision/shepherding
==-==    Specific minor issues that need to be fixed for the paper to be
==-==    accepted for publication. Keep in mind that the additional writing
==-==    needs to fit within the page limits. The intent is not to request
==-==    large amounts of work. Keep in mind that revision is not for
==-==    making major changes to the paper.
==-== Markdown styling and LaTeX math supported.

The authors should explain the reason for choosing the base models in different modules, (e.g., Phi in the generation module and BART in the verification module), and compare them with other models to show the effectiveness of the chosen models. The authors should also deal with the low performance of the proposed method on extracting the "Purpose" and "Condition" components, and recognizing the incorrect policies, especially for the error type of "Incorrect condition" and "Incorrect purpose".

==*== Comments for PC
==-== (hidden from authors)
==-== Markdown styling and LaTeX math supported.

The paper introduces a novel framework for generating and verifying access control policies using language models, designed to be efficient on resource-constrained systems while ensuring data privacy, making it applicable to real-world scenarios. It develops a verification module that identifies and highlights errors in generated policies, aiding administrators in refining policies and preventing errors. However, the paper's choice of Phi as the base model for policy generation and BART for verification raises questions due to a lack of comparison with other advanced models. Additionally, the method's lower performance in extracting specific policy components and recognizing incorrect policies suggests areas for improvement to enhance its effectiveness and reliability.

==*== Reviewer expertise in this domain
==-== Choices:
==-==    1. No familiarity
==-==    2. Some familiarity (I am aware of some work in this domain)
==-==    3. Knowledgeable (I don't necessarily work in this topic domain,
==-==       but I work on related topics or I am cognizant of the work in
==-==       this domain in recent years)
==-==    4. Expert (I've worked on this topic domain or a closely related
==-==       topic domain)
==-== Enter the number of your choice:

(Your choice here) 3. Knowledgeable

==*== Reviewer confidence
==-==    How do you rate your confidence in your assessment of the paper?
==-==    Note that even if you have no familiarity, you may be confident
==-==    that the paper is not great; conversely, even if you are an expert
==-==    in the domain, you may not be confident about the specific paper.
==-== Choices:
==-==    1. Low confidence
==-==    2. Somewhat confident
==-==    3. Quite confident
==-==    4. Confident
==-== Enter the number of your choice:

(Your choice here) 3. Quite confident

==*== Overall merit
==-== Choices:
==-==    1. Reject
==-==    2. Weak reject
==-==    3. Neutral
==-==    4. Weak Accept
==-==    5. Accept
==-== Enter the number of your choice:

(Your choice here) 2. Weak reject

==+== Scratchpad (for unsaved private notes)

==+== End Review
```
