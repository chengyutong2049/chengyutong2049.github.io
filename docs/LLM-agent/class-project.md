---
layout: default
title: Class Project
parent: LLM Agent
permalink: /docs/llm-agent/class-project
nav_order: 3
has_children: false
math: mathjax
---

# Automatic LLM-based Agent for Cyber Attack
{: .no_toc }
{: .fs-9 }
<!-- {: .no_toc } -->


Group Member(s): Yutong Cheng
{: .fs-6 .fw-300 }


# Project Goal
Design a LLM-based agent that can automatically use implemented tools to conduct attacks in the network environment. 

# Progress made so far
- [x] Implement the LLM-based agent
- [x] Collect the tools for conducting attacks
- [x] Design the network environment for the agent to conduct attacks

# Technical Challenges
- Implement the LLM-based agent
  - The agent need to has both the generalization and locality
    -  generalization: need to understand and summarize the current environment and attack stages
    -  locality: need to know how to use the secific tools to conduct attacks
-  Implement the environment for the agent to conduct attacks
     -  The environment need to be able to simulate the real network environment
     -  The environment need to be able to provide the agent with the necessary information for conducting attacks
-  Craft the tool-usage dataset
      -  Need to collect the parameters and scripts for the tools
      -  Need to collect the attack stages and the corresponding tools
  

# Next Steps
- Training the LLM to understand how to select specific tools for conducting attacks
- Complete the tool-usage dataset

# Evaluation Plan
- Evaluate the agent's performance in conducting attacks