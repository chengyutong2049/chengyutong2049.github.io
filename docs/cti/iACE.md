---
layout: default
title: iACE
nav_order: 3
parent: Cyber Threat Intelligence
has_children: true
permalink: /docs/cti/iACE
math: mathjax
---

# Acing the IOC Game: Toward Automatic Discovery and Analysis of Open-Source Cyber Threat Intelligence
{: .fs-9 }

<!-- To make it as easy as possible to write documentation in plain Markdown, most UI components are styled using default Markdown elements with few additional CSS classes needed.

{: .fs-6 .fw-300 } -->
## Challenge
* While IOCs can be extracted from traditional blacklists, like CleanMX [22] and PhishTank [37],
the information delivered by such IOCs is rather thin:
    * only a small number of IOC classes are covered (URL, domain, IP and MD5)
    * the relation between IOCs is not revealed and no context information is provided (e.g., the criminal group behind malfeasance)
    *  Analyzing the cyber-attack campaigns and triaging incident responses become quite difficult when relying on such information.
 *  IOCs from articles in technical blogs and posts in forums are more favorable to security practitioners and extensively harvested, since comprehensive descriptions of the attack are often found there.
    *  Such descriptions are typically informal, in natural languages, and need to be analyzed semantically to recover related attack indicators
    *  For years, this has been done manually by security analysts.
    *  Increasingly, however, the volume and velocity of the information generated from these sources become hard to manage by humans in a cost-effective way.
    *  In our research, we studied over 71,000 articles from 45 technical blogs extensively used by security professionals and found that the number of articles posted here has grown from merely a handful back 10 years ago to over 1,000 every month since last year (see Section 4.1).
    *  Recorded Future is reported to utilize over 650,000 open web sources in 7 languages to harvest IOCs
    *  With the huge amount of information produced by those sources, new technologies are in dire need to automate the identification and extraction of valuable CTI involved
    *  Automatic collection of IOCs from natural-language texts is challenging.
    *  Simple approaches like finding IP, MD5 and other IOClike strings in an article, as today’s IOC providers (AlienVault, Recorded Future) do, does not work well in practice, which easily brings in false positives, mistaking non-IOCs for IOCs
       *  Although three zip files show up in attackrelated articles, MSMSv25-Patch1.zip is clearly not an IOC while the other two are.
       *  Further, even for a confirmed IOC, we need to know its context, e.g., whether it is linked to drive-by download (ok.zip in the figure) or Phishing (clickme.zip)
          *  This can only be done by establishing a relation between the IOC token and other content in the article, such as the terms “downloads”, “attachment” in the example.
       *  
