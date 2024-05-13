---
layout: default
title: Remote Server
nav_order: 3
has_children: true
permalink: /docs/remote-server
math: mathjax
---

# Remote Server
{: .no_toc }
{: .fs-9 }

<!-- To make it as easy as possible to write documentation in plain Markdown, most UI components are styled using default Markdown elements with few additional CSS classes needed.
{: .fs-6 .fw-300 } -->
### Table of contents
{: .no_toc .text-delta }

## Add server user
manually append the public key to the remote server's `~/.ssh/authorized_keys` file.

## copy file from local to server
```bash
scp -i /Users/cc/.ssh/id_rsa_gpu 本地文件地址 yutong@seclab-storage:服务器地址
```
scp -i /Users/cc/.ssh/id_rsa_gpu Osama-review.zip yutong@seclab-storage:/home/yutong/CTINexus/ground_truth

scp -i /Users/cc/.ssh/id_rsa_gpu AVERTIUM.zip yutong@seclab-storage:/home/yutong/CTINexus



## expediate the download speed
```bash
aria2c --max-connection-per-server 16 `the address of the dataset`
```
## look up the free space in the machine
```bash
free -h
```

## debug file
```python
if __name__ == "__main__":
    sys.path[0] = str(Path(__file__).parent.parent.parent.parent)
```
