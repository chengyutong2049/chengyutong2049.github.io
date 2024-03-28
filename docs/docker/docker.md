---
layout: default
title: docker
nav_order: 3
has_children: true
permalink: /docs/docker
math: mathjax
---

# Docker
{: .fs-9 }

<!-- To make it as easy as possible to write documentation in plain Markdown, most UI components are styled using default Markdown elements with few additional CSS classes needed.
{: .fs-6 .fw-300 } -->

##  docker ps
List all running containers.
确保你的服务（在 Docker 容器内）已经启动并且正在监听正确的端口。你可以通过运行 docker ps 查看容器是否正在运行，以及通过 docker logs 容器名或ID 查看容器的日志来确认服务是否正常启动。
```bash
(base) yutong@seclab-storage:~/openai-cti-summarizer$ docker ps
CONTAINER ID   IMAGE                   COMMAND                  CREATED          STATUS          PORTS                                             NAMES
81a502c5549d   openai-summarizer:0.1   "uvicorn app.main:ap…"   30 seconds ago   Up 30 seconds   9999/tcp, 0.0.0.0:9998->80/tcp, :::9998->80/tcp   openai-cti-summarizer-openai-summarizer-1
```
* PORTS: This part shows the port mappings and exposures for the container. It's divided into two main parts:
  * 9999/tcp: This indicates that port 9999 inside the container is exposed, but not mapped to any port on the host. It's available for inter-container communication or could be mapped to a host port if specified in the Docker run command or the Docker Compose file.
  * 0.0.0.0:9998->80/tcp, :::9998->80/tcp: This shows a port mapping from the host to the container. Specifically, it means:
    * 0.0.0.0:9998->80/tcp: Port 80 inside the container is mapped to port 9998 on all IPv4 addresses on the host machine. This allows you to access the container's service (running on port 80) via port 9998 on your host machine using IPv4, e.g., http://localhost:9998 or http://<host-ip>:9998.
  
  ## PORTS: - XX:XX
```yaml
ports:
  - "9998:80"
```
  * "9998" 是宿主机（Host）的端口。
  * "80" 是 Docker 容器内部的端口。
  * [[remote_ip:]remote_port[-remote_port]:]port[/protocol]
```bash
ssh -L 1234:127.0.0.1:1234 username@xx.xx.xx.xx
```

* lsof -i :9997 查看端口占用情况
  * sudo lsof -i :9998 在Linux服务器上要加prefix sudo
* kill -9 PID 杀死进程
