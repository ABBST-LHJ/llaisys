#!/bin/bash
# 依次执行指定的5条指令
pip uninstall llaisys
xmake clean -a
xmake
xmake install
pip install ./python/  -i https://mirrors.aliyun.com/pypi/simple/