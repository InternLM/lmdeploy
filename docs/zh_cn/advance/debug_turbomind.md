# 如何调试 Turbomind

Turbomind 使用 C++ 实现，不像 Python 一样易于调试。该文档提供了调试 Turbomind 的基本方法。

## 前置工作

首先，根据构建[命令](../build.md)完成本地编译。

## 配置 Python 调试环境

由于目前许多大公司在线上生产环境中使用 Centos 7，我们将以 Centos 7 为例来说明配置过程。

### 获取 `glibc` 和 `python3` 的版本

```bash
rpm -qa | grep glibc
rpm -qa | grep python3
```

结果类似于这样：
```
[username@hostname workdir]# rpm -qa | grep glibc
glibc-2.17-325.el7_9.x86_64
glibc-common-2.17-325.el7_9.x86_64
glibc-headers-2.17-325.el7_9.x86_64
glibc-devel-2.17-325.el7_9.x86_64

[username@hostname workdir]# rpm -qa | grep python3
python3-pip-9.0.3-8.el7.noarch
python3-rpm-macros-3-34.el7.noarch
python3-rpm-generators-6-2.el7.noarch
python3-setuptools-39.2.0-10.el7.noarch
python3-3.6.8-21.el7_9.x86_64
python3-devel-3.6.8-21.el7_9.x86_64
python3.6.4-sre-1.el6.x86_64
```

根据上述信息，我们可以看到 `glibc` 的版本是 `2.17-325.el7_9.x86_64`，`python3` 的版本是 `3.6.8-21.el7_9.x86_64`。

### 下载并安装 `debuginfo` 库

从 http://debuginfo.centos.org/7/x86_64 下载 `glibc-debuginfo-common-2.17-325.el7.x86_64.rpm`、`glibc-debuginfo-2.17-325.el7.x86_64.rpm` 和 `python3-debuginfo-3.6.8-21.el7.x86_64.rpm`。

```bash
rpm -ivh glibc-debuginfo-common-2.17-325.el7.x86_64.rpm
rpm -ivh glibc-debuginfo-2.17-325.el7.x86_64.rpm
rpm -ivh python3-debuginfo-3.6.8-21.el7.x86_64.rpm
```

### 验证

```bash
gdb python3
```

输出类似于这样：
```
[username@hostname workdir]# gdb python3
GNU gdb (GDB) Red Hat Enterprise Linux 9.2-10.el7
Copyright (C) 2020 Free Software Foundation, Inc.
License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.
Type "show copying" and "show warranty" for details.
This GDB was configured as "x86_64-redhat-linux-gnu".
Type "show configuration" for configuration details.
For bug reporting instructions, please see:
<http://www.gnu.org/software/gdb/bugs/>.
Find the GDB manual and other documentation resources online at:
   <http://www.gnu.org/software/gdb/documentation/>.

For help, type "help".
Type "apropos word" to search for commands related to "word"...
Reading symbols from python3...
(gdb)
```

如果显示 `Reading symbols from python3`，说明配置成功。

对于其他操作系统，请参考 [DebuggingWithGdb](https://wiki.python.org/moin/DebuggingWithGdb)。

## 设置符号链接

设置符号链接后，不需要每次都通过 `pip` 进行本地安装。

```bash
# 更改目录到 lmdeploy，例如
cd /workdir/lmdeploy

# 因为编译文件在 build 文件夹中
# 设置 lib 和 compile_commands.json 的软链接
cd lmdeploy && ln -s ../build/lib . && cd .. && ln -s build/compile_commands.json .
```

## Start debugging

```bash
# 使用 gdb 启动 API Server，例如
gdb --args python3 -m lmdeploy serve api_server /workdir/Llama-2-13b-chat-hf

# 在 gdb 中设置 lmdeploy 文件夹路径
Reading symbols from python3...
(gdb) set directories /workdir/lmdeploy

# 使用相对路径设置断点，例如
(gdb) b src/turbomind/models/llama/BlockManager.cc:104

# 当出现
# ```
# No source file named src/turbomind/models/llama/BlockManager.cc.
# Make breakpoint pending on future shared library load? (y or [n])
# ```
# 输入 y 并回车

# 运行
(gdb) r

# (可选) 使用 https://github.com/InternLM/lmdeploy/blob/main/benchmark/profile_restful_api.py 发送请求

python3 profile_restful_api.py --server_addr 127.0.0.1:23333 --tokenizer_path /workdir/Llama-2-13b-chat-hf --dataset /workdir/ShareGPT_V3_unfiltered_cleaned_split.json --concurrency 1 --num_prompts 1
```

## 使用 GDB

参考 [Debugging with GDB](https://ftp.gnu.org/old-gnu/Manuals/gdb/html_chapter/gdb_4.html) 进行调试。