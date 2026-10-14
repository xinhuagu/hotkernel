<div align="center">

# hotkernel

**找到最耗时的 kernel，把它变快。**

一个 MCP server：让你的编程 agent 能读 Nsight 和 SASS，给它一套它没法作弊的"验证并记录"协议，再配一个它能在你桌上那张显卡上自己跑的 kernel 速通循环。

[![PyPI](https://img.shields.io/pypi/v/hotkernel)](https://pypi.org/project/hotkernel/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/xinhuagu/hotkernel/blob/main/LICENSE)
![MCP](https://img.shields.io/badge/MCP-Claude%20Code%20·%20Codex%20·%20any%20MCP%20client-black)

[English](https://github.com/xinhuagu/hotkernel/blob/main/README.md) | 中文

</div>

今天你让 Claude"把这个 kernel 跑快点"，它只能靠猜：它看不到 Nsight 的报告，不知道哪个 kernel 吃掉了一步的 60%，没读过编译器生成的 SASS，也没法在你的卡上测自己改的版本。hotkernel 就是它缺的那双手。十个工具包住 `nsys`、`ncu`、`cuobjdump`、`torch.profiler` 和一套严格的验证与记录协议；一个 skill 把它们串成循环：剖析、选最耗时的 kernel、读它的 SASS、对照 roofline、用 Triton 重写、对着基线验证、记录裁决、保留或回滚、再来。

**三个目标，三次 agent 自主运行，一张 RTX 5090：rmsnorm 比 PyTorch 朴素实现快 3.2 倍，softmax 比 `torch.softmax` 快 1.2 倍，gemv 与 cuBLAS 持平。agent 保留下来的每个 kernel 都跑到了这张卡实测流式带宽的 95% 到 105%。**¹

> ¹ 每个区间来自对 agent 最终 kernel 的三次独立 `verify()`，每次与基线交替各跑 7 次，一张 RTX 5090，575 W，WSL2。加速比只相对 `examples/` 里的基线脚本：rmsnorm 是 PyTorch 朴素实现，softmax 是 `torch.softmax`，gemv 是 `torch.mv`（cuBLAS）。"实测流式带宽"是同一张卡、同一计时器下的显存拷贝速度，1501 GB/s（规格 1792 GB/s 的 84%），见[标定](https://github.com/xinhuagu/hotkernel/blob/main/docs/platform-matrix.md)。每个数字的原始日志都在 [`results/`](https://github.com/xinhuagu/hotkernel/tree/main/results)。

## 安装

Claude Code 一条命令（Claude Code 2.1.275 及以上）：把本仓库加为插件市场并装上插件，插件 = MCP server + kernel-speedrun skill。

```bash
claude plugin install hotkernel --marketplace xinhuagu/hotkernel
```

在会话里等价于先 `/plugin marketplace add xinhuagu/hotkernel` 再 `/plugin install hotkernel@hotkernel`。插件只有一个选项：装了你 PyTorch 和 Triton 的那个 Python。用 `/plugin configure hotkernel@hotkernel` 设置，或者安装时直接给：`claude plugin install hotkernel@hotkernel --config hotkernel_python=$HOME/venvs/mytorch/bin/python`。留空时 hotkernel 用 PATH 上第一个能 `import torch` 的 `python`，一个都没有就明确报错。server 本身不依赖 torch，以 `uvx hotkernel` 运行。

Codex 和其他 MCP 客户端（Cursor、DSH）仍按普通 MCP 方式配置：

```bash
codex mcp add --env HOTKERNEL_PYTHON=$HOME/venvs/mytorch/bin/python hotkernel -- uvx hotkernel
```

或者在客户端的 MCP 配置里写 `{"command": "uvx", "args": ["hotkernel"], "env": {"HOTKERNEL_PYTHON": "..."}}`，再把 [`skills/kernel-speedrun/`](https://github.com/xinhuagu/hotkernel/tree/main/skills/kernel-speedrun) 复制到 agent 的 skills 目录。不装插件的 Claude Code 也一样：`claude mcp add -e HOTKERNEL_PYTHON=... hotkernel -- uvx hotkernel`。

## 对准你自己的 kernel

`hotkernel new` 把任意一个 PyTorch 函数变成和示例同一契约的目标：这个函数就是基线，它的 fp32 结果是参考实现，容差按 dtype 给默认值，协议 3 的计时器以文件形式放在脚本旁边，所以脚本在你的 torch 解释器里就能跑，那里装没装 hotkernel 都行。

```bash
hotkernel new gelu --from torch.nn.functional:gelu --shape 4096x8192 --dtype bf16 --out targets
```

会生成 `targets/gelu_baseline.py`、`targets/gelu_candidate.py`（一开始就是基线本身，循环改的就是这个文件）和 `targets/_hotkernel_contract.py`。自己写的函数同样用法，额外输入按调用顺序给：`--from ./mykernels.py:fused --extra w:8192`。然后对基线 `bench` 得到要打败的数字，再把没改过的候选 `verify` 一次，改动之前它必须是 `inconclusive`。agent 用 `new_target` 工具做同样的事。在参考卡上，没改过的 gelu 候选 bench 为 0.0952 ms、verify 判 `inconclusive`，故意写错的候选判 `incorrect`。

## 改前 / 改后

所有行：RTX 5090 @ 575 W，bf16，计时协议 3，三次独立 `verify()`，每次 n=7。

| 目标 | 基线 | 基线均值 | agent 的 kernel 均值 | 加速比 | 占实测带宽 | agent 用时 | agent 写了什么 |
|---|---|---|---|---|---|---|---|
| rmsnorm 4096×8192 | PyTorch 朴素实现（5 个 kernel） | 0.287–0.300 ms | 0.088–0.095 ms | **3.17–3.32 倍** | 95–101% | 10 分钟 7 次实验（另一次 3 分钟的运行找到了同一个 kernel） | 每行一个 Triton 程序，整行放寄存器，一读一写 |
| softmax 4096×8192 | `torch.softmax` | 0.107–0.111 ms | 0.090–0.093 ms | **1.17–1.21 倍** | 96–99% | 6 分钟 4 次实验 | 每行一个程序，max、exp、sum 都在寄存器里，8 个 warp |
| gemv 8192×8192 | `torch.mv`（cuBLAS） | 0.088–0.091 ms | 0.085–0.088 ms | **1.03–1.05 倍**，三次中两次显著：与 cuBLAS 持平 | 101–105%² | 7 分钟 3 次实验 | 自动调参的分行块 GEMV，每个程序 2 行 × 1024 列 |

> ² gemv 几乎全是读；拷贝上限是一半读一半写，所以纯读的 kernel 可以略高于它。

[LEADERBOARD.md](https://github.com/xinhuagu/hotkernel/blob/main/LEADERBOARD.md) 上的每一行都由 `record()` 写入，包括没赢的运行，以及一个被 `verify()` 判为 `incorrect` 的故意写错的候选。每次 agent 运行的完整对话记录在 `results/<目标>/agent-run-*/`。

## agent 拿到的工具

| 工具 | 做什么 | 依赖 |
|---|---|---|
| `gpu_info` | 显卡、驱动、显存、功耗墙、计算能力 | `nvidia-smi` |
| `roofline` | 这张卡的拐点；给定 flops 和 bytes，判断带宽瓶颈还是算力瓶颈并给出可达时间，可对照规格或实测带宽 | 无 |
| `torch_trace` | 用 `torch.profiler` 列出脚本里按设备时间排序的 CUDA kernel | 你的 torch |
| `profile` | 在 Nsight Systems 或 Nsight Compute 下跑命令，返回解析后的报告 | `nsys` / `ncu` |
| `top_kernels` | 从已保存的报告里取最重的 n 个 kernel，附 ncu 吞吐和占用率 | 无 |
| `disasm` | 给 cubin 或 Triton 缓存里最新的同名 kernel 出 PTX 或 SASS | `cuobjdump` |
| `bench` | 把脚本跑 n 次，收集它打印的 `{"ms": …, "ok": …}`，返回均值、标准差、最小值 | 你的 torch |
| `verify` | 基线和候选交替各跑 n 次；给出加速比和裁决：`faster`、`slower`、`inconclusive`、`incorrect`（任一次 `ok=false`，不报加速比）、`invalid`（shape、dtype、种子或计时协议被改） | 你的 torch |
| `record` | 把 verify 结果写到 `results/<目标>/`，从日志重新生成排行榜，并告诉 agent 这次改动是否超过本轮最好成绩 | 无 |
| `new_target` | 用你自己的函数生成目标：基线、空白候选和随附的契约文件，与 `hotkernel new` 相同 | 无 |

`torch_trace`、`bench`、`verify`、`record` 在任何装了 PyTorch 的机器上都能用。Nsight 是可选的，只是让图更清楚。

## 循环

[`skills/kernel-speedrun/SKILL.md`](https://github.com/xinhuagu/hotkernel/blob/main/skills/kernel-speedrun/SKILL.md) 就是全部方法：

1. `bench` 基线，这是要打败的数字。
2. `torch_trace` 或 `profile` 找到真正占大头的 kernel。
3. 用 kernel 的 flops 和 bytes 调 `roofline`：带宽瓶颈的要融合，算力瓶颈的要上 tensor core、加大分块。
4. Triton kernel 第一次编译后 `disasm`：寄存器溢出、访存不合并、长依赖链在 SASS 里都看得见。
5. 一个假设、一处改动，然后 `verify(candidate, baseline, n=7)`。只有差距超过 `2 * sqrt(std_b²/n + std_c²/n)` 且候选每次都通过正确性检查，才返回 `faster`。
6. 无论输赢都 `record()`。它写日志和排行榜，agent 从不改排行榜。只有 `record()` 说这次超过了本轮最好成绩才保留，否则回滚。
7. 连续三次改动都没超过最好成绩，或 kernel 已在实测带宽地板的 10% 以内，就停。

它就是把 [autoresearch](https://github.com/karpathy/autoresearch) 的指标从训练 loss 换成 kernel 墙钟时间。

## 实测，不是宣称

让 LLM 写 CUDA kernel 这件事有信誉问题：2025 年 AI CUDA Engineer 宣称 10 到 100 倍，后来被发现绕过了自己的正确性检查；KernelBench 之所以存在，就是因为"编译通过"和"又对又快"是两回事。hotkernel 的回答是结构性的：

- **协议写在基线里。** shape、dtype、种子、参考实现、容差和计时器都在 agent 不能改的基线文件里，候选只提供一个函数。协议里任何一项不一致，`verify` 就返回 `invalid`。
- **计时器错过两次，都是公开改正的。** 第一版复用同一个输入，让 96 MB 的 L2 冒充了显存带宽。第二版用清零缓冲区的方式清 L2，结果每次调用多算了约 45 微秒别人的写回；是 softmax 的 agent 发现 Nsight 和 `verify` 差了一倍。第三版轮换总量超过 L2 两倍的缓冲区，并用五个流式 kernel 标定，它们都落在规格带宽的 79% 到 84%，与 ncu 硬件计数器一致。旧数据保留，标注为不计入。
- **错了会很响。** `ok` 为假的运行 `verify` 不报加速比，每一行都带 n 和标准差，基线就在仓库里，谁都能重跑。

## 平台

- **需要 NVIDIA 显卡。** 参考卡 RTX 5090（sm_120）。roofline 表覆盖 5090、4090、3090 和 H100，每行附来源；在 `src/hotkernel/gpus.py` 里加你的卡。
- **WSL2 是参考平台。** 九个工具全部在 RTX 5090、WSL2 Ubuntu 24.04、驱动 617.42 上验证过：`nsys` 2026.1.3 和 `ncu` 2026.2.1 从 NVIDIA 的 apt 源解包到用户目录（不需要 sudo），`cuobjdump` 和 `nvdisasm` 直接来自 Triton 的 wheel（不需要装 CUDA toolkit）。`ncu` 需要一次性开放 GPU 性能计数器权限（NVIDIA 控制面板，开发者设置）。版本、命令和原始输出见 [`docs/platform-matrix.md`](https://github.com/xinhuagu/hotkernel/blob/main/docs/platform-matrix.md)。
- **原生 Windows：** server 能跑（纯 Python）。如果你的 torch 在 WSL2 里，通过 `wsl.exe` 启动 server；平台矩阵里有完整的 `claude mcp add` 命令。
- **RTX 50 系上的 Triton：** FlashAttention 2/3/4 在 sm_120 上都跑不了，这正是手调 Triton kernel 在这些卡上重要的原因。

## 许可证

Apache-2.0。
