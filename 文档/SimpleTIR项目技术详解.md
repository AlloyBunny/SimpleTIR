# SimpleTIR 项目技术详解：数据、轨迹、训练架构、工具机制与 3B 冷启动改进

> 本文基于仓库代码、README、SimpleTIR 论文（arXiv:2509.02479）以及本机 3B 训练日志整理。文中会明确区分：**原项目设定**、**本机 3B 复现实验**、**建议改进**。

## 1. 一句话理解 SimpleTIR

SimpleTIR 研究的是：让语言模型在解决数学题时，不是只生成一次完整答案，而是可以反复执行下面的闭环：

1. 阅读题目并进行自然语言推理；
2. 在需要时生成一段 Python；
3. 外部 sandbox 执行 Python；
4. 将执行结果作为新 observation 填回上下文；
5. 模型继续推理、再次调用工具，或者输出最终答案；
6. 用最终答案的规则验证结果作为 RL reward。

一条抽象轨迹可写为：

\[
(q,l_0,f_0,l_1,f_1,\ldots,l_n),
\]

其中：

- \(q\)：题目和工具说明 prompt；
- \(l_i\)：第 \(i\) 次模型生成的语言/代码片段；
- \(f_i\)：执行代码后返回的结果；
- \(l_n\)：最终答案，或没有形成有效动作的失败输出。

SimpleTIR 的论文贡献不只是“给模型一个 Python 工具”，而是指出 **Base 模型进行多轮 TIR Zero-RL 时容易因工具 observation 引入分布漂移，并产生 void turn**。其关键稳定化方法是：识别并 mask/filter 那些既没有代码块、也没有结束答案的中间轮次，使这些无效 token 不参与策略梯度。

## 2. Zero-RL 在这里是什么意思

原论文明确研究从 **Base model** 直接进行可验证奖励 RL，即不先依赖人工 CoT SFT 或专门的 TIR instruction tuning。论文主要实验模型包括 Qwen2.5-7B、Qwen2.5-32B 等 Base 模型。

所以，“使用 Base 做 RL”不是项目错误，而是研究问题本身：

- 单轮 TIR 从 Base 出发可以获得一定提升；
- 直接扩展到多轮时训练不稳定；
- SimpleTIR 试图找出不稳定来源，并尽量维持真正的 Zero-RL 设置。

我们的本机实验则是一个不同问题：**把原本主要在 7B/32B 和更多 GPU 上验证的方法，压缩到 Qwen2.5-3B、4 卡、小 batch 后是否仍有足够探索能力？** 结果显示工程链路能跑，但初始能力和正奖励密度偏低。

## 3. 使用什么数据

### 3.1 本仓库打包的数据

当前训练使用：

| 数据 | 本机条数 | 用途 |
|---|---:|---|
| `simplelr_math_35/train.parquet` | 8,523 | 数学 RL 训练 |
| `deepscaler/train.parquet` | 40,315 | 数学 RL 训练 |
| `simplelr_math_35/test.parquet` | 500 | 当前验证 |
| `deepscaler/aime.parquet` | 30 | AIME 评估 |
| `deepscaler/aime25.parquet` | 30 | AIME 2025 评估 |

当前 3B run 将 `simplelr_math_35/train` 和 `deepscaler/train` 合并使用，总计 48,838 道训练题。

### 3.2 数据是 SimpleTIR 自己造的吗

不是从零自己编写题目。

- **SimpleLR-Math-35** 是 SimpleTIR 作者发布/整理的公开数学数据资产，论文将其作为 RL prompt 数据。当前 parquet 中能看到题目来源式的 `unique_id`、数学 subject、难度 level、标准解答和最终答案。样例风格与竞赛数学/MATH 系列题目一致。
- **DeepScaleR 数据**来自公开的 DeepScaleR 数学 RL 数据资产，SimpleTIR 将其整理成 veRL 所需 parquet schema。
- AIME 数据来自公开竞赛题评测集合。

更准确的表述是：**题目和答案主要来自公开数学数据/竞赛题资源；SimpleTIR 作者做了筛选、组合、格式转换和发布，而不是让模型凭空合成全部题目。**

### 3.3 RL 数据中存了什么

典型行包含：

```text
data_source: simplelr_math_35
prompt: [{role: "user", content: "数学题正文"}]
ability: math
reward_model:
  ground_truth: "118"
  style: rule
extra_info:
  level: 5
  subject: Intermediate Algebra
  answer: "118"
```

重要的是：RL 数据通常只需要 **问题 + 可验证最终答案**，并不要求存放一条监督推理轨迹。数据里的 `solution` 字段可以用于分析或后续构造 SFT，但当前 GRPO 主流程并不会把标准 solution 当作 teacher forcing 标签。

## 4. 训练 prompt 是什么样

核心 prompt 位于 `recipe/simpletir/config/simpletir_trainer.yaml` 的 `data.prompt`。其结构是：

```text
Solve the following problem step by step.
You now have the ability to selectively write executable Python code...
Python code will be executed by an external sandbox...

Code Format:
Each code snippet is wrapped between ``` ... ```.
You need to use print() to output intermediate results.

Answer Format:
You can use final_answer() in code...
You can also use \boxed to return your answer...

User Question:
<数据集中的题目>
```

因此模型知道工具存在，主要依靠的是 **自然语言 prompt 协议**，而不是 OpenAI function calling 那种 JSON schema 或 tokenizer 新增的专用 tool token。

当前配置中的两个动作协议是：

- 代码动作：输出 fenced code block，例如 ` ```py ... ``` `；
- 最终答案动作：文本中形成 `\boxed{...}`，或者在代码中调用 `final_answer(...)`。

## 5. Rollout trajectory 具体是什么样

### 5.1 一个理想的多轮例子

题目：求方程 \(x^2=2\) 的正根近似值。

```text
Prompt / q:
Solve ... You may use executable Python ...
User Question: 求 x^2=2 的正根。

模型第 0 轮 / l0:
可以直接用 Python 验证数值。
```py
import math
print(math.sqrt(2))
```

工具 observation / f0:
Code execution result:
1.4142135623730951

模型第 1 轮 / l1:
因此正根为 \boxed{\sqrt{2}}，数值约为 1.4142135624。
```

最终整条 response 会拼接模型 token 和工具输出，但训练时可以把工具 observation mask 掉，使模型不会被要求“预测 sandbox 返回值”。

### 5.2 当前 3B 真实 validation 例子

验证题是抛物线 `x^2=8y` 中的等边三角形。step 300 的模型大致做了：

1. 正确说出焦点 `(0,2)`；
2. 没有建立对称点 `(-x,y)`、`(x,y)` 的正确几何关系；
3. 错误选择 `(4,0)`；
4. 生成 SymPy 代码求一个与原题无关的距离；
5. 得到 8 并输出 `\boxed{8}`。

真实正确答案是：

\[
16\sqrt 3.
\]

这说明 trajectory 的形式已经成立——模型能生成代码、收到结果、继续回答——但策略能力不足，错误的数学建模被工具“精确地算错”。

### 5.3 原始样例在哪里看

本机 SwanLab run 已保存 step 100/200/300 的文本：

```text
/data/L202500291/outputs/simpletir/swanlab/run-20260712_024923-73dfz9f6/media/text/
```

训练器中的 generation logger 也支持把 validation input/output/score 写入 SwanLab。若设置 `trainer.output_acc_to_file=true`，还能将 good/bad generation 写成本地文本目录。

## 6. 多轮状态机如何判断“调工具”还是“结束”

核心实现在 `recipe/simpletir/agent_utils.py`。

### 6.1 每轮生成后的截断

`_postprocess_responses()` 搜索第一个完整 fenced Python code block。一旦找到，就把该轮后续文本截掉，只保留：

```text
代码块之前的思考 + 第一个代码块
```

这意味着一个完整代码块就是“暂停生成、交给环境执行”的动作边界。

### 6.2 `execute_predictions()` 的三类状态

对每条 active trajectory：

1. **发现完整代码块**：
   - 提取代码；
   - 发给 sandbox；
   - 返回 `Code execution result: ...`；
   - `done=False`，下一轮继续生成。

2. **未发现代码块，但已有最终回答/普通回答**：
   - 认为模型已结束；
   - `done=True`；
   - trajectory 不再参加下一轮。

3. **既无代码块，又不构成有效结束动作的中间输出**：
   - 标记为 void turn；
   - `is_void_turn=True`；
   - SimpleTIR 可 mask 这一轮，避免把分布漂移产生的无效 continuation 用于更新。

这里没有依赖单一的 `<tool_call>` 特殊 token。边界主要由 regex、代码 fence、最终答案格式、最大轮数和 active mask 共同决定。

### 6.3 达到最大轮数怎么办

配置为 `max_turns=5`。在接近最后一轮时，环境会追加提示，要求模型尽快给最终答案；最后仍未结束则强制停止。每轮只有仍处于 `active_mask=True` 的 trajectory 才会进入下一次 vLLM generation。

## 7. TIR 可以使用什么工具

当前数学多轮主流程只有一个外部工具：**Python 代码执行器**。

支持的不是任意 shell，也不是搜索引擎、浏览器、计算器 API 或数据库。模型输出 Python，sandbox 服务执行并返回 stdout/错误信息。

仓库提供两类 sandbox 接口：

- 作者内部高并发 sandbox；
- `sandbox/` 下的本地 HTTP sandbox 示例。

客户端通过环境变量 `SANDBOX_ENDPOINT` 找到服务。当前本机为了安全使用隔离执行服务，并通过 FastAPI/Uvicorn 暴露兼容接口。

`final_answer()` 不是另一个外部工具，它是提供给代码的便捷函数/协议，用来让执行结果表达最终答案。

## 8. 72% code-use 和 8%–18% valid-code 到底是什么意思

### 8.1 剩下的 turn 在做什么

`env/code_use_ratio≈0.72` 表示在已生成的模型轮次中，大约 72% 检测到了代码使用动作。剩下约 28% 可能是：

- 直接进行自然语言推理并给最终 `\boxed{}`；
- 在收到前一次工具结果后，用自然语言总结并结束；
- 只生成自然语言但没有代码，也没有形成最终答案，即 void turn；
- 输出了不完整的代码 fence，因而未被识别为可执行代码；
- 达到长度/轮数限制而结束。

因此“非代码 turn”并不等于异常。一个正常 TIR 轨迹的最后一轮通常就不再调用代码，而是消费 observation 后给出答案。

### 8.2 “前期/后期”指什么

此前说有效代码比例前期约 8%–10%、后期偶尔到 18%，这里的前后指 **训练进程的早期和较后 step**，不是一条 trajectory 内的前几轮和后几轮。

### 8.3 `valid_code` 的准确含义

这里容易混淆，因为代码中有两个层面的指标：

- 环境层 `env/valid_code_ratio`：来自 `AgentHelper.execute_predictions()`，表示识别到了完整代码块，并成功走过 sandbox 执行/返回的代码轮次比例；解析失败、执行异常、timeout 等会记为无效。
- reward extra 中的 `critic/rewards_extra/valid_code`：当前 `hf_math_verify.py` 只是用 regex 检查最终拼接文本中是否存在“代码块后跟 Code execution result”，语义更宽松，并不证明代码数学上正确。

因此日志里 reward extra 的 `valid_code=1` 不能理解成“代码正确”；真正观察工具稳定性更应该看 `env/valid_code_ratio`、成功/失败代码行数和 sandbox 错误分布。

### 8.4 为什么 3B 有效代码率低

常见原因包括：

- Base 3B 对 prompt 协议服从能力弱；
- fence 不完整，语言标记格式错误；
- 使用未定义变量；
- 忘记 import；
- 不调用 `print()`；
- 生成超长或被截断的程序；
- 根据错误数学关系写出可运行但无用的代码；
- 多轮 observation 后发生格式漂移；
- sandbox timeout 或依赖不可用。

从当前真实样例看，最严重的不是 Python 语法，而是 **数学建模错误和工具滥用**。

### 8.5 7B 会高很多吗

合理预期是 7B Base 通常比 3B Base 更容易形成完整代码块、遵守协议并在同样采样预算下偶尔答对难题；原论文也主要在 7B/32B 上报告成功结果。但不能在没有同配置对照实验时声称“必然高很多”。应比较：

- 同一 validation set；
- 同一 temperature、max turns、rollout n；
- code-use ratio；
- sandbox execution success；
- final-answer accuracy/pass@k。

## 9. GRPO 训练和 rollout 如何共享 GPU

### 9.1 当前不是固定切出“训练卡”和“rollout 卡”

本项目默认采用 veRL 的 **hybrid engine / colocated ActorRollout worker**。`main_simpletir.py` 将 `Role.ActorRollout` 映射到同一个 global resource pool；FSDP actor 和 vLLM rollout engine 位于同一组 GPU 资源上。

在当前 4 卡配置中：

- actor 的 FSDP world size 为 4；
- rollout 的 tensor parallel size 为 2；
- 不是 GPU 0–1 永久训练、GPU 2–3 永久 rollout；
- 更接近同一批 GPU 在不同阶段切换角色，并在内存中共置相关 worker/engine。

### 9.2 一个 step 的时序

简化流程是：

1. Ray driver 从 dataloader 取 prompt batch；
2. actor/rollout worker 将 actor 权重同步到 vLLM rollout engine；
3. vLLM 批量采样初始 response；
4. `AgentHelper` 调 sandbox，并只让 active trajectory 继续多轮生成；
5. reward manager 用规则 verifier 打分；
6. rejection sampling/过滤选择有效 group；
7. actor worker 计算 old log-prob、GRPO advantage；
8. FSDP actor 执行 PPO/GRPO update；
9. 下一 step 再次 rollout。

它主要是**阶段式复用**，不是训练和 rollout 永久并行各占一组卡。日志中生成约 60 多秒、actor update 约 0.5 秒，也说明当前小 batch 下大部分 wall time 在 rollout 阶段。

### 9.3 谁负责管理

分三层：

- **veRL**：提供 Ray resource pool、worker group、FSDP actor、vLLM rollout、权重同步、PPO/GRPO 数据流等通用基础设施；
- **SimpleTIR trainer**：在 veRL PPO trainer 上加入多轮 agent loop、sandbox observation、void-turn mask、rejection sampling和 TIR 指标；
- **Ray**：负责跨进程/跨节点 worker 生命周期和 GPU resource scheduling；
- **vLLM**：负责高吞吐 autoregressive sampling；
- **FSDP/PyTorch**：负责训练态参数分片和梯度更新。

所以“边训练边 rollout”不是 SimpleTIR 从零重写的；底层大部分由 veRL 封装，SimpleTIR 的关键增量是多轮工具环境和稳定训练逻辑。

### 9.4 会不会动态调控 GPU 数量

默认不会根据每一步负载自动决定“这次两张卡训练、下次三张卡 rollout”。GPU pool、world size、tensor parallel 等在启动配置时确定。

动态变化的是：

- 每一轮仍 active 的 trajectory 数量；
- padding/batch balance；
- rejection sampling 后 effective batch；
- vLLM 与训练阶段的内存占用和 cache 使用。

若要固定分离 rollout 与 actor GPU，需要使用 veRL 的 split placement/disaggregated 方案或修改 role-resource mapping，不是当前 SimpleTIR 默认设置。

## 10. Reward 为什么稀疏

当前数学 reward 的核心是最终答案规则验证：

```text
正确答案 -> 1
错误答案 -> 0
```

`is_boxed_ratio` 和代码存在性会记录在 extra info 中，但当前 `total_score` 仍等于最终答案 accuracy，它们不是主要优化 reward。

GRPO 又依赖同一 prompt 下多个 rollout 的相对差异。如果单条正确率为 \(p\)，每组采样 \(n\) 条，则全错概率是：

\[
(1-p)^n.
\]

当前 3B 原始 rollout 正确率约 12%，`n=4` 时全错概率约：

\[
0.88^4\approx60\%.
\]

大量 prompt group 没有正样本，组内 advantage 信号很弱。rejection sampling 可以多采样再保留有差异的 group，但会显著增加 rollout 成本，不能创造模型从未探索到的正确推理。

所以你的判断是对的：**reward shaping 不能替代初始可探索性。首先要让模型在有限 N 次采样内偶尔能做对。**

## 11. 3B 是否更需要 Instruct 或冷启动

### 11.1 原项目和我们的判断并不矛盾

- 原论文问题：证明 7B/32B Base 可以做 Zero-RL，并解决多轮 instability；
- 我们的问题：在 3B、4 卡、较小 rollout group 下，如何提高正轨迹密度和训练性价比。

对于后者，使用 Instruct、SFT cold start 或 curriculum 是合理工程改造，即使它不再是严格意义上的 Zero-RL。

### 11.2 为什么小模型更依赖冷启动

小模型通常更容易同时受以下限制：

- 数学先验和长程推理能力较弱；
- instruction following 和输出格式稳定性较弱；
- 工具调用协议更容易漂移；
- 同样 rollout 数下命中正确轨迹的概率更低；
- 多轮错误会复合传播；
- 稀疏 0/1 reward 下，正样本不足导致 GRPO group 方差不足。

DeepSeek-R1 的公开技术路线也区分了纯 RL 的 R1-Zero 与加入少量高质量 cold-start 数据的 R1；后者主要为可读性、格式、稳定性和后续 RL 提供更好的起点。其小模型高性能版本主要通过从更强模型蒸馏获得，也说明对小模型而言，直接依赖稀疏 RL 从零发现完整推理行为通常不是最高性价比路径。

### 11.3 Instruct 会有更好的 TIR 初始表现吗

通常会在以下方面更好：

- 遵循 Python fence/`print()`/`\boxed{}` 协议；
- 正确结束回答；
- 避免 void turn；
- 根据工具 observation 继续回答；
- 基础数学和代码生成成功率。

但这不是无条件保证：

- Instruct 可能过度套模板；
- 可能频繁调用工具；
- 可能被既有 alignment 限制探索；
- 它不一定掌握 SimpleTIR 特有的多轮 observation 格式。

因此最严谨的做法是对 Base、Instruct、TIR-SFT 三个初始化做同一套 128 题 pass@k 和工具有效率评估，而不是仅凭模型名称判断。

## 12. 推荐怎样做冷启动

### 12.1 最推荐：短 TIR-SFT + RL

构造 5K–20K 条高质量轨迹，格式与真实 environment 完全一致：

```text
题目
assistant reasoning + Python block
Code execution result
assistant reasoning / optional second Python block
Code execution result
assistant final boxed answer
```

训练目标只 mask 掉 prompt 和工具 observation，对 assistant token 做 causal LM loss。这样模型学的是：

- 什么时候用工具；
- 如何形成完整代码块；
- 如何读取执行结果；
- 什么时候停止；
- 如何输出可验证答案。

### 12.2 冷启动数据从哪里来

可组合四类来源：

1. **现有数据中的标准 solution**：SimpleLR parquet 已包含部分 `solution`，可转换为无工具或轻工具 SFT；
2. **强模型 teacher 生成**：用 7B/14B/32B Instruct 或高质量 API 模型为现有 RL 问题生成多轮 TIR；
3. **Best-of-N rejection sampling**：对每题采样多条，使用现有 math verifier 自动留下最终答案正确的轨迹；
4. **程序化简单题**：生成算术、方程、组合计数、数值验证等容易自动验算的问题，专门教工具协议，而不是拿它们作为最终 benchmark。

### 12.3 数据过滤标准

必须至少满足：

- 最终答案 verifier 通过；
- 每个代码块可执行；
- observation 与实际执行结果一致；
- 不包含未定义变量和伪造执行结果；
- 工具调用次数合理；
- 最终答案位于 `\boxed{}`；
- 去除 teacher 泄漏、异常长轨迹和大量无意义 print。

可保留一部分**不调用工具的正确答案**，让模型学会“selectively use Python”，否则会形成工具依赖。

### 12.4 低成本版本

如果资源有限：

1. 从 8,523 条 SimpleLR 中选 level 2–3；
2. 用本地 7B-Instruct 每题生成 4–8 条；
3. verifier 留下正确轨迹；
4. 选 2K–5K 条做 1–2 epoch SFT；
5. 用该 checkpoint 开 GRPO，先训练简单子集；
6. 正确率上升后再加入 DeepScaleR 和难题。

### 12.5 Base cold start 还是直接 Instruct

推荐做三组短对照：

- A：3B Base → GRPO；
- B：3B Instruct → GRPO；
- C：3B Base → TIR-SFT → GRPO。

如果只能选一条，我建议 **Base → 少量 TIR-SFT → GRPO**，因为它最能控制数据协议，也最容易把故事讲成“保留 Base 可塑性，同时解决小模型探索不足”。如果追求最快成功，直接从 Instruct 开始通常风险更低。

## 13. 还可以怎样改善

按优先级排序：

1. **修验证**：至少 128 题、`n_val=4`，同时报告 accuracy、pass@4、code execution success、void-turn ratio；
2. **冷启动**：2K–10K 条 verifier-filtered TIR SFT；
3. **课程学习**：先 level 2–3，再逐步混入 level 4–5/DeepScaleR；
4. **提高 rollout n**：从 4 到 8，降低全错 group 比例；
5. **工具选择训练**：混合 tool-needed 与 tool-unnecessary 样本；
6. **轻量 shaping**：最终正确 1.0；有效格式/可执行代码只给很小辅助 reward；无效代码、void turn、小幅惩罚；
7. **长度与轮数课程**：先 1–2 turn，再扩展到 5 turn；
8. **teacher distillation**：蒸馏强模型正确轨迹，而不是要求 3B 从稀疏 reward 独立发现全部策略。

## 14. 必须记住的关键代码路径

| 主题 | 文件 |
|---|---|
| 项目说明与命令 | `README.md` |
| prompt 和默认超参 | `recipe/simpletir/config/simpletir_trainer.yaml` |
| 数据读取与 prompt 拼接 | `recipe/simpletir/utils/dataset/rl_dataset.py` |
| Ray 角色、reward manager、trainer 创建 | `recipe/simpletir/main_simpletir.py` |
| 多轮状态机、代码提取、sandbox observation | `recipe/simpletir/agent_utils.py` |
| GRPO 主循环、rejection sampling、void mask | `recipe/simpletir/simpletir_ray_trainer.py` |
| 数学最终答案 reward | `recipe/simpletir/utils/reward_score/hf_math_verify.py` |
| batch reward 写入 token 位置 | `recipe/simpletir/workers/reward_manager/math_verify.py` |
| veRL FSDP/vLLM worker | `verl/workers/fsdp_workers.py` |
| Ray resource pool 基础设施 | `verl/trainer/ppo/ray_trainer.py` |

## 15. 面试式总结

SimpleTIR 可以概括为：

> 它在 veRL 的 FSDP actor + vLLM rollout 混合训练框架上，实现了一个 Python-sandbox 驱动的多轮数学 agent。模型通过 prompt 学会用 fenced Python 代码表示 tool action，环境执行后将 observation 拼回上下文，active-mask 状态机决定继续生成还是结束。训练只依赖最终答案规则验证进行 GRPO。论文发现 Base 模型从 Zero-RL 扩展到多轮时，会因工具输出分布漂移产生大量 void turn，因此通过 mask 无代码、无答案的无效轮次稳定训练。原论文主要在 7B/32B 上验证；压缩到 3B 和较小 rollout 预算时，核心瓶颈从“训练稳定性”进一步变成“正轨迹探索率”，因此适合引入 TIR cold start、课程学习和更大的 group sampling。

## 16. 外部资料建议

建议重点阅读：

- SimpleTIR: End-to-End Reinforcement Learning for Multi-Turn Tool-Integrated Reasoning，arXiv:2509.02479；
- veRL 文档与 HybridFlow/placement 说明；
- DeepSeek-R1 技术报告中 R1-Zero、cold-start data、RL 和小模型蒸馏部分；
- Qwen2.5-Math 技术报告，用于理解不同规模 Base/Instruct 数学能力与 tool-integrated reasoning 背景；
- DeepScaleR 数据/训练报告，用于理解公开数学 RL prompt 数据来源。
