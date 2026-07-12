# SimpleTIR 项目面试叙事与简历写法：真实、深入、可追问

> 说明：本文不会建议伪造已经完成的实验、指标或他人工作。面试官一旦追问日志、代码、失败案例和对照实验，虚构内容很容易暴露，也可能构成诚信风险。更好的策略是：把**确实完成的复现、诊断和工程改造**讲扎实，把尚未运行的方案明确称为“设计并准备验证的下一阶段实验”。这同样能体现很强的研究和工程能力。

## 1. 最推荐的故事主线

一句话版本：

> 我在只有 4 张 H100 的条件下，将主要面向 7B/32B、多节点环境的 SimpleTIR Zero-RL 方案缩放到 Qwen2.5-3B，打通了 FSDP 训练、vLLM 多轮 rollout、Python sandbox、规则奖励和断点恢复。复现后发现训练链路正常，但 3B 的正轨迹探索率长期只有约 12%，单题 validation 还造成了“准确率始终为 0”的误判。我进一步定位到瓶颈不是 reward 代码失效，而是小模型在稀疏奖励下缺少初始 TIR 能力，并设计了 verifier-filtered cold start、难度课程和更合理评估协议。

这个故事有四个优点：

1. 有真实工程难点：环境、sandbox、Ray、FSDP、vLLM、checkpoint；
2. 有真实实验失败：3B 没有复现 7B 论文效果；
3. 有可信诊断：区分 val、raw rollout accuracy 和 rejection-sampled batch reward；
4. 有研究改进：cold start、curriculum、group sampling、reward shaping，但不夸大未验证结果。

## 2. 三分钟项目介绍模板

> SimpleTIR 是一个多轮工具增强推理的 RL 项目。模型解决数学题时可以生成 Python，外部 sandbox 执行后把结果返回，模型继续推理，最终用规则 verifier 检查 boxed answer。底层采用 veRL：FSDP 负责 actor 训练，vLLM 负责 rollout，Ray 管理 worker 和 GPU resource pool；SimpleTIR 在此之上增加了多轮 agent loop 和 void-turn filtering。
>
> 原论文重点是在 7B/32B Base 上做 Zero-RL。我受限于 4 卡，选择 3B Base 做低资源复现。我完成了持久化环境、本地隔离 sandbox、4 卡参数缩放、训练断点恢复和 SwanLab rollout 记录。训练到 300 多 step 后，表面上 validation 一直为 0，但我发现验证每次只抽固定一道题，所以这个指标没有统计意义。进一步分析训练日志后，原始 rollout accuracy 实际约为 11%–14%，说明 reward 没坏，模型也偶尔能做对；但曲线没有明显上升。
>
> 我抽取 step 100/200/300 trajectory，发现 3B 能调用 Python，但经常先建立错误数学关系，再让工具精确执行错误公式；同时多轮代码执行成功率偏低。这说明主要瓶颈是小模型初始探索能力，而不是简单增加训练 step 或加格式 reward。于是我设计了 Base→TIR-SFT→GRPO 的 cold-start 路线：用强模型对现有公开题目生成多轮工具轨迹，用规则 verifier 和 sandbox 自动过滤，再从简单题和短轨迹开始 curriculum RL。另一个对照是直接用 3B-Instruct 初始化。即使暂未完成所有长跑实验，这个项目让我完整理解了 agentic RL 从数据、环境、rollout 到优化和评估的全链路。

## 3. 你可以诚实声称自己完成了什么

根据当前仓库和运行状态，可以写成：

- 在 4×H100 环境部署并运行 Qwen2.5-3B 多轮 TIR GRPO；
- 将模型、Conda、缓存、Ray 临时文件、checkpoint、日志全部迁移到持久化存储；
- 搭建兼容 `SANDBOX_ENDPOINT` 的 Python 执行服务，并进行隔离/安全检查；
- 调整 batch size、rollout n、tensor parallel、vLLM memory utilization 以适配 4 卡；
- 支持 checkpoint 保存与 step 100/200/300 断点续训；
- 接入 SwanLab，记录训练指标和 validation generation；
- 发现 validation 仅采样 1 题的问题；
- 区分 raw rollout score、rejection-sampled reward 和 validation accuracy；
- 抽取并分析真实多轮 trajectory；
- 定位 3B 的主要失败模式：错误数学建模、工具滥用、无效代码、正奖励密度不足；
- 设计 cold-start + curriculum + pass@k 评估改进方案。

如果没有真正执行后续实验，不要说“提升了 XX%”。可以说：

- “提出并实现了数据生成/评估脚本框架”；
- “设计了三组初始化对照”；
- “预计降低全错 group 比例”；
- “正在验证/下一步计划验证”。

只有对应代码和运行记录真的存在时，才把“设计”升级为“实现”，把“预计”升级为“达到”。

## 4. 推荐的 STAR 故事

### S：背景

- SimpleTIR 原论文在更大模型和更多 GPU 上做 Base Zero-RL；
- 本地只有 4 张 H100，需要复现完整多轮工具 RL；
- 选择 3B 是资源约束下的主动缩放实验。

### T：任务

- 打通训练、rollout、sandbox、reward、日志、checkpoint；
- 判断 3B 效果差是工程 bug、reward bug、评估 bug，还是模型探索能力不足；
- 提出低资源条件下可行的改进路线。

### A：行动

1. 阅读 trainer、agent loop、reward manager 和 veRL worker；
2. 搭建持久化环境与安全 sandbox；
3. 做一步 probe，再启动长训练；
4. 从日志解析分阶段 raw score、effective batch、void turn、code-use、valid-code；
5. 导出 step 100/200/300 generation；
6. 发现 val sample size=1，修正指标解释；
7. 判断 reward 正常但 raw accuracy 横盘；
8. 设计 TIR-SFT cold start、难度 curriculum、`rollout_n=8` 和分层 validation。

### R：结果

真实结果应该这样说：

- 成功运行到 300+ step，并完成 checkpoint/resume；
- 证明 validation=0 是固定单题评估，不代表全局准确率为 0；
- 训练 raw rollout accuracy 约 11%–14%，未出现明显增长；
- 排除 reward 恒零、NaN、OOM、actor 不更新等致命工程问题；
- 将问题定位为 3B Base 的低正轨迹密度和不稳定工具使用；
- 形成一套可验证的 cold-start 改进实验设计。

失败实验本身也是结果。优秀面试官通常更看重你是否能解释为什么失败、如何排除假设、下一步实验是否能证伪。

## 5. 如果后续有时间，最值得真正补做的实验

为了让故事从“诊断完整”升级为“有改进闭环”，最低成本补做：

### 实验一：初始化对照

在固定 128 道验证题上比较：

- Qwen2.5-3B Base；
- Qwen2.5-3B-Instruct；
- 当前 Base-GRPO step 300。

每题 `n=4`，报告：accuracy、pass@4、code execution success、void-turn ratio、平均轮数。

这不需要长训练，却能直接支撑“Instruct 是否提供更好初始探索能力”。

### 实验二：小规模 cold start

- 选 500–2,000 道简单题；
- 用 7B-Instruct 或其他 teacher 生成轨迹；
- verifier 过滤；
- 训练 1 epoch；
- 比较 SFT 前后 pass@4。

只要证明 pass@4、格式成功率或工具执行率改善，就足以构成可信闭环，不必马上完成 800 step RL。

### 实验三：短 RL 对照

从 Base 与 cold-start checkpoint 各跑 50–100 step：

- 每组有正负 reward 的 prompt 比例；
- raw rollout accuracy；
- effective batch size；
- 每个有效更新消耗的 generation token。

核心论点不是一定要达到论文 benchmark，而是 cold start 是否提高 **sample efficiency**。

## 6. 简历写法

### 6.1 保守且可信版本

**SimpleTIR 多轮工具增强推理低资源复现｜PyTorch、veRL、Ray、vLLM、FSDP、GRPO**

- 在 4×H100 上完成 Qwen2.5-3B Base 多轮 Tool-Integrated Reasoning GRPO 训练链路，打通 vLLM rollout、Python sandbox、规则奖励、FSDP 更新、SwanLab 日志及 checkpoint resume。
- 深入分析 SimpleTIR 多轮 agent 状态机与 void-turn masking，定位固定单题 validation 导致的指标失真，并从真实 rollout 中区分 raw accuracy、rejection-sampled reward 与工具执行成功率。
- 发现 3B Base 在稀疏 0/1 reward 下正轨迹率约 11%–14%、训练早期缺乏提升，设计 verifier-filtered TIR cold start、难度课程和 pass@k 分层评估以提升低资源训练的 sample efficiency。

### 6.2 完成小规模 cold start 后可升级

只有真正跑完后再写：

- 构建 X 条由强模型生成、规则 verifier 与 sandbox 双重过滤的多轮 TIR cold-start 数据，完成 Base→SFT→GRPO 两阶段训练。
- 将 128 题验证集上的 pass@4 从 A 提升至 B、工具执行成功率从 C 提升至 D，并减少全零 reward group 比例。

A/B/C/D 必须来自真实日志；绝对不要临时编数字。

### 6.3 英文简历版本

**Low-resource reproduction and diagnosis of SimpleTIR multi-turn tool-integrated RL**

- Built an end-to-end Qwen2.5-3B GRPO pipeline on 4×H100 with veRL/Ray, colocated FSDP actor and vLLM rollout workers, sandboxed Python execution, rule-based math rewards, experiment tracking, and resumable checkpoints.
- Diagnosed misleading zero validation caused by single-example evaluation; analyzed real trajectories and separated raw rollout accuracy from rejection-sampled batch rewards and tool-execution metrics.
- Identified sparse positive trajectories and unstable tool use as the main bottlenecks for 3B Base models; designed verifier-filtered TIR cold start, difficulty curriculum, and pass@k evaluation to improve sample efficiency.

## 7. 高频追问与回答

### Q1：为什么不用 Instruct，非要 Base？

> 原论文研究的是 Zero-RL，所以先用 Base 是为了保持研究设定。但我发现从 7B 缩到 3B 后，研究瓶颈发生变化：小模型在有限 rollout budget 下很难产生正轨迹。因此我把 Base run 作为严格复现基线，同时设计 Instruct 和 Base+TIR-SFT 对照，研究 cold start 对 sample efficiency 的影响。

### Q2：你怎么证明不是 reward 写坏了？

> 训练 raw reward 并非全零，分阶段约 11%–14%；每个 batch 也有 reward max=1，actor grad norm 正常，没有 NaN。规则 verifier 能在已知正确/错误答案上工作。真正为 0 的是每 100 step 固定抽取的一道 validation 题。

### Q3：为什么 `critic/score/mean` 约 0.37，但 raw accuracy 只有约 0.12？

> 因为开启了 rejection sampling。系统先 oversample，再倾向保留包含正负差异的 group 进行 GRPO 更新，所以进入 PPO batch 的分数分布被条件选择过，不能当作策略在原始采样分布上的 accuracy。raw accuracy 更接近 reward extra 的 score mean。

### Q4：GRPO 为什么需要同题多采样？

> GRPO 不训练显式 value model，而是用同一 prompt 下 group reward 的相对标准化构造 advantage。全对或全错时组内方差接近零，学习信号弱，所以小模型低命中率下需要更大的 rollout group、rejection sampling 或 cold start。

### Q5：工具输出为什么要 mask？

> sandbox observation 不是模型生成的 token。如果对其计算 policy loss，相当于要求模型预测外部执行结果，会污染梯度；而且 observation 分布与预训练文本不同，是多轮 drift 的来源之一。因此只训练 assistant action token，并 mask tool output。

### Q6：什么是 void turn？

> 某一轮既没有形成完整代码动作，也没有形成最终回答，却仍产生了一段 continuation。它可能来自 observation 后的格式漂移。SimpleTIR 将其识别并从 loss 中 mask，避免无效 token 反复强化。

### Q7：为什么 reward shaping 不足以解决？

> 格式 reward 最多教会模型写 fence 或 boxed，不能凭空提供正确数学推理。如果模型从不探索到正确答案，辅助 reward 还可能诱发 reward hacking。冷启动的价值是先提高正确轨迹和有效工具轨迹的支持集，再让 RL 优化策略分布。

### Q8：为什么不直接蒸馏 7B，非要 RL？

> 蒸馏能提供高质量起点，但 RL 可以利用最终答案 verifier 在训练分布上继续搜索和优化。工程上更合理的是 SFT/蒸馏负责行为初始化，GRPO负责基于可验证结果的后训练，而不是把两者视为互斥。

### Q9：训练卡和 rollout 卡怎么分？

> 默认是 veRL hybrid engine，同一 global resource pool 上 colocate ActorRollout worker。FSDP actor 和 vLLM rollout 在阶段间复用同一组 GPU，不是永久静态切卡。Ray 管资源和 worker，veRL 管权重同步与 PPO 数据流，SimpleTIR 增加多轮环境。

### Q10：你最大的工程收获是什么？

> Agentic RL 的指标必须按数据生成、环境执行、reward、过滤、优化、验证逐层拆开。只看一个 dashboard scalar 很容易误诊。尤其 rejection sampling 后的 batch reward 不是原始 policy accuracy，工具“执行成功”也不是数学“使用正确”。

## 8. 必须背熟的数字和事实

只背当前真实可核对的内容：

- 当前模型：Qwen2.5-3B Base；
- 硬件：4×H100 80GB；
- 训练数据：8,523 + 40,315 = 48,838 题；
- 当前训练：`train_batch_size=4`、`rollout_n=4`、`max_turns=5`；
- 验证集：SimpleLR test 500 题，但旧配置每次只抽 1 题、`n_val=1`；
- step 0/100/200/300 单题 validation 都为 0；
- raw rollout score 在前 300 step 约 11%–13%，300 后观察窗口约 14%，但无稳定上升结论；
- code-use ratio 约 72%；
- `env/valid_code_ratio` 多数约 8%–10%，个别 step 更高；
- rollout 每 step 约一分钟，actor update 远短于 rollout；
- reward 核心是最终答案规则验证 0/1；
- Base run 是符合原论文 Zero-RL 动机的基线；冷启动是针对 3B/低资源的改进，不是声称原论文做错。

## 9. 不要说的内容

- 不要说“我复现了论文 SOTA”，当前没有；
- 不要说“800 step 后 accuracy 达到某数字”，除非真实跑完并保存日志；
- 不要把 SimpleTIR 作者实现的 void-turn filtering 说成自己原创；
- 不要把 veRL 的 FSDP/vLLM/Ray 基础设施说成自己从零实现；
- 不要把 public dataset 说成自己构造；
- 不要说代码执行成功率等于代码推理正确率；
- 不要说 Instruct 一定优于 Base，应该说这是待验证假设；
- 不要声称做了 ablation，除非确实存在对照配置和日志。

可替代的诚实表达：

- “我复现并缩放了该方案”；
- “我定位并验证了一个评估问题”；
- “我对失败轨迹进行了归因”；
- “我设计了可证伪的改进实验”；
- “这是下一阶段计划，尚未形成最终 benchmark 结论”。

## 10. 面试前复习清单

### 方法

- TIR 轨迹定义；
- GRPO group-relative advantage；
- rejection sampling 的选择偏差；
- sparse outcome reward；
- tool observation masking；
- void-turn masking；
- cold start、SFT、distillation、curriculum 的区别。

### 工程

- Ray driver/worker/resource pool；
- FSDP actor；
- vLLM rollout；
- ActorRollout colocated hybrid engine；
- sandbox endpoint 和安全隔离；
- active mask 如何减少后续轮次 batch；
- checkpoint 中 actor/optimizer/data state；
- SwanLab generation 文件位置。

### 数据与评估

- parquet schema；
- ground truth 如何进入 reward manager；
- accuracy 与 pass@k；
- Base/Instruct/SFT 初始化公平对照；
- 固定单题 validation 为什么无意义；
- 工具执行成功与最终答案正确的区别。

### 失败案例

至少能手推抛物线等边三角形的正确答案，并解释模型为什么错：

\[
2x=\sqrt{x^2+y^2}\Rightarrow y=\sqrt3x,
\quad x^2=8y\Rightarrow x=8\sqrt3,
\quad s=2x=16\sqrt3.
\]

注意重新检查坐标定义：若另两点为 \((\pm a,b)\)，则 `2a=s`、`a^2+b^2=s^2`，所以 `b=\sqrt3a`；代入 `a^2=8b` 得 `a=8\sqrt3`，最终边长是 `16\sqrt3`。面试时不要重复此前未经复核的错误答案。

## 11. 最终建议

最有说服力的定位不是“我把指标编得很好”，而是：

> 我能把一个复杂 agentic RL 系统真正跑起来，并在结果不符合预期时，从验证采样、rollout 分布、工具环境、reward、过滤和优化链路逐层排查。我理解原论文为什么坚持 Zero-RL，也能说明为什么在 3B 低资源场景下需要改变目标，转向 cold-start 提升 sample efficiency。我不会把框架能力冒充成个人原创，而能明确指出自己的工程贡献、诊断贡献和改进设计。

这种叙事经得住追问，也比一个无法解释来源的“提升了 20%”更容易让技术面试官满意。
