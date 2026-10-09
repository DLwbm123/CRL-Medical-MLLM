# 2026 文献筛选：医疗 MLLM 的持续测试时强化学习

检索日期：2026-10-09。范围：ICML、ICLR、NeurIPS 2026 的公开会议目录，包含普通论文，不限于 Spotlight。本文用于路线选择，没有启动实验、下载数据或模型，也没有变更 P4–P9 协议及预算。

## 结论

这个项目仍有研究空间，但需要先建立可信的学习信号，再研究跨病例的持续适应。最直接的相关工作是 **CTRL**；它已经提出过程奖励、历史轨迹记忆和梯度保护的组合。**RaPO** 是值得借鉴的视觉持续 RL 方法；**CaT、T³RL、R1-Reward** 分别提供无参考答案奖励、工具验证、视觉奖励模型的思路。**Evidence-RL** 和 **Hidden Forgetting / RCL** 提醒我们同时检查答案正确性和视觉证据依赖。

因此，建议研究的问题是：**冻结验证器存在误差且病例分布持续变化时，如何让医疗 MLLM 选择值得学习的经验，并同时保持诊断表现与正确的视觉证据依赖？** 这是待验证的研究问题。把 PRM、记忆、KL 和视觉反事实直接拼在一起，尚不足以证明方法新颖。

## 检索覆盖与阅读深度

| 公开目录 | 提取的全部目录记录 | poster / oral / spotlight 路径记录 |
| --- | ---: | ---: |
| [ICLR 2026](https://iclr.cc/Downloads/2026) | 5,515 | 5,466 |
| [ICML 2026](https://icml.cc/Downloads/2026) | 6,691 | 6,629 |
| [NeurIPS 2026](https://neurips.cc/Downloads/2026) | 9,243 | 9,109 |

上述 21,204 个论文事件记录是目录筛选范围，不能解读为逐篇阅读全文，也不等同于独立去重后的论文数量。按标题检索持续 RL、测试时适应、奖励、遗忘、医疗视觉、证据与校准等主题，取得 475 篇候选的官方摘要；475 次摘要请求均成功。最终选取 46 篇会议目录论文及 3 篇直接相关的 2026 预印本制作选型记录，其中 19 篇核验了正文重点段落，其余依据官方摘要。

“正文重点段落”指本次阅读了与选型相关的方法、监督来源、实验设置或限制，不表示检查了全部附录、证明和代码。“官方摘要”不支持推断未公开的奖励公式、数据划分或计算成本。NeurIPS 论文优先寻找 arXiv 正文；未找到可核验全文的候选保留官方链接，不编造 PDF 地址。

主表的“2026”指会议年份。一些论文的 arXiv 首发在 2025 年，仍属于本次 2026 会议范围；补充预印本没有在上述三份目录中匹配到标题，不冒称三会录用。

机器可读选择记录见 [2026_crl_paper_selection.json](2026_crl_paper_selection.json)。完整目录、候选摘要及检索收据保存在本地 outputs/literature_2026。

## 优先阅读与可迁移机制

| 论文 / 方法 | 2026 会议证据 | 本次阅读 | 对本项目的作用 | 关键边界 |
| --- | --- | --- | --- | --- |
| **CTRL: Continual Test-Time Reinforcement Learning for Large Language Models** | [NeurIPS 官方页](https://neurips.cc/virtual/2026/poster/153321) | 官方摘要 | 最接近目标的直接相关工作：PRM 筛选轨迹，历史认知锚点梯度纠偏 | 未取得全文，不能确认其医疗视觉实验、成本和完整设定 |
| **Overcoming Catastrophic Forgetting in Visual Continual Learning with Reinforcement Fine-Tuning / RaPO** | [NeurIPS 官方页](https://neurips.cc/virtual/2026/poster/151854) | [正文](https://arxiv.org/html/2605.09640) | 保留奖励改变组内轨迹排序；优势归一化状态跨任务保留 | 原任务奖励使用类别 / 框真值，不能直接当无标签测试时方法 |
| **Reinforcement Fine-Tuning Naturally Mitigates Forgetting in Continual Post-Training** | [ICML 官方页](https://icml.cc/virtual/2026/poster/61669) | [正文](https://arxiv.org/html/2507.05386) | 持续多模态 RL 的基线、可学习样本过滤及遗忘评价 | 使用最终答案真值；“较少遗忘”不等于不会遗忘 |
| **RL's Razor: Why Online Reinforcement Learning Forgets Less** | [ICLR 官方页](https://iclr.cc/virtual/2026/poster/10011312) | [正文](https://arxiv.org/html/2509.04259) | 解释 on-policy RL 的保守更新倾向 | 当前任务 KL 与遗忘的经验关系不是任意旧任务的保护保证 |
| **Beyond Majority Voting / SR-TTRL** | [ICML / PMLR](https://proceedings.mlr.press/v306/wu26az.html) | 会议 PDF 重点章节 | 按不同答案组成候选池，再反思选择伪标签；保留正确少数派的机会 | 同一策略仍负责反思，可能强化共同的错误；未证明医疗持续适应 |
| **Tool Verification for Test-Time Reinforcement Learning / T³RL** | [NeurIPS 官方页](https://neurips.cc/virtual/2026/poster/153674) | [正文](https://arxiv.org/html/2603.02203) | 将外部可执行证据引入伪奖励 | 主要是数学及可执行逻辑；医学影像没有同等确定的程序判据 |
| **Breaking the Self-Confirming Loop / RLER** | [ICML 官方页](https://icml.cc/virtual/2026/poster/61643) | [正文](https://arxiv.org/html/2510.08977) | 分开分析奖励噪声、策略耦合、错误过奖；用多模型降低耦合 | 集成增加成本；相关错误不能靠多次投票自动消失 |
| **Co-rewarding** | [ICLR 官方页](https://iclr.cc/virtual/2026/poster/10008259) | [正文](https://arxiv.org/html/2508.00410) | 问题改写交叉监督、EMA 教师，减少即时自奖励耦合 | EMA 教师并未冻结；跨视角一致仍可能共同错误 |
| **Self-Harmony** | [ICLR 官方页](https://iclr.cc/virtual/2026/poster/10008756) | [正文](https://arxiv.org/html/2511.01191) | 原问题与重构问题共同评价答案，联合 Solver / Reframer | 数学推理证据；医疗改写必须保持否定、侧别、数量及病情语义 |
| **Compute as Teacher / CaT** | [ICML 官方页](https://icml.cc/virtual/2026/poster/61605) | [正文](https://arxiv.org/html/2509.14234) | 冻结锚点合成伪参考，再用结构化条目和独立 judge 评分 | HealthBench 是医疗文本；未解决影像证据核验和跨病例遗忘 |
| **Rubrics as Rewards / RaR** | [ICLR 官方页](https://iclr.cc/virtual/2026/poster/10008552) | [正文](https://arxiv.org/html/2507.17746) | 把医学评价拆成可审计的奖励条目 | 原方案生成条目使用参考答案；不能用测试真值生成奖励 |
| **Noise-corrected GRPO** | [ICML 官方页](https://icml.cc/virtual/2026/poster/61804) | [正文](https://arxiv.org/html/2510.18924) | 校准奖励误判率，并修正组相对优化信号 | 需要有标注校准组及噪声假设；分布漂移会破坏固定误判率 |
| **R1-Reward** | [ICLR 官方页](https://iclr.cc/virtual/2026/poster/10011570) | [正文](https://arxiv.org/html/2505.02835) | 可作为冻结视觉 judge 候选，先比较奖励排序能力 | 通用视觉偏好 / 正确性不是医疗诊断验证能力 |
| **Evidence-RL** | [NeurIPS 官方页](https://neurips.cc/virtual/2026/poster/150393) | [正文](https://arxiv.org/html/2608.08021v1) | 用证据区域与匹配非证据区域的反事实差值区分图像依据 | 原训练结合正确性奖励，并使用对象框空间先验；不是无监督 CRL |
| **Hidden Forgetting / RCL** | [NeurIPS 官方页](https://neurips.cc/virtual/2026/poster/153314) | [正文](https://arxiv.org/html/2607.02020) | 旧答案正确时也检查视觉 / 文本证据依赖漂移 | 原方法是有目标答案的持续微调；冻结旧依赖也可能保留旧错误 |
| **MedVR** | [ICLR 官方页](https://iclr.cc/virtual/2026/poster/10008521) | [正文](https://arxiv.org/html/2604.08203) | 医疗视觉重定位、工具调用与轨迹筛选 | annotation-free 指中间步骤 / 定位标注，最终答案真值仍用于 RL |
| **ECHO** | [ICML 官方页](https://icml.cc/virtual/2026/poster/63137) | [正文](https://arxiv.org/html/2602.02150) | 自适应探索与优化强度，可作效率对照 | 置信度与熵主要来自模型自身，不提供独立医疗正确性信息 |

### 直接 CRL 工作如何选

**CTRL 必须放在 related work 的最前面。** 官方摘要明确把错误伪标签累积与历史梯度干扰联系起来，再分别使用 PRM 和历史轨迹锚点处理。我们此前考虑的“过程奖励 + 记忆 + 梯度保护”与之明显重合。由于 OpenReview PDF / 公共 API 本次未返回可读全文，当前不能据此声称 CTRL 已在医疗 MLLM 上解决问题，也不能复述其未核验的实验数字。[官方摘要](https://neurips.cc/virtual/2026/poster/153321)

**RaPO 更适合作为机制基线。** 它冻结上一阶段策略，将轨迹相对锚点的漂移映射为保留奖励，在计算组相对优势之前加入；另用跨任务持久的 EMA 标准差稳定优势尺度。它和单纯把 KL 系数从小调大并不等价。但其类别正确性和检测 IoU 奖励依赖真值，迁移到本项目首先需要替换并校准任务奖励。[方法与奖励定义](https://arxiv.org/html/2605.09640)

RFT-CPT、RL's Razor 和 Retaining by Doing 支持把 on-policy RL 纳入持续适应对照。它们的结论应理解为特定设置下的相对优势；错误奖励或强分布变化下，RL 仍会遗忘。不能把这些文献作为“只要改用 RL 就能解决项目”的证据。[RFT-CPT](https://arxiv.org/html/2507.05386)、[RL's Razor](https://arxiv.org/html/2509.04259)、[Retaining by Doing](https://icml.cc/virtual/2026/poster/64375)

### 奖励路线如何选

优先借鉴 **CaT 的冻结参考与可审计条目**，同时让候选视觉 judge 直接观察图像。原 CaT 合成机制主要汇总回答；将其用于医疗影像时，不能假设这些文字已经完整包含病灶证据。基于模型回答生成的条目依然可能继承模型偏差。[CaT](https://arxiv.org/html/2509.14234)

**T³RL 提供的是核验范式。** 原实现把推理转成 Python，执行后加权投票。代码正确执行只验证编码进去的计算 / 约束，不会证明输入的影像观察正确。本文也报告弱 verifier 可能使奖励更不稳定。因此医疗版本需要可核验的观察或工具输出，而不是把所有疾病判断都包装成程序。[T³RL 方法与限制](https://arxiv.org/html/2603.02203)

**SR-TTRL、Co-rewarding、Self-Harmony 是有价值的伪奖励对照。** 它们比只用多数答案提供更丰富的监督，但仍有同源错误。先比较它们能否在正确少数派出现时恢复正确排序，再判断是否值得进行权重更新。SR-TTRL 默认训练配置仍使用 32 条轨迹；论文的小采样实验不能直接视为本项目算力成本保证。[SR-TTRL](https://proceedings.mlr.press/v306/wu26az.html)、[Co-rewarding](https://arxiv.org/html/2508.00410)、[Self-Harmony](https://arxiv.org/html/2511.01191)

**Noise-corrected GRPO 值得作为条件性备选。** 本文的“无偏”依赖所建模的奖励翻转机制及其估计，不意味着对任意医疗 judge 都无偏。除了错误过奖，还要检查组内方差及归一化后的优势是否保留正确方向；不能仅修正一个标量均值就认为完成迁移。[校准设置与假设](https://arxiv.org/html/2510.18924)

### 医疗视觉证据如何选

**Evidence-RL 可迁移的是局部反事实诊断。** 它比较证据区域被替换后的答案支持下降与同尺度参考区域的下降；高视觉敏感性本身并不等于答案正确。原实验还有正确性奖励及 COCO 对象框空间先验。医疗使用需要适合病灶的区域提议，并验证干预没有仅制造伪影；不能把原完整训练流程称为无需医学标签。[Evidence-RL](https://arxiv.org/html/2608.08021v1)

**RCL 可迁移的是评价维度。** 正确答案可能仍保留，但支持它的视觉依据已变。旧准确率、旧视觉依赖和错误依赖纠正应分别报告。原论文依赖先前 checkpoint 的证据分配，适合保护已有正确行为；这也可能保护错误的语言捷径。其附录还出现与主体任务不一致的段落，本次不把全部附录实证作为已复核结论。[RCL](https://arxiv.org/html/2607.02020)

**MedVIGOR 与 ASSET 值得后续取得全文。** 前者研究医疗证据内化，后者直接研究医疗 VLM 的采集质量变化下测试时适应。两者目前只有官方摘要可用于本次选型：ASSET 摘要未证明无标签 RL，MedVIGOR 摘要未证明跨病例持续适应。[MedVIGOR](https://neurips.cc/virtual/2026/poster/150235)、[ASSET](https://neurips.cc/virtual/2026/poster/153026)

## 扩展候选：保留作为基线或邻近文献

下列条目均核对了官方摘要；没有以摘要替代全文核验。

| 方法 / 论文简称 | 官方来源 | 可借鉴内容 | 为什么不能直接替代目标方法 |
| --- | --- | --- | --- |
| Retaining by Doing | [ICML](https://icml.cc/virtual/2026/poster/64375) | on-policy 数据对遗忘的作用 | 后训练比较，不是无标签医疗测试流 |
| Visual-Necessity-Gated Continual Tuning / VNG-CT | [NeurIPS](https://neurips.cc/virtual/2026/poster/149631) | 视觉必要性决定参数更新路径 | 目标答案似然门控；测试奖励来源待解决 |
| DiGPro | [NeurIPS](https://neurips.cc/virtual/2026/poster/154350) | 削弱近期梯度的主导干扰方向 | 不能纠正奖励本身的系统错误 |
| Manifold-Aware Gradient Projection / MGP | [ICML](https://icml.cc/virtual/2026/poster/61115) | 长时 TTA 中的方向泄漏与保护子空间 | 通用 TTA；摘要不足以确认适用自回归医疗 RL |
| Adaptive and Selective Reset / ASR | [ICLR](https://iclr.cc/virtual/2026/poster/10011949) | 按崩溃风险选择性重置 | 重置可能丢失累积能力，需要计入持续评价 |
| CARE | [ICLR](https://iclr.cc/virtual/2026/poster/10006707) | 专家分割模型提供显式医疗 ROI 证据 | 多模块训练系统，部署和奖励成本待核验 |
| Ophiuchus | [ICML](https://icml.cc/virtual/2026/poster/62829) | 决定何时、何处调用医疗视觉工具 | 多阶段有监督 / agentic 训练，不是现成测试时 CRL |
| MedR2FT | [NeurIPS](https://neurips.cc/virtual/2026/poster/150417) | 医疗推理轨迹 bootstrap 与语义奖励 | 蒸馏轨迹质量和标签来源需要全文确认 |
| MMedAgent-RL | [ICLR](https://iclr.cc/virtual/2026/poster/10011724) | RL 优化分诊与专家整合 | 多代理协作目标；单模型持续保留并非摘要主张 |
| MedEvoEval | [NeurIPS](https://neurips.cc/virtual/2026/poster/139548) | 纵向临床情景、资源成本、经验写回评价 | 模拟医生代理基准，不是已验证的 MLLM CRL 算法 |
| MedZERO | [NeurIPS](https://neurips.cc/virtual/2026/poster/155563) | 临时探索与长期知识分离 | 主要是开放医疗推理和知识工具，不等同影像适应 |
| JitRL | [ICML](https://icml.cc/virtual/2026/poster/61517) | 经验记忆估计优势并调制 logits | 无梯度的策略调整；不证明权重适应，环境反馈不同 |
| Future-Gain Guided TTL | [ICML](https://icml.cc/virtual/2026/poster/65581) | 学习能降低后续解码不确定性的 token | “future” 指后续生成，并非后续病例准确率 |
| Reasoning Cache | [ICML](https://icml.cc/virtual/2026/poster/64887) | 短程训练支持长程推理与摘要缓存 | long horizon 主要是单题解码长度，不是病例流遗忘 |
| Never Stop Learning / EXPLORE | [NeurIPS](https://neurips.cc/virtual/2026/poster/153607) | 物理反馈驱动 VLA 的测试时 RL | VLA 有交互和物理奖励；医疗 VQA 不具同样反馈 |
| Ergodic Risk Measures | [NeurIPS](https://neurips.cc/virtual/2026/poster/152958) | 持续 RL 的长期风险目标 | 理论支撑，不能直接给出医疗奖励实现 |
| Fast and Meta Knowledge Learners | [ICLR](https://iclr.cc/virtual/2026/poster/10007645) | 快速迁移与慢速整合分工 | 像素 / 连续控制证据，不是大模型医学适应 |
| HERD / World Models | [NeurIPS](https://neurips.cc/virtual/2026/poster/150468) | 不确定奖励规划与经验质量分层 | 依赖环境动力学 / world model，任务不同 |
| Permanent and Transient Representations | [NeurIPS](https://neurips.cc/virtual/2026/poster/149172) | 长期参数知识和短期非参数适应分离 | 值函数与交互环境，迁移成本未核验 |
| ViToS | [ICML](https://icml.cc/virtual/2026/poster/62640) | 医疗视觉 token 剪枝与问答联合 RL | 稀疏 token 节省不是已测的本项目 GPU 加速 |
| PerceptionRubrics | [ICML](https://icml.cc/virtual/2026/poster/61890) | 必须正确的视觉事实采用门控评分 | 原有 gold caption / rubric 不能从测试答案构造 |
| TreeVGR / TreeBench | [ICLR](https://iclr.cc/virtual/2026/poster/10006404) | 定位与回答联合评价 | 有标注自然图像，不能直接作为无标注医学奖励 |
| GEAR-Align | [NeurIPS](https://neurips.cc/virtual/2026/poster/153068) | 根据视觉必要性与特异性分配梯度 | caption 后训练，不是已经验证的持续 RL |
| ILVAD | [ICML](https://icml.cc/virtual/2026/poster/63514) | 推理时保留视觉证据、无训练基线 | 生成过程的视觉遗忘不同于跨病例参数遗忘 |
| CasePlay | [NeurIPS](https://neurips.cc/virtual/2026/poster/153097) | 文档证据驱动自博弈与 rubric judge | 病例报告不等于独立医学影像真值 |
| MedAgent-Pro | [ICLR](https://iclr.cc/virtual/2026/poster/10008810) | 医学指南检索、工具观察和证据反思 | agentic 工作流，摘要不支持参数 CRL 主张 |
| Deployed RL should be Continual | [ICML](https://icml.cc/virtual/2026/poster/67195) | 部署后非平稳性的研究动机 | position paper，不能作为算法效果证据 |

## 补充：相关的 2026 预印本，三会身份未核验

这些论文的标题没有在本次三个完整目录中匹配到。它们有助于分析，但不纳入“已在三会目录列示”的统计。

| 论文 | 正文 | 对项目的启发 | 已核验边界 |
| --- | --- | --- | --- |
| **RL Forgets! Towards Continual Policy Optimization / CPO** | [arXiv 2607.04364](https://arxiv.org/html/2607.04364) | 旧任务分布上的行为 KL 才与旧回报保护直接相关；以参数移动代理构造累积保护掩码 | 奖励使用真值；代理依赖局部近似，不能保证无遗忘 |
| **Taming the Implicit / RAPO** | [arXiv 2608.03660](https://arxiv.org/html/2608.03660) | 同时调节样本更新强度和采样频率 | 可靠性 / Fisher-inspired 指标是代理，不是独立医学正确性核验 |
| **Reasoning Portability / RDB-CL** | [arXiv 2605.18903](https://arxiv.org/html/2605.18903) | 对可复用旧推理与需要探索的新样本采用不同 KL | 是持续 RLVR 后训练；旧策略置信度不等于正确性 |

区分 **RaPO**（NeurIPS 目录论文，轨迹保留奖励）与 **RAPO**（补充预印本，双通道风险控制）：名字相近，机制和证据不同。

## 与当前项目证据的连接

现有 [P8 报告](p8_report.md) 已显示：多数一致的错误会得到正奖励，正确少数派会遭受负优势；输出协议也影响严格自由生成评分。局部梯度探针不能单独证明某个损失分量造成全部损害。

这使 RLER / SR-TTRL 的奖励失配分析比继续调熵或学习率更相关。RaPO / CPO / RCL 则提供三个不同的保留视角：轨迹奖励、旧行为近似、证据依赖。文献支持分别检验这些因素，不支持从 P8 直接推断 SPINE 机制在所有场景都失败。

## 建议的研究顺序与最小对照

1. **先验证冻结奖励的区分能力。** 在另行授权的开发组上，固定候选回答，比较多数奖励、反思 / 交叉一致、冻结视觉 judge 和证据条目。量化错误过奖、正确少数派召回、排序一致性、组内有效优势及分布变化后的失效。开发组标签仅用于校准 / 离线审计；最终测试标签不进入教师、伪参考、rubric 或更新。
2. **先比较不更新权重的强基线。** 同一教师的重排序 / best-of-N、工具推理和检索记忆先建立效果与成本。如果收益仅来自教师额外推理，尚不能归因于 RL 适应。
3. **奖励达标后再比较持续更新。** 使用同一已验证奖励，逐项比较普通 GRPO / RLOO、RaPO 式保留、历史行为保护、证据依赖保护。先只变一个主要机制，不把多个模块一次加入而失去归因。
4. **按病例流评价。** 主读出在病例到达时、更新之前完成，随后才用无真值反馈更新；观察后续未适应病例是否受益。相同病例更新后再次作答可以单独报告为 transductive 结果，不能代替持续迁移证据。每个阶段另测固定旧能力组及视觉证据依赖。
5. **单列效益、损害与成本。** 当前病例、未来病例、旧病例，答案准确率、严格解析率、视觉依赖和拒绝 / 不更新覆盖率，均分别报告。采用相同骨干、教师预算、rollout / token 上限和计算口径；不能只比较最终均值。

上述是文献导出的研究建议，并非新冻结实验方案或启动许可。数据划分、教师选择、三个未用 seed、完整矩阵成本及独立期限需要在未来明确授权后锁定。本次没有用未实测的效率假设绕过既有预算准入。

## 创新空间与停止条件

目前至少三种简单主张已有明显邻近工作：PRM + 历史梯度保护已有 CTRL；局部视觉反事实奖励已有 Evidence-RL；持续证据依赖保护已有 RCL / VNG-CT。因此，“把通用方法用到医学”可以是有价值的实证工作，但不能自动宣称新的 CRL 机制。

可能的贡献需要靠证据建立：当验证器在跨设备 / 跨病种流中逐渐失准时，如何识别并减少错误经验写入，同时不放弃真实困难病例的学习；以及怎样保护已验证的诊断证据，又允许纠正旧模型的错误依赖。低置信拒绝、风险门控及校准也有大量邻近工作，仍需进一步对照，本文不保证该问题的新颖性。

若冻结教师不能在独立开发组上超过原多数奖励，先停止参数 RL；若教师重排序已经获得相同收益而更新没有带来后续病例收益，调整任务定位；若独立开发组上仍无法同时得到适应收益和可接受保留，停止当前具体机制。项目可以继续，但不应以必须获得正结果为终点。

## 未核验项

没有复现实验或运行作者代码，没有核验全部附录及证明，也没有逐篇读取无关论文全文。CTRL、MedVIGOR、ASSET 等仅取得官方摘要；已读 arXiv 版本也未逐页核对最终 camera-ready。会议身份依据 2026 官方目录及可取得的会议论文页，未对所有 OpenReview 决定记录逐项复审。论文报告的收益不能直接推算本项目的效果、临床可靠性或 GPU 成本。
