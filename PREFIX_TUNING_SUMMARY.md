# DeltaMem prefix-tuning 工作总结(截至 2026-08-10)

整理自 `/work/mingze/delta-mem` 下的 README、out_prefix_value、out_personamem、
out_ctxmask 及各 formal-arm 日志。共四条实验线,按时间排列。

---

## 0. 一句话结论

**在所有自然任务(Qasper / HotpotQA / PersonaMem)上,prefix 记忆的收益来自
task adaptation(steer/window 分支),不是真正的 episodic binding;唯一干净的
binding 正结果是合成 write-once 单事实任务,而它在 ≥2 facts 和 strong-swap
协议下也随即失效。** "correct − swap" 是全程的判据:每次把它做严格了,
prefix 的优势就消失。

---

## 1. Qasper/HotpotQA prefix-steer 线(7 月中下旬,主 README)

架构:frozen Qwen3-4B-Instruct-2507 + 并联 SWA/prefix 记忆分支
(delta_q/k/v/o 修正,gain 0.1,12 层,64 slots,~16.5M 可训练参数 ≈0.4%)。

核心发现:

1. **有 context 时记忆冗余;no-context 才有价值。** base_noctx 0.036 →
   ours_noctx 0.138(+0.10,6/6 seeds);针对 noctx 训练可到 ~0.19。
   扩容(64/128/256 slots)无效 → 瓶颈在 attention-pool 的 WRITE。
2. **prefix 本身几乎不动、几乎不被读。** init→trained cos≈0.996;真正学到
   东西的是 delta_q/delta_o。prefR/R 只在 L0/L3 非零,L12–L33 恒为 0。
3. **种子间 0.68 vs 0.62 的差距**不在权重统计里,而在 ~31% 的 hard docs 上
   worst seed 退化成 Yes/No(781 vs 127 次)。
4. **ms3 的退化是可定位、可消融的 L33 病理**:L33 的记忆读取 100% 打到与
   文档无关的 prefix 上;mask 掉 L33 prefix 可翻转 6/8 退化文档。

→ 结论:该 checkpoint 的 QA 增益属于 steer/window 适配器,不是 prefix。

## 2. 合成 episodic binding 线(7/29–7/30,out_prefix_value)

任务:每 episode 随机重配 name→city,写一次文档后删除再答题,问题本身
无法确定答案 → 排除 task-adapter 捷径。判据:correct vs window(去掉
prefix 列)vs swap(读别的 episode 写的 prefix)。

**正结果(单事实,512 episodes)**:correct ≈0.99–1.00,window ≈0.01–0.07,
swap ≈0.05–0.07;McNemar p<1e-140。plain-CE prefix-only 也成立 —— pair
batching 和 hinge 只加速优化,不是结果的必要条件。多 seed 稳定。

**边界**:
- 容量:2 facts → correct ~0.48;4 facts → 0.23。多 slot 容量是瓶颈。
- 泛化:held-out names 仍 0.86;held-out names+**values** → 0.000。
  是"已知 value 词表上的新绑定",不是 open-vocabulary copy。

**README 之后的 mq4 strong-swap 补充实验(7/30 晚,未写入 README)**:
grouped 协议,128 writes × 4 queries,strong swap(同 keys、不同 values):

| 架构 | window | swap | correct | correct−swap |
|---|---:|---:|---:|---:|
| pool p64 | 0.061 | 0.229 | 0.230 | +0.002 |
| pool p128 | 0.049 | 0.072 | 0.076 | +0.004 |
| pool p256 | 0.059 | 0.260 | 0.250 | −0.010 |
| steeronly (match/arch) | 0.061 | 0.061 | 0.061 | 0.000 |

→ **strong-swap 下 correct≈swap:4-fact 场景里 prefix 提供的是"有记忆存在"
的增益,不是对正确 episode 的绑定。** 单事实的 binding 结果没有随事实数
扩展。

## 3. PersonaMem-v2 线(7/30–7/31,out_personamem)

协议已锁定并验证(官方 Qwen-VeRL 路径,32k 历史,\boxed{} 严格打分);
baseline 复现:公开 SFT 33.3% vs paper 35.0,Base parsed-subset 31.3% vs
30.5。真正的 bar 是 paper 报告的 GRPO 55.5,不是 SFT 35。

- 8-persona vanilla-CE pilot:correct−swap 全部 ±1.5pt 内 → 引入
  four-choice CE + identity contrast 目标。
- 4-persona architecture gate:**branch dropout 0.5 才能逼 prefix 携带信息**
  (dPrefix 从 ≤+1.9 → +15.2pt);gate 本身无法 rank 架构。
- **Formal dev arms(80 个 unseen dev personas / 2031 queries,五个架构,
  19,200 label exposures,全部跑完)**:

| Arm | window | swap | correct | correct−swap | dPrefix |
|---|---:|---:|---:|---:|---:|
| poolsteer D84 (P=0) | 0.315 | 0.839 | 0.840 | +0.001 | — |
| pool p64 | 0.859 | 0.893 | 0.891 | −0.002 | — |
| prefixonly p64 | 0.315 | 0.864 | 0.864 | +0.000 | — |
| standard(Qasper-native) | 0.867 | 0.891 | 0.893 | +0.002 | — |
| hybridpart + pooldrop0.5 | 0.476 | 0.837 | 0.843 | +0.006 | +0.367 |

判定规则要求 correct−swap ≥ 5pt(目标 10pt)且 persona-clustered CI 离 0
—— **五个 arm 全部未通过**。写入的记忆大幅抬高绝对准确率(0.315 →
0.84–0.89),但 swap 同涨:在 unseen personas 上,记忆分支学到的是
"读到某个 persona 记忆时如何答 MCQ"的通用行为,不是对正确 persona 内容
的条件化。这与第 2 节 strong-swap、第 1 节 Qasper 的结论一致。

(architecture gate 的 75.2 vs 26.0 swap-gap 在 train==eval 的 4 persona
记忆化设置下出现,formal 规模上消失 —— gate 结果不可外推。)

## 4. 后续:DEX control study(8/1–8/5,dex_control_report.md)

同仓库的独立线,不是 prefix tuning:复现/对照 NeurIPS'25 DEX 论文
(arXiv:2505.16333),检验其 `O − λ f_D(O)` 的减号是否有独立作用
(sign-flip 恒等式 + residual-adapter / attn-only 对照)。详见
`dex_control_report.md`。

---

## 5. 当前定位与下一步(截至 7/31 的状态)

- 该方向应表述为 **episodic associative memory / write-then-discard /
  personalization state slots**,不是通用长文档压缩。
- 已证明的正 regime 只有:已知 value 词表、单事实、write-once。
- 三个未解决的瓶颈:①多事实容量(2 facts 即腰斩);②strong-swap 下的
  episode 判别;③unseen-persona 泛化时的 swap 条件化。三者可能是同一个
  问题:**WRITE 端(attention pool)没有形成可寻址的 key→value 结构。**
- PersonaMem README 第 6 节还列着欠的 baseline(BM25/Dense-RAG、Mem0、
  query-only shards),当时在排队。

关键文件索引:
- 主架构与 L33 病理:`README.md`, `investigation/`
- 合成 binding:`out_prefix_value/README.md`, `investigation/episodic_kv_test.py`
- strong-swap 补充:`out_prefix_value/eval512_grouped_strongswap_*.json`
- PersonaMem:`out_personamem/README.md`, `official_dev_sft_*.log`
- DEX:`dex_control_report.md`, `out_dex/`
