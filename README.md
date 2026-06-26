# AdaFlow-VLA

**AdaFlow-VLA** 是一个面向机器人操作（Robotic Manipulation）的**视觉-语言-动作大模型（Vision-Language-Action, VLA）**。

本仓库提供的是 **轻量级评测客户端 + 模型架构规范**：

- 真正的模型权重与推理引擎部署在云端（`https://api.sai0.ai`），通过 RESTful API 对外提供动作预测服务。
- 本地无需任何模型代码或 GPU，只依赖 `requests + numpy + Pillow` 即可调用模型。
- 内置 [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) 机器人操作基准的远程评测脚本，开箱即用。
- `model/` 目录提供完整的**模型架构规范（接口定义）**，核心实现为闭源专有部分。

---

## 目录结构

```
saivla-0-main/
├── __init__.py          # 包入口，导出 Sai0VLAClient
├── cli.py               # 命令行入口 sai0-eval
├── client.py            # 远程推理客户端核心（HTTP 封装）
├── libero_eval.py       # LIBERO 基准远程评测主流程
├── utils.py             # 工具函数（图像/状态提取、视频保存、汇总）
└── model/               # 模型架构规范（接口骨架，实现专有）
    ├── architecture.py  # 网络拓扑与模块接口定义
    └── config.py        # 配置 Schema 与超参定义
```

---

## 模型结构

AdaFlow-VLA 是端到端的 VLA 模型：输入**多路相机图像 + 自然语言指令 + 机器人本体状态**，输出**未来若干步的动作序列（Action Chunk）**。

```
        ┌──────────────────────────────────────────────────────────┐
        │                       AdaFlow-VLA                          │
        │                                                            │
        │   Images ──┐                                               │
        │            ├──► VLM Backbone ──► Multi-Scale Features       │
        │   Text   ──┘        │                    │                  │
        │                     │                    ▼                  │
        │                     │         ┌─────────────────────┐       │
        │                     │         │  Feature Fusion      │       │
        │                     │         │  (Adaptive Routing)  │       │
        │                     │         └─────────┬───────────┘       │
        │                     │                   │                   │
        │   Proprio ──────────┼───────────────────┤                   │
        │                     │                   ▼                   │
        │                     │         ┌─────────────────────┐       │
        │                     │         │  OFT Action Head     │       │
        │                     │         │  (Transformer-based) │       │
        │                     │         └─────────┬───────────┘       │
        │                     │                   │                   │
        │                     │                   ▼                   │
        │                     │            Action Chunk               │
        │                     │           (H × action_dim)            │
        └──────────────────────────────────────────────────────────┘
```

### 1. VLM 主干（VLM Backbone）

- 可插拔的视觉-语言模型主干，支持 `eagle2_5_vl` / `qwen3_vl` / `siglip_so400m`。
- 输入原生分辨率 384，支持长上下文视觉 token 的 RoPE 缩放。
- **多尺度特征提取**：从多个 Transformer 层（默认 `[-3, -2, -1]`）抽取隐藏状态，同时捕捉底层空间细节与高层语义信息。

### 2. 多尺度特征融合（Multi-Scale Feature Fusion）

- 采用 **自适应特征路由（Adaptive Feature Routing）**：通过一个可学习的门控网络，根据当前观测与指令动态地为每一层特征分配权重，而非简单拼接。
- 融合策略可选 `concat` / `adaptive_gate` / `learned_weighted_sum`。

### 3. 本体感知编码器（Proprio Encoder）

- 将机器人状态（8 维：双指夹爪 + 末端位置 xyz + 末端姿态轴角）注入动作头。
- 支持多种条件注入方式：`concat` / `film`（默认，Feature-wise Linear Modulation）/ `cross_attention` / `adaptive_norm`。

### 4. OFT 动作头（OFT Action Head）

核心动作解码器，基于 **Optimal Flow Transport（最优流传输）** 思想的 Transformer 架构：

- 使用**可学习的动作查询（Action Queries）**，通过**交叉注意力**关注 VLM 特征。
- 每层结构：`Self-Attention → Cross-Attention(VLM特征) → FiLM Norm(本体状态) → FFN`。
- 默认 6 层 Transformer、16 个注意力头、隐藏维度 1024，可选 spectral norm 提升训练稳定性。
- **双推理模式**：
  - **直接回归（Direct Regression）**：单次前向，L1 监督的 MLP 输出动作。
  - **流匹配（Flow Matching）**：通过条件流匹配的自适应步长 ODE 求解器（Dormand-Prince）迭代细化动作。

### 5. 动作分块（Action Chunking）

- 单次前向预测 **H = 16** 步未来动作（动作维度 7：位置 + 姿态 + 夹爪）。
- 大幅减少部署时的 VLM 推理调用次数。
- 可选**时序集成（Temporal Ensembling）**，在重叠 chunk 间做指数衰减加权，降低动作抖动。

---

## 安装

```bash
# 1. 安装客户端依赖
pip install requests numpy pillow opencv-python imageio tqdm

# 2. 如需运行 LIBERO 评测，额外安装仿真环境
pip install robosuite==1.4.0 libero
```

---

## 启动方式

### 方式一：命令行（推荐）

```bash
# 检查与服务器的连接是否正常
sai0-eval --check --server https://api.sai0.ai --api-key sk-xxx

# 在 LIBERO 基准上运行评测
sai0-eval \
    --server https://api.sai0.ai \
    --api-key sk-xxx \
    --task-suite libero_spatial \
    --trials 10 \
    --max-steps 600 \
    --output-dir ./eval_results
```

也可通过环境变量配置连接信息：

```bash
export SAI0_SERVER=https://api.sai0.ai
export SAI0_API_KEY=sk-xxx
sai0-eval --task-suite libero_object
```

**常用参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--server` | 推理服务器地址 | `http://localhost:5000` |
| `--api-key` | API 密钥 | 读取 `SAI0_API_KEY` |
| `--task-suite` | LIBERO 任务套件（`libero_spatial`/`libero_object`/`libero_goal`/`libero_10`/`libero_90`） | `libero_spatial` |
| `--trials` | 每个任务的试验次数 | `10` |
| `--max-steps` | 每次试验最大步数 | `600` |
| `--action-chunk-exec` | 每次 API 调用执行的动作步数 | `16` |
| `--task-ids` | 指定评测任务 ID（逗号分隔，如 `0,1,2`） | 全部 |
| `--output-dir` | 结果输出目录 | `./eval_results` |
| `--no-video` | 不保存评测视频 | 默认保存 |

### 方式二：作为 SDK 调用

```python
from client import Sai0VLAClient  # 本地开发
# from sai0_vla_client import Sai0VLAClient  # pip 安装后

client = Sai0VLAClient("https://api.sai0.ai", api_key="sk-xxx")

# 发送观测，获取动作序列 (chunk_size, action_dim)
actions = client.act(
    images=[agentview_img, wrist_img],  # RGB 图像列表
    state=state,                        # 机器人状态向量
    instruction="pick up the mug",      # 任务指令
    task_suite="libero_spatial",        # 可选
)
```

### 方式三：在 Python 中直接调用评测函数

```python
from libero_eval import run_libero_eval

run_libero_eval(
    server_url="https://api.sai0.ai",
    api_key="sk-xxx",
    task_suite="libero_spatial",
    num_trials=10,
)
```

---

## 评测流程

整个评测是一个「**观测 → 远程推理 → 执行**」的闭环：

1. **连接校验**：调用 `/version` 确认云端服务可达。
2. **加载任务**：从 LIBERO 取出指定任务套件，逐任务、逐试验运行。
3. **环境初始化**：创建离屏渲染环境，设置初始状态，先执行若干空动作让物体稳定。
4. **Rollout 循环**（每次试验最多 `max_steps` 步）：
   - 从观测中提取 **两路图像**（主视角 agentview + 腕部 wrist）与 **8 维状态向量**。
   - 当本地动作队列为空时，调用 `client.act()` 从云端获取一个动作 chunk（16 步）。
   - 动作填入队列（夹爪维度做符号二值化），逐步执行 `env.step()`。
   - 这种「一次预测多步、本地消费队列」的设计显著减少 API 调用。
5. **成功判定**：当 `done` 且 `reward > 0` 视为任务成功。
6. **结果输出**：将主视角与腕部画面拼接为 mp4 视频；统计每任务及总体成功率，写入 `summary.json`。

输出目录结构：

```
eval_results/
├── summary.json              # 评测汇总（总成功率 + 每任务成功率）
└── videos/
    └── task0/
        ├── trial0_success_xxx.mp4
        └── trial1_fail_xxx.mp4
```

---

## 创新之处

1. **多尺度 VLM 特征 + 自适应路由**
   不同于多数 VLA 只取 VLM 最后一层输出，AdaFlow-VLA 从多个 Transformer 层抽取特征，并用**可学习的门控网络**根据当前观测与指令动态加权融合，兼顾底层空间细节与高层语义。

2. **OFT 动作头（Optimal Flow Transport）**
   基于最优流传输思想的 Transformer 动作头，统一支持**直接回归**与**条件流匹配**两种范式：训练/部署可在「快速单步预测」与「高精度 ODE 迭代细化」之间灵活切换。

3. **FiLM 本体感知调制**
   机器人状态通过 Feature-wise Linear Modulation 注入动作头，使模型能根据当前关节构型自适应调整动作预测，比简单拼接更高效。

4. **动作分块 + 时序集成**
   单次前向预测 16 步动作，结合重叠 chunk 间的指数衰减时序集成，在减少推理调用的同时降低动作抖动，兼顾效率与平滑性。

5. **不确定性估计**
   提供 `predict_with_uncertainty` 接口，通过 MC Dropout 或流匹配多采样估计动作分布方差，为安全部署提供置信度参考。

6. **云端推理 + 极简客户端架构**
   将重量级模型托管在云端，本地客户端零模型依赖、零 GPU 需求，仅需 `requests / numpy / Pillow` 即可接入，大幅降低使用门槛。

---

## 依赖

- **客户端核心**：`requests`、`numpy`、`Pillow`
- **评测工具**：`opencv-python`、`imageio`、`tqdm`
- **仿真环境（评测时）**：`robosuite==1.4.0`、`libero`
- **模型架构定义**：`torch`

---

## 许可与访问

模型权重与完整训练代码托管于 AdaFlow-VLA 推理 API。
`model/` 目录仅提供公开的架构规范，核心实现（特征融合、OFT 动作头、流匹配采样器等）为专有部分，请通过推理 API 访问完整模型。
