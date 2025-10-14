# Wendling Model 使用指南

## 📌 基本用法

```python
from neurolib.models.wendling import WendlingModel
import numpy as np

# 创建连接矩阵
N = 6
Cmat = np.eye(N)  # 结构连接矩阵
Dmat = np.zeros((N, N))  # 距离矩阵

# 创建模型
model = WendlingModel(Cmat=Cmat, Dmat=Dmat)
model.params['duration'] = 10000  # 10 秒
model.params['dt'] = 0.1  # 时间步长
model.params['K_gl'] = 0.15  # 全局耦合强度

# 运行仿真
model.run()

# 提取信号
signals = model.y1 - model.y2 - model.y3  # PSP (金字塔神经元输出)
```

---

## 🎛️ 关键参数说明

### 1. heterogeneity（节点异质性）

**作用**：控制节点间参数的随机变异程度

**取值范围**：0.0 ~ 1.0
- `0.0` = 无变异，所有节点参数相同
- `0.1` = 10% 变异
- `0.3` = 30% 变异（推荐用于全脑网络）
- `0.5` = 50% 变异

**重要特性**：
- ✅ 当 `heterogeneity > 0` 时，B, G, A, p_mean 变成**向量**（每个节点不同）
- ❌ 当 `heterogeneity = 0` 时，这些参数是**标量**（所有节点相同）

**示例**：
```python
# 无异质性（标量模式）
model = WendlingModel(Cmat, Dmat, heterogeneity=0.0)
print(model.params['B'])  # → 22.0 (单个数字)

# 有异质性（向量模式）
model = WendlingModel(Cmat, Dmat, heterogeneity=0.3)
print(model.params['B'])  # → [25.3, 18.7, 22.1, ...] (数组)
```

---

### 2. random_init（初始条件类型）

**作用**：控制状态变量的初始值

**取值**：`True` 或 `False`
- `False` = 零初始条件（所有状态从 0 开始）
- `True` = 随机初始条件（从 random(-0.1, 0.1) 开始）

**使用建议**：

| 场景 | 推荐值 | 原因 |
|------|--------|------|
| Single-node 测试 | `False` | 复现经典 Wendling 2002 波形 |
| Multi-node 网络 | `True` | 避免某些参数组合衰减成稳态 |
| 全脑仿真 | `True` | 更接近真实大脑状态 |

**示例**：
```python
# Single-node
model = WendlingModel(Cmat, Dmat, random_init=False)

# Multi-node
model = WendlingModel(Cmat, Dmat, random_init=True)
```

---

### 3. seed（随机种子）

**作用**：保证结果可重复

```python
model = WendlingModel(Cmat, Dmat, heterogeneity=0.3, seed=42)
# 每次运行产生相同的随机参数和初始条件
```

---

## 🎯 常见使用场景

### 场景 1：Single-node 经典波形复现

```python
# 复现 Wendling 2002 的 6 种活动类型
Cmat = np.array([[0]])
Dmat = np.array([[0]])

model = WendlingModel(
    Cmat=Cmat, 
    Dmat=Dmat,
    heterogeneity=0.0,   # 标量模式
    random_init=False,   # 零初始条件
    seed=42
)

# 设置 Type3 (SWD) 参数
model.params['B'] = 25
model.params['G'] = 15
model.params['A'] = 5
model.params['p_mean'] = 90
model.params['p_sigma'] = 2.0
model.params['duration'] = 10000
model.params['dt'] = 0.1
model.params['K_gl'] = 0.0

model.run()
signal = model.y1[0, :] - model.y2[0, :] - model.y3[0, :]
```

---

### 场景 2：Multi-node 手动指定每个节点的类型

```python
# 想要为每个节点设置不同的 Wendling type
N = 6
NODE_TYPES = ['Type1', 'Type3', 'Type6', 'Type6', 'Type1', 'Type1']

Cmat = np.eye(N)
Dmat = np.zeros((N, N))

# Hack: 使用很小的 heterogeneity 触发向量模式
model = WendlingModel(
    Cmat=Cmat, 
    Dmat=Dmat,
    heterogeneity=0.01,  # 触发向量模式
    random_init=True,    # multi-node 必须用 True
    seed=42
)

# 手动设置每个节点的参数
model.params['B'] = np.array([50, 25, 15, 15, 50, 50])  # Type1, Type3, Type6...
model.params['G'] = np.array([15, 15, 0, 0, 15, 15])
model.params['A'] = np.array([5, 5, 5, 5, 5, 5])
model.params['p_mean'] = np.array([90, 90, 90, 90, 90, 90])
model.params['p_sigma'] = 2.0  # 标量（所有节点共用）

model.params['duration'] = 10000
model.params['dt'] = 0.1
model.params['K_gl'] = 0.0  # 或 0.15 用于耦合

model.run()
```

---

### 场景 3：全脑网络建模（自动随机参数）

```python
# 80-node 全脑网络
N = 80
Cmat = load_structural_connectivity()  # 真实 SC 矩阵
Dmat = load_distance_matrix()

model = WendlingModel(
    Cmat=Cmat, 
    Dmat=Dmat,
    heterogeneity=0.30,  # 30% 参数变异
    random_init=True,    # 随机初始条件
    seed=42
)

# 不需要手动设置参数，已自动生成
model.params['duration'] = 10000
model.params['dt'] = 0.1
model.params['K_gl'] = 0.15  # 全局耦合

model.run()

# 提取所有节点信号
signals = model.y1 - model.y2 - model.y3  # shape: (80, 100000)
```

---

## ⚠️ 重要注意事项

### 1. heterogeneity 的双重作用

```python
# heterogeneity 不仅控制变异程度，还决定参数是否向量化！

# heterogeneity = 0
# → B, G, A, p_mean 是标量
# → 无法为每个节点设置不同值

# heterogeneity > 0
# → B, G, A, p_mean 是向量
# → 可以手动覆盖为任意值
```

**Hack 技巧**：如果想手动设置每个节点的参数，但不想要随机变异：
```python
model = WendlingModel(heterogeneity=0.01, seed=42)  # 很小的变异触发向量模式
model.params['B'] = np.array([50, 25, 15, ...])    # 手动覆盖为精确值
```

---

### 2. random_init 对 multi-node 至关重要

```python
# ❌ 错误：multi-node + random_init=False
model = WendlingModel(Cmat, Dmat, random_init=False)
model.params['B'] = np.array([50, 50, 50])  # high-B type
model.run()
# → 信号会衰减成水平线！

# ✅ 正确：multi-node + random_init=True
model = WendlingModel(Cmat, Dmat, random_init=True)
model.params['B'] = np.array([50, 50, 50])
model.run()
# → 正常振荡
```

**原因**：零初始条件 + multi-node + high-B 参数 → 系统陷入稳态吸引子

---

### 3. p_sigma 未向量化

```python
# 限制：p_sigma 始终是标量
model.params['p_sigma'] = 2.0  # 所有节点共用

# ❌ 无法这样做：
model.params['p_sigma'] = np.array([2.0, 30.0, 2.0, ...])  # 不支持
```

**影响**：不能在同一网络中混用需要不同 p_sigma 的 Wendling types
- Type3, Type6 需要 `p_sigma = 2.0`（低噪声）
- Type1, Type2, Type4, Type5 需要 `p_sigma = 30.0`（高噪声）

**变通方案**：只混用需要相同 p_sigma 的 types

---

## 📊 参数向量化状态

| 参数 | 是否向量化 | 条件 | 手动设置 |
|------|------------|------|----------|
| B | ✅ | heterogeneity > 0 | ✅ 可以 |
| G | ✅ | heterogeneity > 0 | ✅ 可以 |
| A | ✅ | heterogeneity > 0 | ✅ 可以 |
| p_mean | ✅ | heterogeneity > 0 | ✅ 可以 |
| p_sigma | ❌ | 始终标量 | ❌ 不能 |
| K_gl | ❌ | 始终标量 | ✅ 可以 |
| duration | ❌ | 始终标量 | ✅ 可以 |
| dt | ❌ | 始终标量 | ✅ 可以 |

---

## 🔧 Wendling 2002 六种活动类型

### 标准参数

| Type | B | G | A | p_mean | p_sigma | 频率 | 描述 |
|------|---|---|---|--------|---------|------|------|
| Type1 | 50 | 15 | 5 | 90 | 30.0* | 1-7 Hz | Background activity |
| Type2 | 40 | 15 | 5 | 90 | 30.0* | 1-5 Hz | Sporadic spikes |
| Type3 | 25 | 15 | 5 | 90 | 2.0 | 3-6 Hz | SWD (epileptic) |
| Type4 | 10 | 15 | 5 | 90 | 30.0* | 8-13 Hz | Alpha rhythm |
| Type5 | 5 | 25 | 5 | 90 | 30.0* | 10-20 Hz | LVFA |
| Type6 | 15 | 0 | 5 | 90 | 2.0 | 9-13 Hz | Quasi-sinusoidal |

*注：原始论文使用 p_sigma=30.0，但某些实现中使用 2.0

---

## 📚 相关文档

- `STANDARD_PARAMETERS.py` - 标准参数定义
- `HETEROGENEITY_AND_RANDOM_INIT.md` - 详细参数说明
- Wendling et al. (2002) - 原始论文

---

## 🐛 常见问题

### Q1: 为什么我的信号衰减成水平线？
**A**: Multi-node 网络必须使用 `random_init=True`

### Q2: 为什么我无法为每个节点设置不同的 B 值？
**A**: 需要设置 `heterogeneity > 0` 来触发向量模式

### Q3: 为什么 Type1 的振幅很小？
**A**: Type1 需要 `p_sigma=30.0`，但如果其他类型需要 `p_sigma=2.0`，由于 p_sigma 未向量化，只能选择一个值

### Q4: 如何保证结果可重复？
**A**: 设置 `seed` 参数：`WendlingModel(..., seed=42)`

---

**最后更新**: 2025-10-14  
**版本**: neurolib wendling model (custom implementation)
