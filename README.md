# 辅具借用换配

合成借用档案，尺寸为厘米，仅用于借用点的库存和人工测量管理。

项目使用 Python 3.11.2 标准库。资料按来源表以 UTF-8 JSON Lines 保存，每行是一条记录。缺失值写作 `null`，其业务含义见资料约定。列名见 `data/manifest.json`，编号由各资料表原样保留。

[资料约定](docs/materials.md)说明各表的来源、时间及关联语义。[换配规则](docs/exchange-rules.md)说明标签、预测、换配与回测各自的口径。

## 检查资料

```sh
python3 inspect_inputs.py
python3 -m unittest discover -s tests -v
```

## 站点反复更新的分析

工具每次都从资料目录重新读取。站点追加新的复核、退回清点、原因更正或到场确认后，重算即可。

```sh
# 读取计数与资料警告
python3 -m exchange_tool inspect

# 待换配个案的风险预测、所用样本层级与借出时的标签
python3 -m exchange_tool predict --asof 2026-12-22 --site bridge

# 可兑现的换配安排，并对照“只按风险排队”的结果
python3 -m exchange_tool plan --asof 2026-12-22

# 站长视角：每个建议的预测可信度、人员确认、具体配件与保留理由
python3 -m exchange_tool report --asof 2026-12-22 --site bridge

# 模型/资料更新前后，哪些待选建议会改变
python3 -m exchange_tool diff --before 2026-12-20 --after 2026-12-22

# 按时点、站点做前向回测，比较校准、误报代价和可用覆盖
python3 -m exchange_tool backtest --compare base_rate structural --horizon 30
python3 -m exchange_tool backtest --model backoff --site lake --false-negative-cost 3
```

`--asof` 是观察时点：只有该日及以前已经发生的复核、更正、退回与清点才会进入
预测和库存。`--min-support` 控制每一级配置至少需要多少历史样本；达不到时该
个案进入人工测量，不输出风险值。
