# 学生错误倾向第一阶段实验

本项目验证：只使用学生训练时间段中的错误作答，LLM生成的结构化错误倾向画像，能否提高对同一学生后续单选题错误选项的预测。

第一阶段不微调模型，也不使用人工错误机制标签。评价信号是学生在后续时间段真实选择的错误选项。

## 目录

```text
Mistake-tendency/
├─ configs/default.yaml              # 数据、切分、模型和实验配置
├─ src/mistake_tendency/             # 预处理、模型接口、画像与预测代码
│  └─ clients/                       # DeepSeek/GPT/Qwen/Claude/mock适配层
├─ data/processed/                   # 处理后的事件、画像输入和测试任务
├─ result/                           # 模型画像、预测和评估结果
├─ evaluate/evaluate_stage1.py       # 学生级指标与bootstrap评估
├─ evaluate/build_constraint_profiles.py # 导出可用于后续约束的画像
├─ requirements.txt
└─ pyproject.toml
```

## 设计原则

- 按学生时间排序后切分为60%训练、20%验证、20%测试。
- 画像只读取训练时间段中的错误作答。
- 测试任务条件化在“学生实际答错”，模型只在错误选项间分配概率。
- 主指标是先按学生平均、再在学生间平均的NLL。
- 同时计算Top-1、Top-2、MRR、Brier以及学生级bootstrap置信区间。
- 自动生成总体、题目和知识点统计基线。
- API密钥只从环境变量读取，不写入代码或结果。
- 每条模型结果立即追加到JSONL；重新运行会跳过已成功记录，可断点续跑。

## 安装

PowerShell：

```powershell
cd 'Mistake-tendency'
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

如果已在现有Python环境安装依赖，可跳过虚拟环境步骤。

## 1. 预处理

将FoundationalASSIST数据集放到`data/raw/FOUNDATIONALASSIST/`，或在
`configs/default.yaml`中把`paths.raw_data_dir`修改为本机数据集目录。

```powershell
python -m mistake_tendency.cli preprocess --config configs/default.yaml
```

输出：

- `data/processed/events.parquet`
- `data/processed/eligible_students.csv`
- `data/processed/profile_inputs.jsonl`
- `data/processed/validation_tasks.jsonl`
- `data/processed/test_tasks.jsonl`
- `data/processed/manifest.json`

选择题答案会依次尝试规范化精确匹配、数值等价匹配和高阈值唯一模糊匹配。无法唯一映射到选项的作答不会进入单选题预测任务，但仍会在manifest中报告。

## 2. 离线冒烟测试

`mock`只用于验证代码流程，不是实验模型。

```powershell
python -m mistake_tendency.cli profiles --config configs/default.yaml --provider mock --limit 5 --overwrite
python -m mistake_tendency.cli predict --config configs/default.yaml --provider mock --condition profile --split validation --limit-students 5 --overwrite
python evaluate/evaluate_stage1.py --config configs/default.yaml --split validation --prediction mock_profile=result/predictions__mock__deterministic-mock__profile__validation.jsonl
```

## 3. DeepSeek验证集实验

当前配置默认使用DeepSeek。先在当前PowerShell会话设置密钥：

```powershell
$env:DEEPSEEK_API_KEY='你的密钥'
```

先运行20名学生的费用和格式检查：

```powershell
python -m mistake_tendency.cli profiles --config configs/default.yaml --provider deepseek --limit 20
python -m mistake_tendency.cli predict --config configs/default.yaml --provider deepseek --condition no_profile --split validation --limit-students 20
python -m mistake_tendency.cli predict --config configs/default.yaml --provider deepseek --condition profile --split validation --limit-students 20
python -m mistake_tendency.cli predict --config configs/default.yaml --provider deepseek --condition raw_history --split validation --limit-students 20
python -m mistake_tendency.cli predict --config configs/default.yaml --provider deepseek --condition shuffled_profile --split validation --limit-students 20
```

确认输出正确后，去掉`--limit`和`--limit-students`运行全部学生。已有成功记录会自动跳过；需要从头运行时显式添加`--overwrite`。

验证集评估：

```powershell
python evaluate/evaluate_stage1.py `
  --config configs/default.yaml `
  --split validation `
  --prediction no_profile=result/predictions__deepseek__deepseek-flash__no_profile__validation.jsonl `
  --prediction profile=result/predictions__deepseek__deepseek-flash__profile__validation.jsonl `
  --prediction raw_history=result/predictions__deepseek__deepseek-flash__raw_history__validation.jsonl `
  --prediction shuffled_profile=result/predictions__deepseek__deepseek-flash__shuffled_profile__validation.jsonl
```

锁定提示词、模型和配置后，把`--split validation`替换为`--split test`运行最终测试。测试集不要用于修改提示词或阈值。

将画像和验证集预测合并为后续模拟器可直接读取的约束文件：

```powershell
python evaluate/build_constraint_profiles.py `
  --config configs/default.yaml `
  --profiles result/profiles__deepseek__deepseek-flash.jsonl `
  --profile-predictions result/predictions__deepseek__deepseek-flash__profile__validation.jsonl `
  --no-profile-predictions result/predictions__deepseek__deepseek-flash__no_profile__validation.jsonl `
  --output result/constraint_profiles__deepseek__deepseek-flash.jsonl
```

该文件为每名学生增加`validation`字段，并为每项倾向增加：

- `usable_for_constraint`
- `effective_weight = strength × confidence`

默认只有至少3道成对验证题且画像NLL优于无画像条件的学生，才会被标记为可用于约束。阈值可在配置中修改。

## 4. 切换模型供应商

### GPT

```powershell
$env:OPENAI_API_KEY='你的密钥'
python -m mistake_tendency.cli profiles --config configs/default.yaml --provider gpt --limit 20
```

GPT适配器使用OpenAI Responses API。模型名在`configs/default.yaml`中修改为账户可用的精确模型名。

### Qwen

```powershell
$env:DASHSCOPE_API_KEY='你的密钥'
$env:QWEN_BASE_URL='你的百炼OpenAI兼容地址'
python -m mistake_tendency.cli profiles --config configs/default.yaml --provider qwen --limit 20
```

不同地域或业务空间的Qwen兼容地址可能不同，因此可通过`QWEN_BASE_URL`覆盖配置。

### Claude

```powershell
$env:ANTHROPIC_API_KEY='你的密钥'
python -m mistake_tendency.cli profiles --config configs/default.yaml --provider claude --limit 20
```

Claude适配器使用Anthropic Messages API。新模型可能不接受`temperature`参数，因此适配器没有强制传入该参数。

## 5. 输出解释

画像文件：

```text
result/profiles__<provider>__<model>.jsonl
```

每个学生包含：

- `profile.tendencies`
- 每项倾向的作用知识点和题型范围
- 强度和置信度
- 支持证据事件ID
- 典型错误答案
- 全局错误选项位置偏好

预测文件：

```text
result/predictions__<provider>__<model>__<condition>__<split>.jsonl
```

评估文件：

```text
result/evaluation/metrics_summary_<split>.csv
result/evaluation/student_metrics_<split>.csv
result/evaluation/pairwise_comparisons_<split>.csv
```

`pairwise_comparisons`中：

- NLL差值小于0表示候选条件更好；
- Top-1差值大于0表示候选条件更好；
- `better_than_reference=true`表示学生级bootstrap的95%置信区间支持改善。

## API依据

- DeepSeek JSON输出与OpenAI兼容调用：https://api-docs.deepseek.com/guides/json_mode/
- OpenAI Python Responses API：https://developers.openai.com/api/reference/python
- Qwen OpenAI兼容调用：https://help.aliyun.com/en/model-studio/first-api-call-to-qwen
- Claude Messages API：https://platform.claude.com/docs/zh-CN/home

