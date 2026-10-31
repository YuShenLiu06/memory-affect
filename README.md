# Memory Affect

一个由宿主请求驱动的Python记忆与三维情感库，提供SQLite持久化、可解释召回、有效命中强化和版本化状态。无后台唤醒；宿主负责模型与Agent循环。DSH插件单独在[dsh-memory-affect](https://github.com/YuShenLiu06/dsh-memory-affect)。

## 安装

Python >=3.10。下载[0.1.0 Release](https://github.com/YuShenLiu06/memory-affect/releases/tag/v0.1.0)中的wheel和SHA256SUMS，先核对文件摘要：

```sh
shasum -a 256 memory_affect-0.1.0-py3-none-any.whl
# 应为24fc7594d1b7a509d1476cd6f65edd0bf0d621562015094996385459179e9fd2
python3 -m venv .venv
.venv/bin/pip install ./memory_affect-0.1.0-py3-none-any.whl
```

唯一必需第三方依赖为Pydantic >=2.8,<3。不是PyPI发布，不应从未知同名包替代。

## 最小使用

```python
from memory_affect import MemoryLibrary
from memory_affect.models import MemoryInput, SearchQuery

lib = MemoryLibrary('private-memory.sqlite3', initial_time=1000)
lib.manage_memory('add-demo', memory=MemoryInput(
    event_id='demo-tea', content='合成示例：偏好绿茶', occurred_at=1000,
), evidence='local synthetic example')
request = lib.begin_request('user-1', 1001)
candidates = lib.search(request, SearchQuery(text='绿茶'))
# search只读；宿主确认实际选入上下文后才commit selected_ids。
lib.close()
```

真实宿主时间必须可信且同一request固定。仅真实用户发送时begin_request；到期但无用户请求不执行。请使用私有存储路径，勿提交实际记忆、日志或凭据。

## 语义与边界

记忆、当前事件评价与全局情绪分开存储。情绪happy/angry/sad可为负或超过100；100是标尺而非上限。候选、预览、周期展示不算有效命中；宿主确认最终选入后按canonical事件去重记账，累计k按1/k影响情绪。重评默认future_only，删除不倒改已经发生的情感历史。

基线快照、最新持久版本和待注入内容分离；冲突明确返回，不能用假新用户ID绕过。核心只给提示/schema/结构化结果，不执行主Agent循环。冻结发行物保留早期兼容桥接与HTTP模块；新DSH集成只在独立插件演进，不宣称冻结包已移除所有历史兼容接口。

## 源码与测试

本公开项目是**冻结发行物的可重建源码整理**：9个运行模块逐字节来自既有0.1.0 wheel。找到的原preview2工程早于该冻结产物，只有1个同名运行模块完全一致；因此没有把旧工程冒称最终原始仓库。构建配置和现有105项回归测试取自该早期工程，具体来源和验证结果见PROVENANCE.md。没有修改核心运行逻辑。

```sh
python3 -m venv .venv
.venv/bin/pip install '.[dev]'
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests
.venv/bin/mypy src
```

测试为合成数据，不请求真实模型。真实模型效果、Windows和完整宿主端到端不由核心单元测试证明。MIT许可证见LICENSE；第三方说明见THIRD_PARTY_NOTICES.md。
