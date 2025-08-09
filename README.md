## 功能

用于从有声小说中删除重复的广告，手动标记一处广告之后通过频谱分析找到其他广告在音频中的位置并进行删除。

## 使用方式

1. 参考 [uv 官网](https://docs.astral.sh/uv/getting-started/installation/) 安装 uv
2. 使用 `uv sync` 下载依赖
3. 修改参数后，使用 `uv run main.py` 开始分析音频