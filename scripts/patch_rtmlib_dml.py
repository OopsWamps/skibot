"""给 venv 里的 rtmlib 打 DML 补丁：把 onnxruntime 后端的 'cuda' 设备映射到
DmlExecutionProvider（onnxruntime-directml 包不提供 CUDA EP，NVIDIA 卡走 DML）。

幂等：已打补丁则跳过。rtmlib 升级后补丁会丢，重跑本脚本即可。
配合 pyproject 的 onnxruntime-directml 依赖（Windows），见 docs/DECISIONS.md 第 6 条。

用法：.venv/Scripts/python.exe scripts/patch_rtmlib_dml.py
"""

import sys
from pathlib import Path

ORIGIN = """        'cuda':
        'CUDAExecutionProvider',"""
PATCHED = """        'cuda':
        'DmlExecutionProvider',  # 本地补丁：onnxruntime-directml 无 CUDA EP，NVIDIA 卡走 DML（scripts/patch_rtmlib_dml.py）"""


def main() -> int:
    try:
        import rtmlib  # noqa: F401
    except ImportError:
        print("rtmlib 未安装（应随 sports2d 安装），无从打补丁")
        return 1

    base = Path(rtmlib.__file__).parent / "tools" / "base.py"
    text = base.read_text(encoding="utf-8")
    if "DmlExecutionProvider" in text:
        print(f"已是补丁状态，跳过：{base}")
        return 0
    if ORIGIN not in text:
        print(f"未找到预期代码块（rtmlib 版本变了？），请手工检查：{base}")
        return 1
    base.write_text(text.replace(ORIGIN, PATCHED), encoding="utf-8")
    print(f"补丁完成：{base}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
