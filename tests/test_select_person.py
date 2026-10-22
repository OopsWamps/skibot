"""选人打分测试：python tests/test_select_person.py
纯离线：select_person 只收轨迹与已采样像素，不碰视频（打分与采样已分离）。
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows GBK 控制台兜底

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skibot.select_person import (  # noqa: E402
    color_score, resolve_color, select_person,
)
from skibot.skeleton import (  # noqa: E402
    L_HIP, L_KNEE, L_SHOULDER, R_HIP, R_KNEE, R_SHOULDER, size_score,
)


def track(x0: float, spread: float, drift: float, n: int = 20,
          y0: float = 100, y1: float = 300) -> np.ndarray:
    """(n,26,3) 合成轨迹：躯干六点齐全，横向位置随帧漂移 drift。"""
    kp = np.zeros((n, 26, 3))
    kp[:, :, 2] = 0.0
    for f in range(n):
        dx = x0 + f * drift
        for idx, (px, py) in ((L_SHOULDER, (dx, y0)), (R_SHOULDER, (dx + spread, y0)),
                              (L_HIP, (dx, (y0 + y1) / 2)), (R_HIP, (dx + spread, (y0 + y1) / 2)),
                              (L_KNEE, (dx, y1)), (R_KNEE, (dx + spread, y1))):
            kp[f, idx] = [px, py, 1.0]
    return kp


def test_size_score_prefers_bigger():
    big = track(100, 100.0, 0.0, y0=50, y1=450)  # 包围盒主轴 400px
    small = track(100, 20.0, 0.0)                # 主轴 200px
    assert size_score(big) > size_score(small)
    print("  尺寸分：大骨架胜出 ✓")


def test_auto_mode_picks_main_subject():
    main_ = track(100, 100.0, 4.0)   # 大且位移多
    extra = track(400, 20.0, 0.5)    # 小且基本不动
    sel = select_person({0: main_, 1: extra})
    assert sel.pid == 0, sel
    assert not sel.color_fallback
    assert "自动判定" in sel.reason
    # 得分行有名字：原始值不经过展示舍入（归一化用原始值）
    row0 = next(r for r in sel.rows if r.pid == 0)
    assert row0.size_n == 1.0 and row0.disp_n == 1.0
    print("  自动模式：主角胜出，得分行字段有契约 ✓")


def test_color_mode_locks_on():
    main_ = track(100, 100.0, 4.0)
    pink = track(400, 20.0, 0.5)     # 尺寸/位移都输，但穿目标色
    pixels = {0: np.full((30, 3), [230.0, 235.0, 245.0]),   # 雪地白
              1: np.full((30, 3), [255.0, 105.0, 180.0])}   # hot pink
    sel = select_person({0: main_, 1: pink}, colors_by_pid=pixels, color_name="pink")
    assert sel.pid == 1, sel
    assert not sel.color_fallback
    assert "颜色" in sel.reason
    print("  颜色模式：锁定目标色，压过尺寸/位移 ✓")


def test_color_fallback_to_auto():
    kp0 = track(100, 100.0, 4.0)
    kp1 = track(400, 20.0, 0.5)
    snow = np.full((30, 3), [230.0, 235.0, 245.0])  # 全是雪地，无目标色
    sel = select_person({0: kp0, 1: kp1},
                        colors_by_pid={0: snow, 1: snow}, color_name="pink")
    assert sel.color_fallback
    assert sel.pid == 0  # 回退后自动判定接管
    assert sel.color_best == 0.0
    print("  颜色不可信：回退自动判定并标记 ✓")


def test_color_score_filters_snow():
    spec = resolve_color("pink")
    assert color_score(np.zeros((0, 3)), spec) == 0.0
    assert color_score(np.full((10, 3), [230.0, 235.0, 245.0]), spec) == 0.0  # 雪地被滤掉
    print("  颜色分：零样本与雪地均得 0 ✓")


if __name__ == "__main__":
    test_size_score_prefers_bigger()
    test_auto_mode_picks_main_subject()
    test_color_mode_locks_on()
    test_color_fallback_to_auto()
    test_color_score_filters_snow()
    print("全部通过")
