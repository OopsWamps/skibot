"""临时批量驱动：读路径清单，逐条调 skibot.cli.main。绕开 shell 参数替换的坑。

用法：.venv/Scripts/python.exe scripts/_run_batch.py <path_list.txt> <result_dir>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from skibot.cli import main  # noqa: E402

if __name__ == "__main__":
    list_file, result_dir = sys.argv[1], sys.argv[2]
    paths = [p for p in Path(list_file).read_text(encoding="utf-8").split() if p]
    rc = main(["analyze", *paths, "--no-blur", "--result-dir", result_dir])
    sys.exit(rc)
