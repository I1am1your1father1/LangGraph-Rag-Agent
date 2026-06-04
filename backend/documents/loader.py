from pathlib import Path


def load_text_file(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix not in {".txt", ".md"}:
        raise ValueError("当前版本只支持 .txt 和 .md 文件")

    text = path.read_text(encoding="utf-8", errors="ignore")
    return clean_text(text)


def clean_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)