import re


def validate_citations(answer: str, citations: list[dict]) -> dict:
    valid_indexes = {str(c["index"]) for c in citations}
    used_indexes = set(re.findall(r"\[(\d+)\]", answer))

    invalid_indexes = used_indexes - valid_indexes

    return {
        "used_indexes": sorted(used_indexes),
        "invalid_indexes": sorted(invalid_indexes),
        "is_valid": len(invalid_indexes) == 0,
    }