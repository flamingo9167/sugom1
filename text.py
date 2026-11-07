"""화면 출력용 텍스트 보정."""
from __future__ import annotations

import re

_ZWSP = "​"
# CommonMark 에서 '구두점'으로 취급되는 문자 (+ 자주 쓰는 전각/특수 기호)
_PUNCT = r"""[!-/:-@\[-`{-~·“”‘’「」『』（）%]"""
_CODE = re.compile(r"(```[\s\S]*?```|`[^`\n]*`)")
_EMPH = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1")


def _fix_segment(text: str) -> str:
    def rep(m: re.Match) -> str:
        mark, inner = m.group(1), m.group(2)
        if re.match(_PUNCT, inner[0]):
            inner = _ZWSP + inner
        if re.search(_PUNCT + "$", inner):
            inner = inner + _ZWSP
        return mark + inner + mark

    return _EMPH.sub(rep, text)


def fix_korean_emphasis(text: str) -> str:
    """`**27%**였습니다` 처럼 구두점으로 끝나는 굵은 글씨 뒤에 한글이 붙으면
    CommonMark 규칙상 굵게 처리되지 않는다. 보이지 않는 폭 0 공백을 넣어 해결한다.
    (코드 블록/인라인 코드는 건드리지 않음)"""
    if "**" not in text and "__" not in text:
        return text
    parts = _CODE.split(text)
    return "".join(p if i % 2 else _fix_segment(p) for i, p in enumerate(parts))
