"""PDF 텍스트 추출 (페이지 번호 보존)."""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

from pypdf import PdfReader
from pypdf.errors import DependencyError, PdfReadError

# 일부 PDF는 공백 대신 NUL 등 제어문자를 사용한다 → 공백으로 치환
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")


def clean_text(text: str) -> str:
    text = _CONTROL_CHARS.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


@dataclass
class PdfDocument:
    name: str
    pages: list[str] = field(default_factory=list)  # 인덱스 0 = 1페이지
    error: str | None = None

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def char_count(self) -> int:
        return sum(len(p) for p in self.pages)

    @property
    def has_text(self) -> bool:
        return self.char_count > 0

    def full_text(self) -> str:
        """페이지 표시가 포함된 전체 텍스트."""
        return "\n\n".join(
            f"[{self.name} - p.{i}]\n{text}"
            for i, text in enumerate(self.pages, start=1)
            if text.strip()
        )


def extract_pdf(name: str, data: bytes) -> PdfDocument:
    """PDF 바이트에서 페이지별 텍스트를 추출한다. 실패해도 예외 대신 error 를 채워 반환."""
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            # 빈 암호로 열리는 PDF(권한 제한만 걸린 경우)는 허용
            try:
                if not reader.decrypt(""):
                    return PdfDocument(name=name, error="암호로 보호된 PDF는 분석할 수 없습니다.")
            except DependencyError:
                raise
            except Exception:
                return PdfDocument(name=name, error="암호로 보호된 PDF는 분석할 수 없습니다.")
        pages = []
        for page in reader.pages:
            try:
                pages.append(clean_text(page.extract_text() or ""))
            except Exception:
                pages.append("")
        doc = PdfDocument(name=name, pages=pages)
        if not doc.has_text:
            doc.error = "텍스트를 추출할 수 없습니다. 스캔 이미지 PDF일 수 있습니다."
        return doc
    except DependencyError:
        # AES 암호화 PDF 는 cryptography 패키지가 필요 (requirements.txt 에 포함)
        return PdfDocument(
            name=name,
            error="암호화된 PDF를 여는 데 필요한 구성요소(cryptography)가 설치되지 않았습니다.",
        )
    except PdfReadError:
        return PdfDocument(name=name, error="손상되었거나 올바르지 않은 PDF 파일입니다.")
    except Exception as e:  # noqa: BLE001
        return PdfDocument(name=name, error=f"PDF를 읽는 중 오류가 발생했습니다: {e}")
