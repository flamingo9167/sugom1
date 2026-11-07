"""OpenAI 호출: 스트리밍, 토큰 계산, 긴 문서 Map-Reduce, Q&A 컨텍스트 구성, 오류 변환."""
from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import openai
import tiktoken
from openai import OpenAI

from utils.config import get_api_key, get_config, get_model
from utils.pdf_parser import PdfDocument
from utils.prompts import MAP_PROMPT, QA_SYSTEM, REDUCE_NOTE, build_system_prompt

# 한 번의 요청에 넣을 문서 토큰 한도 (모델 컨텍스트보다 넉넉히 작게)
def _int_config(key: str, default: int) -> int:
    try:
        value = int(str(get_config(key, str(default))).replace(",", "").replace("_", ""))
        return value if value >= 1000 else default
    except (TypeError, ValueError):
        return default


CONTEXT_TOKEN_LIMIT = _int_config("CONTEXT_TOKEN_LIMIT", 100000)
QA_HISTORY_TURNS = 6


class LLMError(Exception):
    """사용자에게 그대로 보여줄 수 있는 메시지를 가진 오류."""


# ---------------- 사용량 ----------------
@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, prompt: int, completion: int) -> None:
        self.input_tokens += prompt or 0
        self.output_tokens += completion or 0

    def merge(self, other: "Usage") -> None:
        self.add(other.input_tokens, other.output_tokens)

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


# ---------------- 클라이언트 / 오류 ----------------
def get_client() -> OpenAI:
    api_key = get_api_key()
    if not api_key:
        raise LLMError("OpenAI API 키가 설정되지 않았습니다. `.env` 또는 Streamlit Secrets를 확인하세요.")
    return OpenAI(api_key=api_key, max_retries=2, timeout=300)


def model_name() -> str:
    return get_model()


def friendly_error(e: Exception) -> str:
    if isinstance(e, LLMError):
        return str(e)
    if isinstance(e, openai.AuthenticationError):
        return "API 키가 올바르지 않습니다."
    if isinstance(e, openai.RateLimitError):
        if "insufficient_quota" in str(e):
            return "OpenAI 사용 한도(크레딧)가 부족합니다. 결제 설정을 확인하세요."
        return "요청이 너무 많습니다. 잠시 후 다시 시도해주세요."
    if isinstance(e, openai.NotFoundError):
        return f"`{model_name()}` 모델에 접근할 수 없습니다. 모델명과 계정 권한을 확인하세요."
    if isinstance(e, openai.PermissionDeniedError):
        return f"`{model_name()}` 모델을 사용할 권한이 없습니다. 계정 권한을 확인하세요."
    if isinstance(e, (openai.APIConnectionError, openai.APITimeoutError)):
        return "OpenAI 서버에 연결할 수 없습니다. 네트워크 상태를 확인하고 다시 시도해주세요."
    if isinstance(e, openai.BadRequestError):
        if "context_length" in str(e) or "maximum context" in str(e):
            return "문서가 너무 길어 처리할 수 없습니다. 파일 수를 줄여 다시 시도해주세요."
        return f"요청이 거부되었습니다: {getattr(e, 'message', e)}"
    if isinstance(e, openai.APIStatusError):
        return f"OpenAI 서버 오류가 발생했습니다 (코드 {e.status_code}). 잠시 후 다시 시도해주세요."
    return f"알 수 없는 오류가 발생했습니다: {e}"


# ---------------- 토큰 ----------------
def _encoding() -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model_name())
    except Exception:
        return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text, disallowed_special=()))


# ---------------- 호출 ----------------
# temperature 를 지원하지 않는 모델(예: 추론 모델)을 만나면 기억해 두고 이후 생략
_TEMPERATURE_UNSUPPORTED: set[str] = set()


def _create(client: OpenAI, messages: list[dict], temperature: float | None, stream: bool):
    model = model_name()
    kwargs: dict = {"model": model, "messages": messages}
    if stream:
        kwargs.update(stream=True, stream_options={"include_usage": True})
    if temperature is not None and model not in _TEMPERATURE_UNSUPPORTED:
        try:
            return client.chat.completions.create(temperature=temperature, **kwargs)
        except openai.BadRequestError as e:
            if "temperature" not in str(e):
                raise
            _TEMPERATURE_UNSUPPORTED.add(model)
    return client.chat.completions.create(**kwargs)


def temperature_supported() -> bool:
    return model_name() not in _TEMPERATURE_UNSUPPORTED


def stream_chat(messages: list[dict], usage: Usage, temperature: float | None = None) -> Iterator[str]:
    """응답 텍스트 조각을 yield 한다. 사용량은 usage 에 누적. 오류는 LLMError 로 변환."""
    try:
        client = get_client()
        stream = _create(client, messages, temperature, stream=True)
        for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
            if getattr(chunk, "usage", None):
                usage.add(chunk.usage.prompt_tokens, chunk.usage.completion_tokens)
    except LLMError:
        raise
    except Exception as e:  # noqa: BLE001
        raise LLMError(friendly_error(e)) from e


def complete(messages: list[dict], usage: Usage, temperature: float | None = None) -> str:
    try:
        client = get_client()
        resp = _create(client, messages, temperature, stream=False)
        if resp.usage:
            usage.add(resp.usage.prompt_tokens, resp.usage.completion_tokens)
        return resp.choices[0].message.content or ""
    except LLMError:
        raise
    except Exception as e:  # noqa: BLE001
        raise LLMError(friendly_error(e)) from e


# ---------------- 문서 → 텍스트 / 청크 ----------------
def _page_blocks(docs: list[PdfDocument]) -> list[tuple[str, int, str]]:
    """(파일명, 페이지번호, '[파일명 - p.N]\\n본문') 목록."""
    blocks = []
    for d in docs:
        for i, text in enumerate(d.pages, start=1):
            if text.strip():
                blocks.append((d.name, i, f"[{d.name} - p.{i}]\n{text}"))
    return blocks


def docs_to_text(docs: list[PdfDocument]) -> str:
    return "\n\n".join(d.full_text() for d in docs if d.has_text)


def split_into_chunks(docs: list[PdfDocument], limit: int) -> list[str]:
    """페이지 경계 기준으로 limit 토큰 이하 청크로 나눈다. 한 페이지가 limit 보다 크면 잘라서 넣는다."""
    enc = _encoding()
    chunks: list[str] = []
    cur: list[str] = []
    cur_tokens = 0
    for _, _, block in _page_blocks(docs):
        tokens = enc.encode(block, disallowed_special=())
        if len(tokens) > limit:  # 아주 긴 단일 페이지
            if cur:
                chunks.append("\n\n".join(cur))
                cur, cur_tokens = [], 0
            for start in range(0, len(tokens), limit):
                chunks.append(enc.decode(tokens[start : start + limit]))
            continue
        if cur_tokens + len(tokens) > limit and cur:
            chunks.append("\n\n".join(cur))
            cur, cur_tokens = [], 0
        cur.append(block)
        cur_tokens += len(tokens)
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


# ---------------- 분석 ----------------
def prepare_analysis(
    docs: list[PdfDocument],
    analysis_type: str,
    language: str,
    custom_instruction: str,
    usage: Usage,
    temperature: float | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """최종(스트리밍) 호출에 쓸 messages 를 만든다. 문서가 길면 청크별 요약(map)을 먼저 수행."""
    system = build_system_prompt(analysis_type, language, custom_instruction)
    full_text = docs_to_text(docs)
    if count_tokens(full_text) <= CONTEXT_TOKEN_LIMIT:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Documents:\n\n{full_text}"},
        ]

    # Map: 청크별 상세 요약
    chunks = split_into_chunks(docs, CONTEXT_TOKEN_LIMIT)
    partials = _summarize_parts(chunks, language, usage, temperature, on_progress)

    # 부분 요약을 합쳐도 한도를 넘으면(아주 긴 문서) 요약을 다시 묶어 요약한다.
    joined = _join_partials(partials)
    for _ in range(5):  # 안전장치: 최대 5단계
        if count_tokens(joined) <= CONTEXT_TOKEN_LIMIT or len(partials) <= 1:
            break
        groups = _group_by_tokens(partials, CONTEXT_TOKEN_LIMIT)
        partials = _summarize_parts(groups, language, usage, temperature, on_progress)
        joined = _join_partials(partials)
    return [
        {"role": "system", "content": system + "\n\n" + REDUCE_NOTE},
        {"role": "user", "content": f"Partial summaries:\n\n{joined}"},
    ]


def _summarize_parts(
    parts: list[str],
    language: str,
    usage: Usage,
    temperature: float | None,
    on_progress: Callable[[int, int], None] | None,
) -> list[str]:
    results = []
    for idx, part in enumerate(parts, start=1):
        if on_progress:
            on_progress(idx, len(parts))
        results.append(
            complete(
                [
                    {"role": "system", "content": MAP_PROMPT.format(index=idx, total=len(parts), language=language)},
                    {"role": "user", "content": part},
                ],
                usage,
                temperature,
            )
        )
    return results


def _join_partials(partials: list[str]) -> str:
    return "\n\n---\n\n".join(f"(Part {i})\n{p}" for i, p in enumerate(partials, start=1))


def _group_by_tokens(texts: list[str], limit: int) -> list[str]:
    groups: list[str] = []
    cur: list[str] = []
    cur_tokens = 0
    for t in texts:
        n = count_tokens(t)
        if cur and cur_tokens + n > limit:
            groups.append("\n\n---\n\n".join(cur))
            cur, cur_tokens = [], 0
        cur.append(t)
        cur_tokens += n
    if cur:
        groups.append("\n\n---\n\n".join(cur))
    return groups


def needs_chunking(docs: list[PdfDocument]) -> bool:
    return count_tokens(docs_to_text(docs)) > CONTEXT_TOKEN_LIMIT


# ---------------- Q&A ----------------
_WORD = re.compile(r"[0-9A-Za-z가-힣]{2,}")


def _terms(text: str) -> set[str]:
    words = {w.lower() for w in _WORD.findall(text)}
    # 한국어 조사 대응: 2글자 접두어도 함께 사용 ("해지조건은" → "해지")
    words |= {w[:2] for w in words if re.match(r"[가-힣]", w) and len(w) > 2}
    return words


def build_qa_context(docs: list[PdfDocument], question: str, limit: int = CONTEXT_TOKEN_LIMIT) -> str:
    """문서가 한도 이내면 전체, 넘으면 질문과 관련성 높은 페이지만 골라 한도 내로 구성."""
    full_text = docs_to_text(docs)
    if count_tokens(full_text) <= limit:
        return full_text

    q_terms = _terms(question)
    blocks = _page_blocks(docs)
    scored = []
    for order, (name, page, block) in enumerate(blocks):
        b_lower = block.lower()
        score = sum(b_lower.count(t) for t in q_terms)
        scored.append((score, order, block))
    scored.sort(key=lambda x: (-x[0], x[1]))

    enc = _encoding()
    picked: list[tuple[int, str]] = []
    used = 0
    for score, order, block in scored:
        n = len(enc.encode(block, disallowed_special=()))
        if used + n > limit:
            continue
        picked.append((order, block))
        used += n
    picked.sort()  # 원래 순서대로
    return "\n\n".join(b for _, b in picked)


def build_qa_messages(
    docs: list[PdfDocument], history: list[dict], question: str, language: str
) -> list[dict]:
    context = build_qa_context(docs, question)
    system = QA_SYSTEM.format(language=language) + f"\n\nDocuments:\n\n{context}"
    recent = [{"role": m["role"], "content": m["content"]} for m in history[-QA_HISTORY_TURNS * 2 :]]
    return [{"role": "system", "content": system}, *recent, {"role": "user", "content": question}]
