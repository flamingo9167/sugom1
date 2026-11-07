"""PDF 분석기 - Streamlit 앱."""
from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Iterator
from datetime import datetime

import streamlit as st

from utils.config import get_api_key, get_app_password, get_model
from utils.llm import (
    CONTEXT_TOKEN_LIMIT,
    LLMError,
    Usage,
    build_qa_messages,
    count_tokens,
    docs_to_text,
    needs_chunking,
    prepare_analysis,
    stream_chat,
    temperature_supported,
)
from utils.pdf_parser import PdfDocument, extract_pdf
from utils.prompts import ANALYSIS_DESCRIPTIONS, ANALYSIS_TYPES, LANGUAGES
from utils.text import fix_korean_emphasis

st.set_page_config(page_title="PDF 분석기", page_icon="📄", layout="wide")

# 최소한의 스타일: 직접 만든 클래스(.hero, .steps)와 data-testid 만 사용
st.markdown(
    """
<style>
[data-testid="stMainBlockContainer"] { padding-top: 2.5rem; max-width: 1100px; }
.hero h1 { font-size: 2rem; font-weight: 700; margin: 0 0 .25rem 0; padding: 0; letter-spacing: -0.02em; }
.hero p { margin: 0; opacity: .72; }
.steps { display: flex; flex-wrap: wrap; gap: .5rem; margin: 1rem 0 .5rem; padding: 0; list-style: none; }
.steps li {
  display: flex; align-items: center; gap: .45rem; font-size: .9rem;
  padding: .35rem .8rem; border-radius: 999px;
  border: 1px solid color-mix(in srgb, currentColor 18%, transparent);
}
.steps li .n {
  display: inline-grid; place-items: center; width: 1.3rem; height: 1.3rem; border-radius: 50%;
  font-size: .75rem; font-weight: 700; background: color-mix(in srgb, currentColor 12%, transparent);
}
.steps li.done { border-color: #2F9E6B; }
.steps li.done .n { background: #2F9E6B; color: #fff; }
.steps li.done .label::after { content: " ✓"; color: #2F9E6B; font-weight: 700; }
/* AI 결과/답변 안의 Markdown 제목이 너무 커지지 않도록 */
[data-testid="stTabPanel"] [data-testid="stMarkdownContainer"] h1 { font-size: 1.45rem; }
[data-testid="stTabPanel"] [data-testid="stMarkdownContainer"] h2 { font-size: 1.3rem; }
[data-testid="stTabPanel"] [data-testid="stMarkdownContainer"] h3 { font-size: 1.15rem; }
[data-testid="stTabPanel"] [data-testid="stMarkdownContainer"] h4 { font-size: 1.02rem; }
/* 문서 요약 지표: 데스크톱 4열, 모바일 2열 */
.stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .75rem; margin: .25rem 0 .75rem; }
.stats > div { padding: .75rem 1rem; border-radius: .6rem;
  border: 1px solid color-mix(in srgb, currentColor 14%, transparent); }
.stats .k { font-size: .82rem; opacity: .7; }
.stats .v { font-size: 1.45rem; font-weight: 700; letter-spacing: -0.01em; }
@media (max-width: 640px) {
  [data-testid="stMainBlockContainer"] { padding-top: 4rem; }
  .hero h1 { font-size: 1.6rem; }
  .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .stats .v { font-size: 1.2rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------- 접근 비밀번호 (APP_PASSWORD 를 설정한 경우에만) ----------------
_app_password = get_app_password()
if _app_password and not st.session_state.get("authenticated"):
    st.markdown("### 🔒 PDF 분석기")
    with st.form("login"):
        pw = st.text_input("비밀번호", type="password")
        if st.form_submit_button("입장", type="primary"):
            if hmac.compare_digest(pw.encode(), _app_password.encode()):
                st.session_state.authenticated = True
                st.rerun()
            st.error("비밀번호가 올바르지 않습니다.")
    st.stop()


@st.cache_data(show_spinner=False, max_entries=50)
def cached_extract(name: str, data: bytes) -> PdfDocument:
    return extract_pdf(name, data)


@st.cache_data(show_spinner=False, max_entries=20)
def cached_token_count(text: str) -> int:
    return count_tokens(text)


def md_escape(text: str) -> str:
    """파일명 등 사용자 문자열을 Markdown 으로 안전하게 출력."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|<>~$])", r"\\\1", text)


def show_markdown(text: str, target=st) -> None:
    target.markdown(fix_korean_emphasis(text))


# ---------------- 세션 상태 ----------------
if "docs_signature" not in st.session_state:
    st.session_state.docs_signature = None
    st.session_state.analysis_result = None  # dict(text, type, language, files, usage, model, created)
    st.session_state.chat_history = []  # [{"role", "content"}]
    st.session_state.total_usage = Usage()
    st.session_state.pending_question = None
    # 스트리밍 중인 내용. 실행이 끝나면 None 으로 되돌린다.
    # 다음 실행 시작 때 None 이 아니면 = 이전 실행이 사용자 조작으로 중단된 것.
    st.session_state.analysis_partial = None  # dict(type, language, text)
    st.session_state.analysis_interrupted = None
    st.session_state.chat_partial = None  # dict(question, answer)

# 중단된 실행 복구: 사용자가 분석/답변 도중 다른 위젯을 조작하면 Streamlit 이 실행을 멈춘다.
if st.session_state.analysis_partial is not None:
    st.session_state.analysis_interrupted = st.session_state.analysis_partial
    st.session_state.analysis_partial = None
if st.session_state.chat_partial is not None:
    _cp = st.session_state.chat_partial
    st.session_state.chat_history += [
        {"role": "user", "content": _cp["question"]},
        {"role": "assistant", "content": (_cp["answer"] + "\n\n" if _cp["answer"] else "")
         + "*⚠️ 답변 도중 다른 조작이 있어 중단되었습니다. 다시 질문해 주세요.*"},
    ]
    st.session_state.chat_partial = None


def tee_into(gen: Iterator[str], holder: dict, field: str) -> Iterator[str]:
    """스트리밍 조각을 화면에 넘기면서 session_state 에도 누적 (중단 대비)."""
    for chunk in gen:
        holder[field] += chunk
        yield chunk


def reset_chat() -> None:
    st.session_state.chat_history = []


def ask(question: str) -> None:
    st.session_state.pending_question = question


# ---------------- 헤더 (실행 끝에 그려서 이번 실행의 진행 상태까지 반영) ----------------
def render_hero(has_docs: bool) -> None:
    step_done = [
        has_docs,
        st.session_state.analysis_result is not None,
        bool(st.session_state.chat_history),
    ]
    steps_html = "".join(
        f'<li class="{"done" if done else ""}"><span class="n">{i}</span><span class="label">{label}</span></li>'
        for i, (label, done) in enumerate(zip(["PDF 업로드", "AI 분석", "문서에 질문"], step_done), start=1)
    )
    hero_slot.markdown(
        f"""
<div class="hero">
  <h1>📄 PDF 분석기</h1>
  <p>PDF를 올리면 AI가 내용을 요약·분석하고, 문서에 대한 질문에 출처와 함께 답해 드립니다.</p>
  <ol class="steps">{steps_html}</ol>
</div>
""",
        unsafe_allow_html=True,
    )


def render_usage() -> None:
    tu: Usage = st.session_state.total_usage
    usage_slot.caption(f"이번 세션 사용 토큰: 입력 {tu.input_tokens:,} / 출력 {tu.output_tokens:,}")


hero_slot = st.empty()

# ---------------- 사이드바 ----------------
with st.sidebar:
    st.subheader("① PDF 업로드")
    uploaded_files = st.file_uploader(
        "PDF 파일을 끌어다 놓거나 선택하세요",
        type=["pdf"],
        accept_multiple_files=True,
        help="여러 개를 한 번에 올릴 수 있습니다. 텍스트가 있는 PDF만 분석됩니다.",
    )

    st.subheader("② 분석 설정")
    analysis_type = st.selectbox("분석 유형", ANALYSIS_TYPES)
    language_label = st.radio("응답 언어", list(LANGUAGES.keys()), horizontal=True)
    language = LANGUAGES[language_label]

    if temperature_supported():
        temperature = st.slider(
            "창의성 (temperature)", 0.0, 1.0, 0.3, 0.1,
            help="낮을수록 사실 위주로, 높을수록 다양하게 답합니다.",
        )
    else:
        temperature = None

    st.subheader("정보")
    with st.container(border=True):
        if get_api_key():
            st.markdown("✅ **API 키 연결됨**")
        else:
            st.markdown("❌ **API 키 없음**")
            st.caption("`.env` 또는 Streamlit Secrets에 `OPENAI_API_KEY`를 설정하세요.")
        st.caption(f"모델: `{get_model()}`")
        if temperature is None:
            st.caption("이 모델은 temperature 조절을 지원하지 않아 기본값으로 동작합니다.")
        usage_slot = st.empty()  # 스크립트 끝에서 채움

# ---------------- 문서 추출 ----------------
render_hero(bool(uploaded_files))  # 추출 중에도 헤더가 보이도록 먼저 그림 (끝에서 갱신)
docs: list[PdfDocument] = []
file_hashes: list[tuple[str, str]] = []
if uploaded_files:
    with st.spinner(f"PDF {len(uploaded_files)}개에서 텍스트를 추출하는 중..."):
        for f in uploaded_files:
            data = f.getvalue()
            file_hashes.append((f.name, hashlib.sha1(data).hexdigest()))
            docs.append(cached_extract(f.name, data))

valid_docs = [d for d in docs if d.has_text]

# 업로드 구성이 바뀌면 이전 분석 결과와 대화 초기화
signature = tuple(file_hashes)
if signature != st.session_state.docs_signature:
    st.session_state.docs_signature = signature
    st.session_state.analysis_result = None
    st.session_state.analysis_interrupted = None
    st.session_state.pending_question = None
    reset_chat()

NO_TEXT_MSG = "텍스트를 추출할 수 있는 PDF가 없습니다. 스캔 이미지가 아닌 PDF를 올려주세요."

# ---------------- 빈 상태 ----------------
if not docs:
    st.write("")
    st.info("👈 왼쪽 사이드바에서 PDF 파일을 업로드해주세요. (모바일은 왼쪽 위 **»** 버튼)")
    cards = [
        ("📝", "요약 · 분석", "요약, 핵심 포인트, 상세 분석 등 원하는 방식으로 문서를 정리합니다."),
        ("⚖️", "문서 비교", "여러 PDF를 올리면 공통점과 차이점을 표로 비교합니다."),
        ("💬", "문서 Q&A", "궁금한 점을 질문하면 페이지 출처와 함께 답합니다."),
    ]
    cols = st.columns(3)
    for col, (icon, title, desc) in zip(cols, cards):
        with col.container(border=True, height="stretch"):
            st.markdown(f"### {icon}")
            st.markdown(f"**{title}**")
            st.caption(desc)
    render_hero(bool(valid_docs))
    render_usage()
    st.stop()

tab_info, tab_analysis, tab_chat = st.tabs(["📄 문서 정보", "🔍 분석", "💬 Q&A"])

# ---------------- 문서 정보 ----------------
with tab_info:
    total_tokens = cached_token_count(docs_to_text(valid_docs)) if valid_docs else 0
    stats = [
        ("문서 수", f"{len(docs)}개"),
        ("총 페이지", f"{sum(d.page_count for d in docs):,}"),
        ("총 글자 수", f"{sum(d.char_count for d in docs):,}"),
        ("예상 토큰", f"{total_tokens:,}"),
    ]  # 숫자만 들어가므로 HTML 로 출력해도 안전
    st.markdown(
        '<div class="stats">'
        + "".join(f'<div><div class="k">{k}</div><div class="v">{v}</div></div>' for k, v in stats)
        + "</div>",
        unsafe_allow_html=True,
    )
    if total_tokens > CONTEXT_TOKEN_LIMIT:
        st.info(
            f"문서가 길어({total_tokens:,} 토큰) 분석할 때 여러 부분으로 나눠 요약한 뒤 통합합니다. "
            "시간이 조금 더 걸릴 수 있습니다.",
            icon="ℹ️",
        )

    for idx, doc in enumerate(docs):
        with st.container(border=True):
            head = st.columns([5, 1], vertical_alignment="center")
            head[0].markdown(f"**📄 {md_escape(doc.name)}**")
            with head[1]:
                if doc.error:
                    st.badge("확인 필요", icon=":material/warning:", color="orange")
                else:
                    st.badge("분석 가능", icon=":material/check_circle:", color="green")
            if doc.error:
                st.warning(doc.error, icon="⚠️")
            st.caption(f"{doc.page_count}페이지 · {doc.char_count:,}자")
            if doc.has_text:
                with st.expander("텍스트 미리보기"):
                    st.text_area(
                        "추출된 텍스트 (앞부분 2,000자)",
                        doc.full_text()[:2000],
                        height=220,
                        disabled=True,
                        key=f"preview_{idx}",
                        label_visibility="collapsed",
                    )


def build_markdown(result: dict) -> str:
    files = "\n".join(f"- {n}" for n in result["files"])
    return (
        f"# PDF 분석 결과 - {result['type']}\n\n"
        f"- 생성 시각: {result['created']}\n- 모델: {result['model']}\n\n"
        f"## 대상 문서\n{files}\n\n---\n\n{result['text']}\n"
    )


def render_result_footer(result: dict) -> None:
    u: Usage = result["usage"]
    foot = st.columns([3, 2], vertical_alignment="center")
    foot[0].caption(
        f"🕒 {result['created']} · {result['type']} · {result['language']} · "
        f"토큰 입력 {u.input_tokens:,} / 출력 {u.output_tokens:,}"
    )
    foot[1].download_button(
        "Markdown으로 다운로드",
        icon=":material/download:",
        data=build_markdown(result).encode("utf-8"),
        file_name=f"pdf_analysis_{result['created'].replace('-', '').replace(':', '').replace(' ', '_')}.md",
        mime="text/markdown",
        on_click="ignore",  # 다운로드만 하고 앱을 다시 실행하지 않음
        use_container_width=True,
    )


# ---------------- 분석 ----------------
with tab_analysis:
    with st.container(border=True):
        st.markdown(f"**분석 유형 · {analysis_type}**")
        st.caption(f"{ANALYSIS_DESCRIPTIONS[analysis_type]}  (사이드바에서 유형과 응답 언어를 바꿀 수 있습니다)")
        st.caption("대상: " + ", ".join(md_escape(d.name) for d in valid_docs) if valid_docs else NO_TEXT_MSG)

        custom_instruction = ""
        if analysis_type == "사용자 지정":
            custom_instruction = st.text_area(
                "분석 지시문",
                placeholder="예: 이 문서에서 비용과 관련된 내용만 표로 정리해줘",
            )

        disabled_reason = None
        if not valid_docs:
            disabled_reason = NO_TEXT_MSG
        elif analysis_type == "문서 비교" and len(valid_docs) < 2:
            disabled_reason = "문서 비교는 텍스트가 있는 PDF가 2개 이상 필요합니다. 사이드바에서 PDF를 더 올려주세요."
        elif analysis_type == "사용자 지정" and not custom_instruction.strip():
            disabled_reason = "분석 지시문을 입력하면 실행할 수 있습니다."
        elif not get_api_key():
            disabled_reason = "OpenAI API 키가 설정되지 않았습니다. `.env` 또는 Streamlit Secrets를 확인하세요."

        run = st.button(
            "분석 실행",
            icon=":material/auto_awesome:",
            type="primary",
            disabled=disabled_reason is not None,
            use_container_width=True,
        )
        if disabled_reason:
            st.caption(f"ℹ️ {disabled_reason}")

    if run:
        st.session_state.analysis_result = None
        st.session_state.analysis_interrupted = None
        partial = {"type": analysis_type, "language": language_label, "text": ""}
        st.session_state.analysis_partial = partial
        usage = Usage()
        hint = st.empty()
        hint.caption("⏳ 분석이 끝날 때까지 다른 버튼이나 설정을 바꾸지 마세요. 조작하면 분석이 중단됩니다.")
        try:
            if needs_chunking(valid_docs):
                with st.status("긴 문서를 나눠서 요약하는 중...", expanded=True) as status:
                    progress = st.progress(0.0)

                    def on_progress(i: int, n: int) -> None:
                        progress.progress((i - 1) / n, text=f"{i}/{n} 부분 요약 중...")
                        status.update(label=f"긴 문서를 나눠서 요약하는 중... ({i}/{n})")

                    messages = prepare_analysis(
                        valid_docs, analysis_type, language, custom_instruction, usage, temperature, on_progress
                    )
                    progress.progress(1.0, text="부분 요약 완료")
                    status.update(label="부분 요약 완료 · 통합 분석을 작성합니다", state="complete", expanded=False)
            else:
                messages = prepare_analysis(valid_docs, analysis_type, language, custom_instruction, usage, temperature)

            with st.container(border=True):
                slot = st.empty()
                with st.spinner("AI가 문서를 분석하고 있습니다... 결과가 나오는 대로 표시됩니다."):
                    text = slot.write_stream(tee_into(stream_chat(messages, usage, temperature), partial, "text"))
                text = text if isinstance(text, str) else "".join(map(str, text))
                show_markdown(text, slot)  # 한국어 굵은 글씨 보정 후 다시 그림
            st.session_state.analysis_result = {
                "text": text,
                "type": analysis_type,
                "language": language_label,
                "files": [d.name for d in valid_docs],
                "usage": usage,
                "model": get_model(),
                "created": f"{datetime.now():%Y-%m-%d %H:%M}",
            }
            render_result_footer(st.session_state.analysis_result)
        except LLMError as e:
            st.error(str(e), icon="🚫")
        finally:
            st.session_state.total_usage.merge(usage)
        # 정상 종료/LLMError 일 때만 여기 도달한다. 사용자 조작으로 실행이 중단되면 Streamlit 의
        # 중단 예외가 위로 빠져나가 이 줄이 실행되지 않으므로, partial 이 남아 다음 실행에서 복구된다.
        st.session_state.analysis_partial = None
        hint.empty()
    elif st.session_state.analysis_result:
        result = st.session_state.analysis_result
        with st.container(border=True):
            show_markdown(result["text"])
        render_result_footer(result)
    elif st.session_state.analysis_interrupted:
        intr = st.session_state.analysis_interrupted
        st.warning(
            "분석 도중 다른 조작이 있어 분석이 중단되었습니다. **분석 실행**을 다시 눌러주세요."
            + (" 아래는 중단 전까지 받은 내용입니다." if intr["text"] else ""),
            icon="⚠️",
        )
        if intr["text"]:
            with st.container(border=True):
                show_markdown(intr["text"])
    else:
        st.caption("분석 결과가 여기에 표시됩니다.")

# ---------------- Q&A ----------------
USER_AVATAR = ":material/person:"
AI_AVATAR = ":material/smart_toy:"
EXAMPLE_QUESTIONS = [
    "이 문서의 핵심 내용을 3줄로 알려줘",
    "중요한 숫자나 날짜를 정리해줘",
    "주의해야 할 점이나 예외 사항은?",
]

with tab_chat:
    if not valid_docs:
        st.info(NO_TEXT_MSG)
    else:
        top = st.columns([4, 1], vertical_alignment="center")
        top[0].caption("업로드한 문서 내용만을 바탕으로 답하며, 답변에 (파일명, 페이지) 출처를 표시합니다.")
        top[1].button(
            "대화 초기화",
            icon=":material/delete_sweep:",
            on_click=reset_chat,
            use_container_width=True,
            disabled=not st.session_state.chat_history,
        )

        messages_box = st.container()
        typed = st.chat_input("문서에 대해 질문하세요", disabled=not get_api_key())
        question = typed or st.session_state.pending_question
        st.session_state.pending_question = None

        with messages_box:
            if not st.session_state.chat_history and not question:
                st.markdown("**이런 질문을 해보세요**")
                ex_cols = st.columns(len(EXAMPLE_QUESTIONS))
                for col, q in zip(ex_cols, EXAMPLE_QUESTIONS):
                    col.button(
                        q, on_click=ask, args=(q,), use_container_width=True,
                        disabled=not get_api_key(), key=f"ex_{q}",
                    )

            for m in st.session_state.chat_history:
                with st.chat_message(m["role"], avatar=USER_AVATAR if m["role"] == "user" else AI_AVATAR):
                    if m["role"] == "user":
                        st.markdown(md_escape(m["content"]))  # 사용자가 입력한 그대로 표시
                    else:
                        show_markdown(m["content"])

            if question:
                with st.chat_message("user", avatar=USER_AVATAR):
                    st.markdown(md_escape(question))
                usage = Usage()
                with st.chat_message("assistant", avatar=AI_AVATAR):
                    try:
                        msgs = build_qa_messages(valid_docs, st.session_state.chat_history, question, language)
                        cp = {"question": question, "answer": ""}
                        st.session_state.chat_partial = cp
                        slot = st.empty()
                        with st.spinner("문서에서 답을 찾는 중..."):
                            answer = slot.write_stream(tee_into(stream_chat(msgs, usage, temperature), cp, "answer"))
                        answer = answer if isinstance(answer, str) else "".join(map(str, answer))
                        show_markdown(answer, slot)
                        st.session_state.chat_history += [
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": answer},
                        ]
                    except LLMError as e:
                        st.error(str(e), icon="🚫")
                    finally:
                        st.session_state.total_usage.merge(usage)
                    st.session_state.chat_partial = None  # 정상/오류 종료 (중단 시에는 남아서 다음 실행에서 복구)

# ---------------- 마무리: 헤더 / 누적 사용량 ----------------
render_hero(bool(valid_docs))
render_usage()
