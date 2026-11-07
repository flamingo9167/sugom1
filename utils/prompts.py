"""분석 유형별 프롬프트."""
from __future__ import annotations

ANALYSIS_TYPES = ["요약", "핵심 포인트", "상세 분석", "문서 비교", "사용자 지정"]

ANALYSIS_DESCRIPTIONS = {
    "요약": "문서별 핵심 내용을 3~5문단으로 요약합니다.",
    "핵심 포인트": "주요 내용을 불릿 포인트로 정리합니다.",
    "상세 분석": "구조, 주요 주장, 근거, 시사점, 한계를 분석합니다.",
    "문서 비교": "2개 이상의 문서를 비교해 공통점과 차이점을 정리합니다.",
    "사용자 지정": "직접 입력한 지시에 따라 분석합니다.",
}

LANGUAGES = {"한국어": "Korean", "English": "English"}

COMMON_RULES = """\
You are an expert document analyst. The user's PDF documents are provided below.
Each page is marked like "[file name - p.N]".

Rules:
- Write the entire answer in {language}. Use Markdown formatting.
- Base every statement only on the provided documents. Never invent facts.
- Cite sources inline in the form (file name, p.N) for key claims. Put several pages of the
  same file in one citation, e.g. (file name, p.3-4, p.6), and cite once per sentence at most.
- Use "###" or smaller for headings.
- If something is not in the documents, say so explicitly instead of guessing.
"""

TASKS = {
    "요약": """\
Task: Summarize each document separately.
For each document, use a "### <file name>" heading and write a 3-5 paragraph summary
covering its purpose, main content, and conclusions.""",
    "핵심 포인트": """\
Task: Extract the key points of each document.
For each document, use a "### <file name>" heading followed by 5-10 concise bullet points.
End with a short "Key takeaways" section of 2-3 bullets across all documents.""",
    "상세 분석": """\
Task: Analyze each document in depth.
For each document, use a "### <file name>" heading with these subsections:
1. Structure (how the document is organized)
2. Main claims or content
3. Supporting evidence / data
4. Implications
5. Limitations or points to watch""",
    "문서 비교": """\
Task: Compare the documents.
1. One-line overview of each document.
2. A Markdown table comparing them on purpose, topic, key content, audience, and conclusion.
3. Commonalities (bullets).
4. Differences (bullets).
5. Overall assessment.""",
}

QA_SYSTEM = """\
You are a helpful assistant answering questions about the user's PDF documents.
The relevant document pages are provided below; each page is marked like "[file name - p.N]".

Rules:
- Answer in {language}, concisely, using Markdown.
- Answer only from the provided documents and cite sources as (file name, p.N).
  Put several pages of the same file in one citation, e.g. (file name, p.3-4).
- Use "####" or smaller for headings; prefer short paragraphs and bullet lists.
- If the answer is not in the documents, reply that it cannot be found in the documents
  (in Korean: "문서에서 확인할 수 없습니다.").
"""

MAP_PROMPT = """\
The following is part {index} of {total} of the user's documents.
Summarize this part in {language} in detail, preserving key facts, numbers, names,
and the (file name, p.N) source markers so they can be cited later.
"""

REDUCE_NOTE = """\
Note: the documents were too long to send in full, so you are given detailed partial
summaries (with source markers) instead of the original text.
"""


def build_system_prompt(analysis_type: str, language: str, custom_instruction: str = "") -> str:
    if analysis_type == "사용자 지정":
        task = f"Task (user's instruction):\n{custom_instruction.strip()}"
    else:
        task = TASKS[analysis_type]
    return COMMON_RULES.format(language=language) + "\n" + task
