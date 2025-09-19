"""
Main Global Planner system - 완전 정리된 버전
"""
import numpy as np
import re, os, json
import sys
from typing import List, Dict, Tuple, Any
from collections import defaultdict, Counter
import pickle
from functools import lru_cache

from mv_index import MultiviewIndex
# GlobalPlanner.py 상단의 import 부분을 이렇게 수정하세요:

from rag_selector import (
    select_examples_with_multiview,
    select_examples_env_act_hybrid,
    select_examples_env_act_centroid_from_vectors,
    select_examples_env_act_centroid_from_vectors_flexible,
    select_examples_env_act_centroid_with_flexible_matching,  # 이 줄 추가
    select_examples_env_act_centroid,
    _hydrate_task,
    _parse_breakdown_tasks,
)

from core_config import (
    BREAKDOWN_TEMPERATURE, BREAKDOWN_MAX_TOKENS, 
    PLAN_TEMPERATURE, PLAN_MAX_TOKENS,
    OPENAI_OK, OPENAI_ERR, BASE_DIR,
    call_chat, print_info, ask_yes_no, norm
)

from long_term_plan_cache import LongTermPlanCache
from rag_feedback_db import RAGFeedbackDB
from act_feedback_db import ActFeedbackDB

from alias_feedback import (
    canon_set, has_alias_in_set, ask_and_record_feedback,
    alias_candidates, synthesize_chunk_from_registry
)
from rag_cache_system import (
    try_cache_first_from_breakdown, try_feedback_cache, 
    persist_plan_to_longterm_cache, _inject_numbered_steps
)

from clustered_operator_registry import (build_clustered_registry, save_clustered_registry, ClusteredOperatorDB)
reg = build_clustered_registry('task_clustering_results.json')
save_clustered_registry(reg, 'cor_registry.json')

# optional canonicalizer (safe if missing)
try:
    from clustered_operator_registry import load_clustered_registry
    from act_canon import Canon
    _COR = load_clustered_registry('cor_registry.json')
    _CANON = Canon(_COR)
except Exception:
    _CANON = None

# JSON 인덱스 전역 변수
_TASK_JSON_INDEX = None
_TASK_BY_SUB = None
FORMAT_HEADER_RE = re.compile(r"===\s*(최종\s*자동화\s*계획|Final\s*Automation\s*Plan)\s*===")
_MV_MODEL = None
ENABLE_FULL_CONTEXT_PROMPT = True
_CHUNK_LINE_RE = re.compile(r"^-\s*CHUNK\s+(\d+):\s*\[(.*?)\]\s*\(steps\s+(\d+)-(\d+)\)")


def _load_multiview_model(pkl_path: str):
    """멀티뷰 모델 로드 (전역에서 한 번만)"""
    global _MV_MODEL
    if _MV_MODEL is None:
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            # multiview_clustering.py에서 생성된 구조 확인
            if hasattr(data, 'clusterer') and hasattr(data, 'model'):
                _MV_MODEL = data
                print(f"멀티뷰 모델 로드 성공: {pkl_path}")
            else:
                print(f"경고: 예상과 다른 pkl 구조: {type(data)}")
                _MV_MODEL = data
                
        except Exception as e:
            print(f"멀티뷰 모델 로드 실패: {e}")
            _MV_MODEL = None
    
    return _MV_MODEL

def _encode_query_mv(text: str) -> np.ndarray:
    """쿼리 텍스트를 멀티뷰 모델로 인코딩"""
    global _MV_MODEL
    if _MV_MODEL is None:
        # fallback: OpenAI 임베딩 사용
        from openai import OpenAI
        client = OpenAI()
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=[text]
        )
        return np.array(response.data[0].embedding)
    
    try:
        # multiview_clustering.py의 embed 함수 사용
        from multiview_clustering import embed
        return np.array(embed(text))
    except Exception as e:
        print(f"인코딩 실패: {e}")
        return np.random.normal(0, 0.1, 1536)  # fallback

def _load_mv_index(pkl_path: str):
    """MultiviewIndex 인스턴스 생성"""
    return MultiviewIndex(pkl_path)

def _mv_encode(idx, text: str):
    """텍스트 인코딩"""
    if hasattr(idx, 'encode'):
        return idx.encode(text)
    # 간단한 fallback
    return np.array([hash(w) % 100003 for w in text.split()]).astype(float)

def _hydrate_task_item(task_id: str):
    """호환용 래퍼 - rag_selector의 통일된 함수 사용"""
    global _TASK_JSON_INDEX
    if not _TASK_JSON_INDEX:
        return None
    return _hydrate_task(task_id, _TASK_JSON_INDEX)

def _initialize_json_index():
    """JSON 인덱스 초기화"""
    global _TASK_JSON_INDEX, _TASK_BY_SUB
    try:
        json_path = os.path.join(BASE_DIR, "task_clustering_results.json")
        _TASK_JSON_INDEX, _TASK_BY_SUB = load_task_json_index(json_path)
        print(f"JSON 인덱스 로드 완료: {len(_TASK_JSON_INDEX)}개 태스크")
    except Exception as e:
        print(f"JSON 인덱스 로드 실패: {e}")
        _TASK_JSON_INDEX, _TASK_BY_SUB = {}, {}

def _extract_steps_verbatim(ex: dict) -> list[str]:
    """
    예시(dict)에서 '실제 스텝'을 가능한 많이 추출.
    - "N. 0#event, arg" / "0#event, arg" / "1#event, arg" 모두 허용
    - steps 문자열에 탭(\t)·연속공백으로 이어진 다중 스텝을 '개별 스텝'으로 분리
    - content/기타 필드에서도 추출 (느슨한 패턴)
    - 중복 제거, 상한 없음
    """
    import re
    seen = set()
    steps: list[str] = []

    # 탭·공백으로 이어진 스텝을 '다음 스텝 토큰([01]#...)'이 나타나기 전까지로 잘라내는 세그먼터
    step_pat = re.compile(
        r'(?:\d+\.\s*)?[01]#[^,]+,\s*.*?(?=(?:\s+(?:\d+\.\s*)?[01]#)|\s*$)',
        flags=re.S
    )

    def _push_many(text: str):
        if not text:
            return
        # 개별 스텝 토큰으로 분리
        for m in step_pat.findall(text.replace('\r', ' ')):
            s = m.strip()
            if not s:
                continue
            key = re.sub(r'\s+', ' ', s.lower())
            if key not in seen:
                seen.add(key)
                steps.append(s)

    # 1) steps 필드
    raw_steps = ex.get("steps")
    if isinstance(raw_steps, str):
        _push_many(raw_steps)
    elif isinstance(raw_steps, (list, tuple)):
        for item in raw_steps:
            _push_many(str(item))

    # 2) content / des_text
    if not steps:
        content = ex.get("content") or ex.get("des_text") or ""
        _push_many(content)

    # 3) 기타 필드
    if not steps:
        for field in ("title", "file_name", "act_text"):
            txt = ex.get(field)
            if isinstance(txt, str) and "#" in txt:
                _push_many(txt)

    return steps

def _render_full_context_logs(placeholder_map: dict) -> str:
    """
    placeholder_map 형태가 {tag: ex} 또는 {tag: [ex,...]} 모두 지원.
    각 ex의 '전체 스텝'을 그대로 나열(번호는 보기용으로만 붙임).
    """
    if not placeholder_map:
        return "No context logs."

    import re
    blocks: list[str] = []

    def _one_ex(tag: str, ex: dict, idx: int):
        file_name = ex.get("file_name") or ex.get("src") or ex.get("id") or "unknown"
        env_info  = ", ".join(ex.get("env_tag", []) or []) or ex.get("env_text") or ex.get("env") or "unknown"
        act_info  = ex.get("act_tag") or ex.get("act_text") or ex.get("act") or "unknown"
        title     = ex.get("title") or ""

        lines = [f"=== {tag}:{idx} ===",
                 f"FILE: {file_name}",
                 f"ENV: {env_info}",
                 f"ACT: {act_info}"]
        if title:
            lines.append(f"TITLE: {title}")

        st = _extract_steps_verbatim(ex)
        if st:
            lines.append("STEPS:")
            for i, s in enumerate(st, 1):
                if re.match(r'^\s*\d+\.\s*[01]#', s):
                    lines.append(f"  {s}")
                else:
                    lines.append(f"  {i}. {s}")
        else:
            lines.append("STEPS: (none)")

        blocks.append("\n".join(lines))

    for tag, val in placeholder_map.items():
        if isinstance(val, dict) and ("steps" in val or "content" in val or "des_text" in val):
            _one_ex(tag, val, 1)
        elif isinstance(val, (list, tuple)):
            for i, ex in enumerate(val, 1):
                if isinstance(ex, dict):
                    _one_ex(tag, ex, i)

    return "\n".join(blocks)

# ====== NEW: JSON hydration (id -> steps/title/act) ======
def load_task_json_index(json_path: str):
    data = _load_json(json_path)
    results = data.get("results") or data
    env_clusters = results.get("env_clusters") or {}
    by_task_id = {}
    by_sub_id = defaultdict(list)
    for cid, cinfo in env_clusters.items():
        for sid, sinfo in (cinfo.get("act_des_subclusters") or {}).items():
            for t in (sinfo.get("tasks") or []):
                tid = t.get("unique_id") or t.get("id") or ""
                if not tid: 
                    continue
                rec = {
                    "task_id": tid,
                    "cluster_id": str(cid),
                    "subcluster_id": str(sid),
                    "title": t.get("title") or t.get("des_text") or "",
                    "act_text": t.get("act_text") or t.get("act_tag") or "",
                    "steps": t.get("steps") or "",
                    "file_name": t.get("file_name") or "",
                }
                by_task_id[tid] = rec
                by_sub_id[str(sid)].append(tid)
    return by_task_id, by_sub_id
    
def _canonize_act_for_task(act_list: list[str], env_full_key: str) -> str:
    """act_canon이 있으면 사용, 없으면 간단 토큰 폴백"""
    for a in act_list or []:
        if not a:
            continue
        if _CANON:
            try:
                canon = _CANON.canonicalize(a, env_app_hint=env_full_key)
                if canon:
                    return canon
            except Exception:
                pass
        # fallback: 간단 정규화
        return re.sub(r"\s+", "_", _norm_text(a))
    return "search"

# --- small JSON loader (ADD THIS) ---
def _load_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise RuntimeError(f"JSON file not found: {path}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in {path}: {e}")

# ===== JSON steps helper (kept) =====
def _split_steps_field(unit) -> list:
    s = unit.get("steps")
    if not s:
        return []
    if isinstance(s, list):
        raw = s
    else:
        raw = re.split(r"[\t\r\n]+", str(s))
    out = []
    pat = re.compile(r"^\s*[01]#\s*[^,]+,\s*.+$")
    for ln in raw:
        t = (ln or "").strip()
        if t and pat.match(t):
            out.append(t)
    return out

# ===== ENV→Cluster scoping =====
TOKEN_SPLIT = re.compile(r"[^a-z0-9가-힣]+", re.IGNORECASE)

# ===== ENV helpers (REPLACE/ADD) =====
TOKEN_SPLIT = re.compile(r"[^a-z0-9가-힣]+", re.IGNORECASE)
ENV_SPLITTER = re.compile(r"\s*,\s*")

def _norm_text(s: str) -> str:
    s = (s or "").lower().replace("_", " ")
    s = TOKEN_SPLIT.sub(" ", s)
    return " ".join(t for t in s.split() if t)

def _env_key(s: str) -> str:
    """ENV 문자열을 비교용 키로 정규화 (소문자, 슬래시 통일, 공백/하이픈 유지)"""
    s = (s or "").strip().replace("\\", "/")
    s = re.sub(r"\s+", " ", s)
    return s.lower()

def _env_base_key(env_full: str) -> str:
    """하이픈 앞까지만 남겨 베이스 ENV 추출 (예: web/chrome-skyscanner -> web/chrome)"""
    k = _env_key(env_full)
    return k.split("-", 1)[0] if "-" in k else k

def _envs_from_field(field) -> list[str]:
    """env_tag(list) 또는 env_text(str, 'a, b' 형태)를 리스트로 평탄화 + 정규화"""
    out = []
    if isinstance(field, list):
        for x in field:
            out.extend(_envs_from_field(x))
    elif field:
        for piece in ENV_SPLITTER.split(str(field)):
            p = piece.strip()
            if p:
                out.append(_env_key(p))
    return out

def _extract_env_from_task_line(tl: str) -> str:
    m = re.search(r"ENV\[([^\]]+)\]", tl or "")
    if not m:
        return ""
    # Task 라인에 여러 ENV가 올 수 있어도 첫 번째를 "해당 Task의 주 ENV"로 사용
    first_env = m.group(1).split(",", 1)[0].strip()
    return _env_key(first_env)

def _task_lines(header_text: str):
    return [ln.strip() for ln in (header_text or "").splitlines() if ln.strip().startswith("Task#")]

def _bag_for_task_unit(tu: dict) -> Counter:
    bag = Counter()
    bag.update(_norm_text(tu.get("title") or "").split())
    bag.update(_norm_text(tu.get("act_text") or tu.get("act_tag") or "").split())
    bag.update(_norm_text(tu.get("des_text") or "").split())
    return bag

# 통일된 라우팅 함수
# GlobalPlanner.py의 unified_rag_select_examples() 함수 수정

# GlobalPlanner.py 파일에서 다음 수정사항 적용

# 1. import 부분에 새 함수 추가 (기존 import 문에 추가)
from rag_selector import (
    select_examples_with_multiview,
    select_examples_env_act_hybrid,
    select_examples_env_act_centroid_from_vectors,
    select_examples_env_act_centroid_from_vectors_flexible,
    select_examples_env_act_centroid_with_flexible_matching,  # 이 줄 추가
    select_examples_env_act_centroid,
    _hydrate_task,
    _parse_breakdown_tasks,
)

# 2. unified_rag_select_examples 함수를 찾아서 다음으로 교체
def unified_rag_select_examples(command: str, header_text: str, top_total: int = 9999):
    """
    개선된 라우팅 우선순위:
    1) 벡터 PKL 2종 있으면 -> 유연한 centroid 매칭 경로  
    2) 결과 PKL 2종 있으면 -> 결과 PKL 센트로이드 경로
    3) 아니면 -> 멀티뷰
    """
    global _TASK_JSON_INDEX, _TASK_BY_SUB
    if _TASK_JSON_INDEX is None:
        _initialize_json_index()

    model_path = os.path.join(BASE_DIR, "multiview_clustering_model.pkl")
    task_res_pkl = os.path.join(BASE_DIR, "task_clustering_results.pkl")
    unit_res_pkl = os.path.join(BASE_DIR, "task_unit_clustering_results.pkl")
    task_vec_pkl = os.path.join(BASE_DIR, "task_vectors.pkl")
    unit_vec_pkl = os.path.join(BASE_DIR, "task_unit_vectors.pkl")

    have_vecs = os.path.exists(task_vec_pkl) and os.path.exists(unit_vec_pkl)
    have_results = os.path.exists(task_res_pkl) and os.path.exists(unit_res_pkl)

    # 1) 유연한 centroid 매칭 우선  
    if have_vecs:
        print("[selector] route=flexible-centroid | centroid 계산 + 유연한 ENV/ACT 매칭")
        try:
            return select_examples_env_act_centroid_with_flexible_matching(
                header_text,
                task_vec_pkl,
                unit_vec_pkl,
                _TASK_JSON_INDEX,
                per_task=2
            )
        except Exception as e:
            print(f"[WARN] 유연한 centroid 방식 실패: {e}")
            # 기존 방식으로 fallback
            try:
                print("[selector] route=fallback-to-original-flexible")
                return select_examples_env_act_centroid_from_vectors_flexible(
                    header_text,
                    task_vec_pkl,
                    unit_vec_pkl,
                    _TASK_JSON_INDEX,
                    per_task=2
                )
            except Exception as e2:
                print(f"[WARN] 기존 flexible vectors-only도 실패: {e2}")

    # 2) 결과 PKL 센트로이드 (기존과 동일)
    if have_results:
        print("[selector] route=results-centroid")
        try:
            return select_examples_env_act_centroid(
                header_text,
                task_res_pkl,
                unit_res_pkl,
                task_vec_pkl if os.path.exists(task_vec_pkl) else None,
                unit_vec_pkl if os.path.exists(unit_vec_pkl) else None,
                _TASK_JSON_INDEX,
                per_task=2
            )
        except Exception as e:
            print(f"[WARN] Results-centroid 실패: {e}")

    # 3) 멀티뷰 
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모든 방법 실패. 누락: {model_path}")
    print("[selector] route=multiview")
    return select_examples_with_multiview(
        header_text, model_path, _TASK_JSON_INDEX, top_total=top_total
    )

def llm_breakdown(command: str) -> str:
    """
    Breaks down a user command for Windows desktop automation.
    - Uses flexible ENV/ACT tags (no fixed whitelist).
    - Opening/switching apps: include explicit Win-search steps WHEN RELEVANT (soft rule).
    - Output must be parseable: Task Unit header + numbered Task# blocks.
    """
    prompt = f"""Your job is to break down a command given by the user for execution on a Windows desktop.
Identify tasks, group them into a Task Unit, and label them. A Task Unit contains a sequence of Tasks to fulfill the command.

Create a structured labeling in the following format:

Task Unit : **ENV[environment/platform-subdomains] ACT[action_category/specific_action] Descriptive Title**

Task#1: ENV[environment/platform-subdomains] ACT[action]
Description: [Brief description of this specific subtask, max 20 words]

Task#2: ENV[environment/platform-subdomains] ACT[action]
Description: [Brief description of this specific subtask, max 20 words]

[List of numbered tasks goes here, keeping original numbering]

Always use "Task Unit : **ENV[] ACT[] Title**" and Task#N blocks exactly, because I will parse your response.

RULES FOR TAGS
1) ENV tag format (examples):
   - local/[program]      → local/FileExplorer, local/Terminal
   - web/[sitename]       → web/Chrome-UniversityPortal, web/Chrome-ShoppingPlatform
   - app/[appname]        → app/Excel, app/VSCode

2) ACT tag format:
   - Use one or more semantically meaningful actions (flexible, not from a fixed list).
   - Examples: ACT[search, filter_sort], ACT[add_to_cart, product_purchase], ACT[open_document], ACT[copy], ACT[paste], ACT[export]

3) Multi-environment Task Units:
   - Compose a sequence of environments that realistically fulfills the goal.
   - Examples:
     • ENV[web/Chrome-AcademicSite] when a user requests a paper summary
     • ENV[local/FileExplorer, app/Excel] for a unit involving file browsing and spreadsheet editing

4) OPENING & SWITCHING GUIDELINE (soft rule, include when relevant):
   - When an application or browser needs to be opened, or when switching between web and app (or app↔app),
     add explicit open/switch tasks that use Windows search:
     • press Win → type target app/site (e.g., "chrome") → press Enter
   - These should appear as separate tasks with ENV=local/Windows (e.g., ACT[open_application] or ACT[switch_application]).
   - Only include these when they are needed for the current command.

5) Descriptions:
   - Each Task description must be ≤ 20 words and clearly state the immediate goal of that step.

User Command:
{command}"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a meticulous task labeler for desktop/web automation. "
                "Always produce strictly structured, parseable outputs. "
                "Include Win-based opening/switching tasks when relevant, but do not over-prescribe."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    return call_chat(messages, temperature=BREAKDOWN_TEMPERATURE, max_tokens=BREAKDOWN_MAX_TOKENS)

def _build_messages_for_plan(
    command: str,
    header_text: str,
    examples_text: str,
    log_catalog_text: str,
    example_traces_text: str,
    seed_plan_text: str = "",
    chunk_requirements: str = "",
    placeholder_map: Dict[str, Dict[str, Any]] = None,
) -> List[Dict[str, str]]:    
    sys_rules = [
        "You are a Desktop GUI automation planner for Windows 10.",
        "Plan GUI control automation subtasks to fulfill the user task.",
        "Allowed Events (18): click/rightclick/drag/scroll/press/text input/open/close/switch focus/go-to/save/copy/paste/delete/rename/login/repeat/wait.",
        "Every plan line MUST be exactly: 'n. b#Event, object' where b ∈ {0,1} is the assist bit.",
        "Assist bit usage: 0 = agent performs; 1 = user must act.",
        "",
        "CRITICAL: You have been provided with CONTEXT LOGS containing real automation steps.",
        "- REUSE these steps directly whenever possible",
        "- Only change the specific values (search terms, file names, etc.) to match the current task",
        "- Keep the same event types and step structure",
        "- Do NOT invent new steps when existing ones can be adapted",
        "- Opening/switching should follow Win-based pattern when relevant, as seen in breakdown & context.",
        "Output ONLY the numbered plan steps. No explanations, no sections, no commentary.",
        "**Each numbered line MUST contain exactly ONE 'B#Event, object' pair. Do NOT combine multiple steps on a single line.**"
    ]

    messages = [
        {"role": "system", "content": "\n".join(sys_rules)},
        {"role": "user", "content": f"TASK BREAKDOWN:\n{header_text.strip()}"},
    ]

    # 핵심: 실제 컨텍스트 로그 추가
    if placeholder_map:
        full_context = _render_full_context_logs(placeholder_map)
        messages.append({"role": "user", "content": f"CONTEXT LOGS:\n{full_context}"})
        
        messages.append({"role": "user", "content": 
            "INSTRUCTIONS:\n"
            "1. Look at the CONTEXT LOGS above - these contain real automation steps\n"
            "2. Reuse these steps as much as possible\n" 
            "3. Adapt only the specific values to match the current task\n"
            "4. For 'search LLM at chatgpt and copy at word':\n"
            "   - Change search terms from the logs to 'LLM at chatgpt'\n"
            "   - Keep the same step structure for opening Chrome, searching, copying, opening Word, pasting\n"
            "5. Output format: just the numbered steps, nothing else"
        })
    else:
        messages.append({"role": "user", "content": "No context logs available. Create a basic plan."})

    if seed_plan_text:
        messages.append({"role": "user", "content": f"SEED PLAN:\n{seed_plan_text.strip()}"})
    
    messages.append({"role": "user", "content": f"CURRENT TASK: {command}"})
    
    return messages

def _llm_assign_steps_to_tasks(breakdown_text: str, final_steps_text: str) -> str:
    """
    LLM을 사용해서 breakdown된 task들과 plan을 의미적으로 매칭하여 포맷팅
    """
    
    prompt = f"""You are a task-step matching assistant. Your job is to assign each numbered step to the most appropriate task based on the breakdown and plan.

TASK BREAKDOWN:
{breakdown_text.strip()}

PLAN STEPS:
{final_steps_text.strip()}

Your job is to:
1. Analyze each step and determine which task it belongs to
2. Output the result in the exact format shown below
3. Each step should be assigned to exactly one task
4. Steps should be assigned based on their content and the task's ENV/ACT

OUTPUT FORMAT (follow exactly):
=== 최종 자동화 계획 ===
MAIN PLAN
{final_steps_text.strip()}

MAIN PLAN
Task#1: ENV[...] ACT[...]
Description: ...
[list the specific step numbers that belong to this task]

Task#2: ENV[...] ACT[...]
Description: ...
[list the specific step numbers that belong to this task]

... (continue for all tasks)

RULES:
- Keep the exact ENV and ACT content from the breakdown
- Keep the exact Description content from the breakdown  
- Only list the step numbers (with their full content) that semantically match each task
- For example, steps with "win" and "chrome" should go to the task about opening Chrome
- Steps with "chatgpt.com" and "CNN" should go to the search task
- Steps with "ctrl+a" and "ctrl+c" should go to the copy task
- Steps with "word" should go to the Word opening task
- Steps with "ctrl+v" should go to the paste task

Generate the formatted output now:"""

    try:
        messages = [
            {"role": "system", "content": "You are a precise task-step matching assistant. Follow the format exactly."},
            {"role": "user", "content": prompt}
        ]
        
        formatted_result = call_chat(messages, temperature=0.0, max_tokens=PLAN_MAX_TOKENS)
        
        if formatted_result and "=== 최종 자동화 계획 ===" in formatted_result:
            return formatted_result
        else:
            # Fallback to simple assignment if LLM fails
            return _simple_assign_steps_to_tasks(breakdown_text, final_steps_text)
            
    except Exception as e:
        print(f"[WARN] LLM step assignment failed: {e}, using fallback")
        return _simple_assign_steps_to_tasks(breakdown_text, final_steps_text)

def _simple_assign_steps_to_tasks(breakdown_text: str, final_steps_text: str) -> str:
    """
    간단한 fallback 방식으로 스텝을 Task에 할당 (기존 균등 분할 방식)
    """
    import re
    
    # Task 정보 파싱
    task_lines = []
    for line in breakdown_text.splitlines():
        line = line.strip()
        if line.startswith("Task#"):
            task_lines.append(line)
    
    # 스텝 파싱
    step_lines = []
    step_pattern = re.compile(r'^\s*(\d+)\.\s*([01]#.*)')
    for line in final_steps_text.splitlines():
        match = step_pattern.match(line.strip())
        if match:
            step_lines.append(line.strip())
    
    if not task_lines or not step_lines:
        return final_steps_text
    
    # 스텝을 Task 수만큼 균등 분할
    total_steps = len(step_lines)
    num_tasks = len(task_lines)
    steps_per_task = max(1, total_steps // num_tasks)
    
    # 포맷팅된 출력 생성
    output_lines = [
        "=== 최종 자동화 계획 ===",
        "MAIN PLAN"
    ]
    
    # 전체 스텝 한번 출력
    output_lines.extend(step_lines)
    output_lines.append("")
    output_lines.append("MAIN PLAN")
    
    # Task별로 스텝 할당
    current_step_idx = 0
    for i, task_line in enumerate(task_lines):
        # Task 헤더 추출
        task_match = re.match(r'(Task#\d+):\s*(ENV\[.*?\])\s*(ACT\[.*?\])', task_line)
        if task_match:
            task_num = task_match.group(1)
            env_part = task_match.group(2)
            act_part = task_match.group(3)
            
            output_lines.append(f"{task_num}: {env_part} {act_part}")
            
            # Description 찾기
            desc_line = None
            task_idx = breakdown_text.find(task_line)
            if task_idx != -1:
                remaining_text = breakdown_text[task_idx + len(task_line):]
                for line in remaining_text.splitlines():
                    line = line.strip()
                    if line.startswith("Description:"):
                        desc_line = line
                        break
                    elif line.startswith("Task#"):
                        break
            
            if desc_line:
                output_lines.append(desc_line)
            
            # 이 Task에 할당될 스텝들
            if i == num_tasks - 1:  # 마지막 Task
                task_steps = step_lines[current_step_idx:]
            else:
                end_idx = min(current_step_idx + steps_per_task, total_steps)
                task_steps = step_lines[current_step_idx:end_idx]
                current_step_idx = end_idx
            
            output_lines.extend(task_steps)
            if i < num_tasks - 1:  # 마지막이 아니면 빈 줄 추가
                output_lines.append("")
    
    return "\n".join(output_lines)

def _auto_assign_steps_to_tasks(breakdown_text: str, final_steps_text: str) -> str:
    """
    브레이크다운된 Task들과 최종 스텝들을 자동으로 매칭해서 포맷팅
    """
    import re
    
    # Task 정보 파싱
    task_lines = []
    for line in breakdown_text.splitlines():
        line = line.strip()
        if line.startswith("Task#"):
            task_lines.append(line)
    
    # 스텝 파싱
    step_lines = []
    step_pattern = re.compile(r'^\s*(\d+)\.\s*([01]#.*)')
    for line in final_steps_text.splitlines():
        match = step_pattern.match(line.strip())
        if match:
            step_lines.append(line.strip())
    
    if not task_lines or not step_lines:
        return final_steps_text
    
    # 스텝을 Task 수만큼 균등 분할
    total_steps = len(step_lines)
    num_tasks = len(task_lines)
    steps_per_task = max(1, total_steps // num_tasks)
    
    # 포맷팅된 출력 생성
    output_lines = [
        "=== 최종 자동화 계획 ===",
        "MAIN PLAN"
    ]
    
    # 전체 스텝 한번 출력
    output_lines.extend(step_lines)
    output_lines.append("")
    output_lines.append("MAIN PLAN")
    
    # Task별로 스텝 할당
    current_step_idx = 0
    for i, task_line in enumerate(task_lines):
        # Task 헤더 추출
        task_match = re.match(r'(Task#\d+):\s*(ENV\[.*?\])\s*(ACT\[.*?\])', task_line)
        if task_match:
            task_num = task_match.group(1)
            env_part = task_match.group(2)
            act_part = task_match.group(3)
            
            output_lines.append(f"{task_num}: {env_part} {act_part}")
            
            # Description 찾기
            desc_line = None
            task_idx = breakdown_text.find(task_line)
            if task_idx != -1:
                remaining_text = breakdown_text[task_idx + len(task_line):]
                for line in remaining_text.splitlines():
                    line = line.strip()
                    if line.startswith("Description:"):
                        desc_line = line
                        break
                    elif line.startswith("Task#"):
                        break
            
            if desc_line:
                output_lines.append(desc_line)
            
            # 이 Task에 할당될 스텝들
            if i == num_tasks - 1:  # 마지막 Task
                task_steps = step_lines[current_step_idx:]
            else:
                end_idx = min(current_step_idx + steps_per_task, total_steps)
                task_steps = step_lines[current_step_idx:end_idx]
                current_step_idx = end_idx
            
            output_lines.extend(task_steps)
            if i < num_tasks - 1:  # 마지막이 아니면 빈 줄 추가
                output_lines.append("")
    
    return "\n".join(output_lines)

# --- object token helpers (add) ---
URL_RX = re.compile(r"(?:https?://)?(?:www\.)?([a-z0-9.-]+\.[a-z]{2,})", re.IGNORECASE)

def _parse_step_triples(plan_text: str):
    triples = []
    pat = re.compile(r"^\s*(\d+)\.\s*[01]#\s*([^,]+)\s*,\s*(.+?)\s*$")
    for ln in plan_text.splitlines():
        m = pat.match(ln)
        if not m:
            continue
        try:
            n = int(m.group(1))
            ev = m.group(2).strip()
            obj = m.group(3).strip()
        except (IndexError, ValueError):
            continue
        triples.append((n, ev, obj))
    return triples

def _auto_generate_chunk_info(plan_text: str, chunk_requirements: str) -> str:
    """플랜에 chunk 정보를 자동으로 추가"""
    if not chunk_requirements:
        return plan_text
    
    # 플랜의 총 스텝 수 계산
    steps = _parse_step_triples(plan_text)
    total_steps = len(steps)
    
    # chunk_requirements 파싱: "1:Log1,Log2;2:Log3,Log4"
    chunks = []
    for part in chunk_requirements.split(";"):
        if ":" in part:
            task_num, tags_str = part.split(":", 1)
            tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()]
            chunks.append((int(task_num), tags))
    
    # 스텝을 chunk별로 균등 분할
    chunk_info_lines = []
    if chunks:
        steps_per_chunk = max(1, total_steps // len(chunks))
        current_step = 1
        
        for i, (task_num, tags) in enumerate(chunks):
            start_step = current_step
            if i == len(chunks) - 1:  # 마지막 청크
                end_step = total_steps
            else:
                end_step = min(current_step + steps_per_chunk - 1, total_steps)
            
            tags_str = ",".join(tags)
            chunk_info_lines.append(f"- CHUNK {task_num}: [{tags_str}] (steps {start_step}-{end_step})")
            current_step = end_step + 1
    
    # 플랜에 chunk 정보 추가
    if chunk_info_lines:
        chunk_info = "\n\n[CHUNK INFO]\n" + "\n".join(chunk_info_lines)
        return plan_text + chunk_info
    
    return plan_text

def plan_with_llm(
    command: str,
    header_text: str,
    examples_text: str,
    log_catalog_text: str,
    example_traces_text: str,
    seed_plan_text: str = "",
    chunk_requirements: str = "",
    placeholder_map: Dict[str, Dict[str, Any]] = None,
) -> Tuple[str, List[Tuple[str,str,str]]]:
    """풀 컨텍스트 전용 plan_with_llm + chunk info 자동 생성"""
    
    messages = _build_messages_for_plan(
        command=command,
        header_text=header_text,
        examples_text=examples_text,
        log_catalog_text=log_catalog_text,
        example_traces_text=example_traces_text,
        seed_plan_text=seed_plan_text,
        chunk_requirements=chunk_requirements,
        placeholder_map=placeholder_map,
    )
    
    # Chunk 요구사항이 있으면 추가 지시사항 제공
    if chunk_requirements:
        chunk_instruction = f"""
CHUNK REQUIREMENTS:
{chunk_requirements}

Each task should use the corresponding logs. After generating the plan, add chunk information at the end in this format:

[CHUNK INFO]
- CHUNK 1: [Log1,Log2] (steps 1-X)
- CHUNK 2: [Log3,Log4] (steps X+1-Y)
- CHUNK 3: [Log5,Log6] (steps Y+1-Z)

Adjust the step ranges based on your actual plan."""
        
        messages.append({"role": "user", "content": chunk_instruction})
    
    plan_text = call_chat(messages, temperature=PLAN_TEMPERATURE, max_tokens=PLAN_MAX_TOKENS)
    if not plan_text:
        raise RuntimeError("LLM이 빈 플랜을 반환했습니다.")

    # Chunk 정보가 없으면 자동 생성
    if "[CHUNK INFO]" not in plan_text and chunk_requirements:
        plan_text = _auto_generate_chunk_info(plan_text, chunk_requirements)

    try:
        steps = _parse_step_triples(plan_text)
    except Exception as e:
        preview = "\n".join(plan_text.splitlines()[:30])
        raise RuntimeError(f"_parse_step_triples 실패({e}). 미리보기:\n{preview}") from e

    if not steps:
        preview = "\n".join(plan_text.splitlines()[:30])
        raise RuntimeError(f"LLM 플랜에서 유효한 0#/1# 스텝을 파싱하지 못했습니다. 미리보기:\n{preview}")

    return plan_text, steps

def parse_chunk_info(plan_text: str):
    out = []
    s = plan_text.find("[CHUNK INFO]")
    if s == -1:
        return out
    for ln in plan_text[s:].splitlines():
        m = _CHUNK_LINE_RE.match(ln.strip())
        if not m:
            continue
        out.append({
            "chunk_num": int(m.group(1)),
            "tags": [t.strip() for t in m.group(2).split(",") if t.strip()],
            "start": int(m.group(3)),
            "end": int(m.group(4)),
        })
    return out

def print_chunk_ranges(plan_text: str) -> None:
    all_plan_lines = []
    pat_line = re.compile(r"^\s*(\d+)\.\s*(.*)$")
    for ln in plan_text.splitlines():
        m = pat_line.match(ln)
        if m:
            all_plan_lines.append((int(m.group(1)), ln.rstrip()))

    ci = parse_chunk_info(plan_text)
    if not ci:
        return

    print("\n=== Chunk Ranges (실제 플랜 라인) ===")
    for ch in ci:
        a, b = ch["start"], ch["end"]
        print(f"\n[CHUNK {ch['chunk_num']}] steps {a}-{b}")
        for n, raw in all_plan_lines:
            if a <= n <= b:
                print("  " + raw)

def _extract_step_lines_for_prompt(unit: dict, max_lines: int = 14) -> str:
    """해당 유닛의 스텝을 모두 반환 (부트스트랩 포함, 줄수 제한 없음)"""
    raw_steps = _split_steps_field(unit)
    if raw_steps:
        return "\n".join(f"{i}. {s}" for i, s in enumerate(raw_steps, 1))

    # fallback: 본문에서 0#/1# 패턴 전부 수집
    block = (unit.get("content") or unit.get("des_text") or "").splitlines()
    out_lines = []
    pat = re.compile(r"^\s*\d+\.\s*[01]#\s*[^,]+,\s*.+$")
    for ln in block:
        s = ln.strip()
        if pat.match(s):
            out_lines.append(s)
    return "\n".join(out_lines)

def _compose_example_traces(placeholder_map: Dict[str, Dict[str, Any]], limit: int = 4) -> str:
    """Task Unit과 개별 Task의 서브태스크 시퀀스 조합"""
    parts = []
    for tag, item in list(placeholder_map.items())[:limit]:
        env_s = ", ".join(item.get("env_tag", []) or [])
        act_s = item.get("act_tag", "") or ""
        ttl = item.get("title", "") or ""
        data_type = item.get("data_type", "unknown")
        
        steps = _extract_step_lines_for_prompt(item, max_lines=14)
        if not steps:
            continue
            
        # 데이터 타입에 따라 헤더 구분
        type_indicator = "「TASK UNIT」" if data_type == "task_unit" else "「TASK」"
        header = f"### {tag} {type_indicator} ENV[{env_s}] ACT[{act_s}] Title[{ttl}]"
        
        parts.append(f"{header}\n{steps}")
    
    return "\n\n".join(parts) if parts else ""

def map_tags_to_logs(chunk_lines, placeholder_map=None):
    out = []
    for ch in chunk_lines:
        refs = []
        for tag in ch.get("tags", []):
            unit = (placeholder_map or {}).get(tag, {}) if placeholder_map else {}
            if unit:
                preview = _extract_step_lines_for_prompt(unit, max_lines=8) or ""
            else:
                preview = ""
            refs.append({
                "tag": tag,
                "unique_id": unit.get("unique_id"),
                "file_name": unit.get("file_name"),
                "data_type": unit.get("data_type"),
                "env_tag": unit.get("env_tag"),
                "act_tag": unit.get("act_tag"),
                "preview_steps": preview,
            })
        ch2 = dict(ch)
        ch2["referenced_logs"] = refs
        ch2.pop("tags", None)
        out.append(ch2)
    return out

def complete_plan_with_registry(
    breakdown_text: str, 
    base_plan_text: str, 
    fill: Dict[str, str] = None, 
    coverage_tau: float = 0.78
) -> Tuple[str, Dict]:
    """플랜의 누락 ACT를 레지스트리에서 보충"""
    try:
        from plan_gap_filler import complete_plan_with_registry as _complete, summarize_gap
        return _complete(breakdown_text, base_plan_text, fill=fill, coverage_tau=coverage_tau)
    except ImportError:
        # 폴백: 기본 보충 로직
        try:
            from act_registry import resolve_acts_in_plan_text, acts_from_text
            
            required = (canon_set(resolve_acts_in_plan_text(breakdown_text or "")) 
                        | canon_set(acts_from_text(breakdown_text or "")))
            present  = (canon_set(resolve_acts_in_plan_text(base_plan_text or "")) 
                        | canon_set(acts_from_text(base_plan_text or "")))
            
            missing = [a for a in required if not has_alias_in_set(a, present)]
            injected = []
            
            for aid in missing:
                for cand in alias_candidates(aid):
                    try:
                        sk = synthesize_chunk_from_registry(cand, fill=fill) or []
                    except Exception:
                        sk = []
                    if sk:
                        for step in sk:
                            if isinstance(step, (list, tuple)) and len(step) >= 3:
                                injected.append(f"0#{step[1]}, {step[2]}")
                        break
            
            if injected:
                final_plan = _inject_numbered_steps(base_plan_text, injected)
            else:
                final_plan = base_plan_text
                
            gap_summary = {
                "required": list(required),
                "present": list(present), 
                "missing": missing,
                "injected_count": len(injected)
            }
            
            return final_plan, gap_summary
            
        except Exception as e:
            print(f"플랜 보충 실패: {e}")
            return base_plan_text, {"error": str(e)}

def summarize_gap(gap_summary: Dict) -> str:
    """gap 요약 출력 (plan_gap_filler의 summarize_gap 폴백)"""
    try:
        from plan_gap_filler import summarize_gap as _summarize
        return _summarize(gap_summary)
    except ImportError:
        missing = gap_summary.get("missing", [])
        covered = gap_summary.get("covered", [])
        if isinstance(missing, set):
            missing = list(missing)
        if isinstance(covered, set):
            covered = list(covered)
        return f"missing={missing} covered={covered}"
    

####################################################
# 걍 실행할라고 추가한 것들~~~~~
##############################################################

# ---- main.py 하위호환용 래퍼 (리스트가 아니라 PlanBundle을 반환) ----
def process_command(command: str) -> str:
    # 필요시 전처리(정규화 등) — 현재는 그대로 반환
    return command

#----------------------------------------끄읕-----------------

def global_planner(command: str) -> dict:
    """
    통합 RAG 기반 플래너.
    - 최종 출력(LLM 포맷 문자열)을 PlanBundle(dict)로 파싱해 반환.
    - PlanBundle 스키마:
        {
          "steps": [ {"id": int, "assist": 0/1, "event": str, "object": str, "task_id": Optional[str]}, ... ],
          "tasks": { "Task#1": {"env":[...], "act":[...], "description": str, "step_ids":[...]}, ... },
          "task_order": ["Task#1", "Task#2", ...]
        }
    """
    import re

    # ---------- 내부 헬퍼(로컬) : 포맷 문자열 → PlanBundle ----------
    STEP_LINE = re.compile(r'^\s*(\d+)\.\s*([01])#\s*([^,]+)\s*,\s*(.+?)\s*$')
    TASK_HEADER = re.compile(r'^(Task#\d+):\s*ENV\[(.*?)\]\s*ACT\[(.*?)\]\s*$', re.IGNORECASE)

    def _parse_main_steps_block(lines, start_idx):
        steps = []
        i = start_idx
        while i < len(lines):
            ln = lines[i].rstrip()
            if ln.strip() == "MAIN PLAN" and i > start_idx:
                break
            m = STEP_LINE.match(ln)
            if m:
                nid = int(m.group(1))
                assist = int(m.group(2))
                event = m.group(3).strip()
                obj = m.group(4).strip()
                steps.append({"id": nid, "assist": assist, "event": event, "object": obj})
            i += 1
        return steps, i  # i == 두 번째 MAIN PLAN 위치(또는 EOF)

    def _parse_task_block(lines, start_idx):
        """Task#N 블록 하나 파싱."""
        m = TASK_HEADER.match(lines[start_idx].strip())
        if not m:
            return None, start_idx + 1

        task_id = m.group(1).strip()
        env_raw = [s.strip() for s in (m.group(2) or "").split(",") if s.strip()]
        act_raw = [s.strip() for s in (m.group(3) or "").split(",") if s.strip()]

        desc = ""
        step_ids = []
        i = start_idx + 1

        # 보조: "1, 3-5" 숫자/범위에서 id 수집
        def _collect_ids_from_text(text: str):
            for tok in re.findall(r'\d+(?:\s*-\s*\d+)?', text):
                if "-" in tok:
                    a, b = [int(x) for x in re.split(r'\s*-\s*', tok)]
                    for k in range(min(a, b), max(a, b) + 1):
                        step_ids.append(k)
                else:
                    step_ids.append(int(tok))

        while i < len(lines):
            ln = lines[i].rstrip()
            if not ln.strip():
                i += 1
                continue
            if TASK_HEADER.match(ln):  # 다음 Task 시작
                break
            if ln.strip().lower().startswith("description:"):
                desc = ln.split(":", 1)[1].strip()
            m2 = STEP_LINE.match(ln)
            if m2:
                step_ids.append(int(m2.group(1)))
            else:
                # 숫자만 적은 라인(범위 포함) 대응
                _collect_ids_from_text(ln)
            i += 1

        step_ids = sorted(set(step_ids))
        return {
            "task_id": task_id,
            "env": env_raw,
            "act": act_raw,
            "description": desc,
            "step_ids": step_ids
        }, i

    def _parse_formatted_to_planbundle(formatted: str) -> dict:
        lines = [ln.rstrip("\n") for ln in (formatted or "").splitlines()]
        # 첫 MAIN PLAN 탐색
        main_positions = [i for i, ln in enumerate(lines) if ln.strip() == "MAIN PLAN"]
        if not main_positions:
            # MAIN PLAN이 없으면 최소 스텝만 파싱 시도
            steps = []
            for ln in lines:
                m = STEP_LINE.match(ln.strip())
                if m:
                    steps.append({
                        "id": int(m.group(1)),
                        "assist": int(m.group(2)),
                        "event": m.group(3).strip(),
                        "object": m.group(4).strip(),
                        "task_id": None
                    })
            return {"steps": steps, "tasks": {}, "task_order": []}

        first_main = main_positions[0]
        steps, second_main_idx = _parse_main_steps_block(lines, first_main + 1)

        tasks = {}
        task_order = []
        i = second_main_idx + 1 if second_main_idx < len(lines) else len(lines)
        while i < len(lines):
            ln = lines[i].strip()
            if not ln:
                i += 1
                continue
            if TASK_HEADER.match(ln):
                tinfo, j = _parse_task_block(lines, i)
                if tinfo:
                    tasks[tinfo["task_id"]] = {
                        "env": tinfo["env"],
                        "act": tinfo["act"],
                        "description": tinfo["description"],
                        "step_ids": tinfo["step_ids"]
                    }
                    task_order.append(tinfo["task_id"])
                    i = j
                    continue
            i += 1

        # step → task 매핑
        owner = {}
        for tid, tinfo in tasks.items():
            for sid in tinfo.get("step_ids", []):
                owner[sid] = tid
        for s in steps:
            s["task_id"] = owner.get(s["id"])

        return {"steps": steps, "tasks": tasks, "task_order": task_order}

    # ---------- 원래 로직 시작 ----------
    print_info("통합 RAG 시스템으로 동작 - Task Unit + 개별 Task 모두 활용")
    if OPENAI_OK:
        print_info("API 키가 성공적으로 설정되었습니다.")
    else:
        print_info(f"OpenAI 초기화 실패: {OPENAI_ERR or '원인 불명'} (LLM 호출 불가)")
        # 빈 번들 반환 (호출부의 예외처리 편의용)
        return {"steps": [], "tasks": {}, "task_order": []}

    print(f"Command: {command}")

    # 1) Breakdown
    print_info("GPT API 호출 중... (Task Breakdown)")
    try:
        breakdown = llm_breakdown(command)
    except Exception as e:
        print(f"Breakdown 실패: {e}")
        return {"steps": [], "tasks": {}, "task_order": []}

    # 2) LongTermPlanCache 우선
    print("검색: 캐시에서 유사한 플랜 검색 중...")
    try:
        cache_result = try_cache_first_from_breakdown(command, breakdown)
    except Exception as _e:
        cache_result = None
        print(f"캐시 조회 실패(무시): {_e}")

    final_plan_text = None
    placeholder_map = None  # 디버깅 출력에 사용

    if cache_result:
        header, plan, used_cache_id = cache_result
        # 캐시 플랜 보강
        final_plan_text, gap = complete_plan_with_registry(
            breakdown_text=breakdown,
            base_plan_text=plan,
            fill={"query": command},
            coverage_tau=0.78,
        )
        print("[GAP]", gap.get("missing", []))

        # 시드 리플랜으로 정합화
        try:
            replanned_text, _ = plan_with_llm(
                command=command,
                header_text=breakdown,
                examples_text="",
                log_catalog_text="",
                example_traces_text="",
                seed_plan_text=final_plan_text,
            )
            final_plan_text = replanned_text or final_plan_text
        except Exception as e:
            print(f"[WARN] Seed-patch replan failed: {e}")

        if final_plan_text:
            # 스텝만 발췌 → LLM 매칭 포맷팅
            step_lines = []
            pat_line = re.compile(r'^\s*\d+\.\s*[01]#')
            for ln in (final_plan_text or "").splitlines():
                if pat_line.match(ln.strip()):
                    step_lines.append(ln.strip())
            steps_only_text = "\n".join(step_lines)

            formatted_output = _llm_assign_steps_to_tasks(breakdown, steps_only_text)
            print(formatted_output)

            # >>> 반환용 PlanBundle
            return _parse_formatted_to_planbundle(formatted_output)

        # 어떤 이유로도 플랜이 없으면 빈 번들
        return {"steps": [], "tasks": {}, "task_order": []}

    # 2-B) Feedback 캐시
    fb_plan = try_feedback_cache(command)
    if fb_plan:
        final_plan_text, gap = complete_plan_with_registry(
            breakdown_text=breakdown,
            base_plan_text=fb_plan,
            fill={"query": "winword"} if "word" in command.lower() else {"query": command},
            coverage_tau=0.70,
        )
        if final_plan_text:
            step_lines = []
            pat_line = re.compile(r'^\s*\d+\.\s*[01]#')
            for ln in (final_plan_text or "").splitlines():
                if pat_line.match(ln.strip()):
                    step_lines.append(ln.strip())
            steps_only_text = "\n".join(step_lines)

            formatted_output = _llm_assign_steps_to_tasks(breakdown, steps_only_text)
            print(formatted_output)
            return _parse_formatted_to_planbundle(formatted_output)

        return {"steps": [], "tasks": {}, "task_order": []}

    # 3) 통합 RAG 예시 선택
    print("\n" + breakdown.strip())
    print("Task Unit + 개별 Task 통합 RAG 검색 중...")
    try:
        _initialize_json_index()
        placeholder_map, log_catalog_text, examples_text, chunk_requirements = unified_rag_select_examples(
            command, breakdown, top_total=6
        )
    except Exception as e:
        print(f"통합 RAG 검색 실패: {e}")
        import traceback
        traceback.print_exc()
        return {"steps": [], "tasks": {}, "task_order": []}

    print(f"선택된 예시: {len(placeholder_map)}개")
    type_counts = {}
    for item in placeholder_map.values():
        data_type = item.get("data_type", "unknown")
        type_counts[data_type] = type_counts.get(data_type, 0) + 1
    if type_counts:
        print("데이터 타입 분포:")
        for k, v in type_counts.items():
            print(f"  - {k}: {v}")

    # 디버깅 출력
    if placeholder_map:
        print(f"\n[DEBUG] 선택된 로그들:")
        for tag, ex in placeholder_map.items():
            steps_dbg = _extract_steps_verbatim(ex)
            print(f"  {tag}: {ex.get('file_name', 'unknown')} - {len(steps_dbg)} steps")
            if steps_dbg:
                print(f"    샘플: {steps_dbg[0][:50]}...")

    # 4) 플랜 생성
    print_info("GPT API 호출 중... (Plan Generation)")
    try:
        plan_text, triples = plan_with_llm(
            command=command,
            header_text=breakdown,
            examples_text="",
            log_catalog_text="",
            example_traces_text="",
            seed_plan_text="",
            chunk_requirements="",
            placeholder_map=placeholder_map,
        )
        final_plan_text = plan_text
    except Exception as e:
        print(f"Plan 생성 실패: {e}")
        return {"steps": [], "tasks": {}, "task_order": []}

    # 5) 포매팅 + PlanBundle 변환
    try:
        step_lines = []
        pat_line = re.compile(r'^\s*\d+\.\s*[01]#')
        for ln in (final_plan_text or "").splitlines():
            if pat_line.match(ln.strip()):
                step_lines.append(ln.strip())
        steps_only_text = "\n".join(step_lines)

        formatted_output = _llm_assign_steps_to_tasks(breakdown, steps_only_text)
        print(formatted_output)
        return _parse_formatted_to_planbundle(formatted_output)

    except Exception as e:
        print(f"[WARN] formatter failed ({e}); falling back to MAIN PLAN only.")
        print("\n=== 최종 자동화 계획 ===")
        print("MAIN PLAN")
        # 폴백: MAIN PLAN 라인만으로 최소 PlanBundle 생성
        minimal_lines = []
        for ln in (final_plan_text or "").splitlines():
            ln = ln.strip()
            if ln and re.match(r'^\s*\d+\.\s*[01]#', ln):
                print(ln)
                minimal_lines.append(ln)

        formatted_fallback = "MAIN PLAN\n" + "\n".join(minimal_lines)
        return _parse_formatted_to_planbundle(formatted_fallback)

###############
#------------------------글로벌플래너 끄ㅌ---------------------^^
###############

def debug_selected_logs(placeholder_map):
    """선택된 로그들이 실제 스텝을 가지고 있는지 확인"""
    print(f"\n=== 선택된 로그 디버깅 ===")
    for tag, ex in placeholder_map.items():
        print(f"\n{tag}:")
        print(f"  - file_name: {ex.get('file_name', 'None')}")
        print(f"  - title: {ex.get('title', 'None')}")
        print(f"  - env_tag: {ex.get('env_tag', 'None')}")
        print(f"  - act_tag: {ex.get('act_tag', 'None')}")
        
        steps = _extract_steps_verbatim(ex)
        print(f"  - extracted steps: {len(steps)}")
        for i, step in enumerate(steps[:3]):  # 처음 3개만 출력
            print(f"    [{i+1}] {step}")
        if len(steps) > 3:
            print(f"    ... and {len(steps)-3} more steps")

# 엔트리 포인트
if __name__ == "__main__":
    try:
        cmd = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else input("Enter command: ").strip()
    except EOFError:
        cmd = ""
    if not cmd:
        print("No command provided.")
        sys.exit(1)
    global_planner(cmd)