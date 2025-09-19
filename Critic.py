# -*- coding: utf-8 -*-
"""
Critic.py (patched)

Purpose
-------
- Judge the outcome of the last executed substep using UI signals and execution logs
- Decide one of: success | failure | retryable
- If retryable, propose a *replacement sub-plan* that will replace [start,end] indices in the current plan
  (GlobalPlanner-level granularity for substeps after the failure point)

Key updates
-----------
1) GUI metadata aware:
   - Accepts gui_meta_data passed from UI_control.py (which is produced by GUI_Parser.py)
   - Safely handles missing metadata with fallbacks
   - Injects high-signal fields into the prompt (window title, component counts, parsing method, etc.)

2) Input validation & safe defaults:
   - None / missing fields → sensible defaults
   - Bounded replacements & schema validation for LLM JSON

3) Prompting refinements:
   - Priority-ordered UI signals (Window/URL, Key Components, Count drift, Error Dialogs, Exceptions)
   - Step-by-step reasoning scaffold (task goal, current state, trace review, signals, final judgment)
   - Context compression guidelines (CRITICAL / CONTEXTUAL / NOISE)

Return schema (JSON)
--------------------
{
  "status": "success" | "failure" | "retryable",
  "reason": "<short human-readable rationale>",
  "confidence": float in [0,1],
  "plan_bundle_patch": {
      "replace_range": [start_index, end_index],
      "substeps": [
          {
             "index": int,                # new substep index; local to replace_range
             "env": str,                  # optional: environment constraints
             "act": str,                  # required: high-level action
             "description": str,          # optional: user-friendly description
             "arguments": dict | list | str,  # optional
             "rationale": str             # optional
          },
          ...
      ]
  } | null,
  "diagnostics": {
      "signals": {
          "active_window_title_changed": bool | null,
          "component_count_anomaly": bool | null,
          "error_dialog_detected": bool | null,
          "exception_in_logs": bool | null
      },
      "used_metadata": { ... },           # normalized GUI meta actually used
      "notes": str                        # optional extra notes for debugging
  }
}

Compatibility notes
-------------------
- If you already consume 'status' + (optional) 'plan_bundle_patch', this remains compatible.
- 'plan_bundle_patch' is only present for 'retryable'.
- 'replace_range' echoes the input replace_range (clamped to safe bounds).
- 'confidence' can be used to gate automatic apply vs. human review.

Examples
--------
# 1) Retryable with one-step patch
{
  "status": "retryable",
  "reason": "Menu failed to open due to stale click; propose alternative path via Alt+F.",
  "confidence": 0.72,
  "plan_bundle_patch": {
      "replace_range": [5,5],
      "substeps": [
          {
              "index": 5,
              "act": "key",
              "description": "Open the browser menu via keyboard",
              "arguments": {"keys": "Alt+F"},
              "rationale": "Keyboard menu open is robust to UI layout drift."
          }
      ]
  },
  "diagnostics": {...}
}

# 2) Retryable with three-step patch
{
  "status": "retryable",
  "reason": "Direct click missed due to DOM churn; propose robust path.",
  "confidence": 0.68,
  "plan_bundle_patch": {
      "replace_range": [7,8],
      "substeps": [
          {
              "index": 7,
              "act": "key",
              "description": "Open menu",
              "arguments": {"keys": "Alt+F"}
          },
          {
              "index": 8,
              "act": "type",
              "description": "Search menu for 'Settings'",
              "arguments": {"text": "settings"}
          },
          {
              "index": 9,
              "act": "enter",
              "description": "Confirm navigation to Settings"
          }
      ]
  },
  "diagnostics": {...}
}
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    # Use the same OpenAI API pattern as LocalPlanner.py (client.chat.completions.create)
    from openai import OpenAI  # openai>=1.0 style
except Exception:
    OpenAI = None


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _clamp_range(start: int, end: int, total: int) -> Tuple[int, int]:
    if total <= 0:
        return (max(0, start), max(0, end))
    start = max(0, min(start, total - 1))
    end = max(0, min(end, total - 1))
    if end < start:
        end = start
    return (start, end)


def _normalize_gui_meta(gui_meta_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize GUI meta with safe defaults."""
    meta = gui_meta_data or {}
    active_window_title = meta.get("active_window_title") or "Unknown"
    visible_count = _safe_int(meta.get("visible_component_count"), 0)
    hidden_count = _safe_int(meta.get("hidden_component_count"), 0)
    total_count = _safe_int(meta.get("total_component_count"), visible_count + hidden_count)
    all_windows = meta.get("all_windows") or []
    if isinstance(all_windows, list):
        all_windows = all_windows[:10]
    else:
        all_windows = []
    parsing_method = meta.get("parsing_method") or "unknown"
    timestamp = meta.get("timestamp") or "unknown"

    return {
        "active_window_title": active_window_title,
        "visible_component_count": visible_count,
        "hidden_component_count": hidden_count,
        "total_component_count": total_count,
        "all_windows_count": _safe_int(meta.get("all_windows_count"), len(all_windows)),
        "all_windows": all_windows,
        "parsing_method": parsing_method,
        "timestamp": timestamp,
    }


def _extract_high_signal_flags(
    gui_meta: Dict[str, Any],
    execution_logs: List[str],
    prev_window_title: Optional[str] = None,
) -> Dict[str, Optional[bool]]:
    """Derive simple boolean/tri-state signals the LLM can also read."""
    active_title = gui_meta.get("active_window_title", "Unknown")
    total_count = gui_meta.get("total_component_count", 0)
    parsing_method = gui_meta.get("parsing_method", "unknown")

    # Heuristic thresholds for anomalies (tunable)
    component_count_anomaly = None
    if isinstance(total_count, int) and total_count >= 10000:
        component_count_anomaly = True
    elif isinstance(total_count, int) and total_count <= 0:
        component_count_anomaly = True
    else:
        component_count_anomaly = False

    # Window title changes are strong signals of navigation progress/regression
    title_changed = None
    if prev_window_title is not None:
        title_changed = (active_title != prev_window_title)
    # else keep None to avoid misleading triage

    # Error dialog or exception clues in execution logs
    logs_join = "\n".join(execution_logs or [])
    error_dialog_detected = True if ("alert" in logs_join.lower() or "dialog" in logs_join.lower()) else None
    exception_in_logs = True if ("traceback" in logs_join.lower() or "exception" in logs_join.lower() or "timeout" in logs_join.lower()) else None

    return {
        "active_window_title_changed": title_changed,
        "component_count_anomaly": component_count_anomaly,
        "error_dialog_detected": error_dialog_detected,
        "exception_in_logs": exception_in_logs,
        "parsing_method_unknown": (parsing_method == "unknown"),
    }


class Critic:
    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.1,
        client: Optional[Any] = None,
    ) -> None:
        """
        Initialize Critic.

        - If `client` is None, tries to build OpenAI() using env key.
        - Uses chat.completions.create with response forced to JSON schema (by instruction).
        """
        self.model = model
        self.temperature = temperature

        if client is not None:
            self.client = client
        else:
            if OpenAI is None:
                raise RuntimeError("OpenAI client is not available. Please install openai>=1.0 and set OPENAI_API_KEY.")
            self.client = OpenAI()

    # ---------------------
    # Public API
    # ---------------------
    def judge_and_replan(
        self,
        fail_event: Dict[str, Any],
        plan_bundle: Dict[str, Any],
        task_context: Dict[str, Any],
        failed_substep_index: int,
        replace_range: Tuple[int, int],
    ) -> Dict[str, Any]:
        """
        Decide success/failure/retryable and optionally propose a replacement sub-plan.

        Parameters
        ----------
        fail_event : dict
            Recent failure event info (e.g., action attempted, message, timestamp, etc.)
            May contain 'gui_meta_data' (but UI_control.py already passes it in task_context).

        plan_bundle : dict
            { "main_plan": [ Task# objects as dicts ], "total_substeps": int }

        task_context : dict
            {
              "tid": str|int,
              "env": str,
              "act": str,
              "description": str,
              "execution_logs": List[str],
              "gui_meta_data": Dict[str,Any]  # added
            }

        failed_substep_index : int
            The substep index at which failure occurred (0-based in the current plan execution).

        replace_range : (int,int)
            The [start,end] indices (inclusive) of substeps to replace IF 'retryable' is decided.

        Returns
        -------
        decision : dict
            See module docstring "Return schema (JSON)".
        """
        # --- Normalize & guard inputs
        exec_logs = task_context.get("execution_logs") or []
        gui_meta = _normalize_gui_meta(task_context.get("gui_meta_data"))
        total_substeps = _safe_int(plan_bundle.get("total_substeps"), 0)
        start, end = _clamp_range(
            _safe_int(replace_range[0], failed_substep_index),
            _safe_int(replace_range[1], failed_substep_index),
            total_substeps if total_substeps > 0 else max(failed_substep_index + 1, 1),
        )

        # High-signal flags for diagnostics (and to cue the model)
        prev_title = fail_event.get("prev_active_window_title") if isinstance(fail_event, dict) else None
        signal_flags = _extract_high_signal_flags(gui_meta, exec_logs, prev_window_title=prev_title)

        # Derive a compact, high-value context slice
        high_value_context = self._build_high_value_context(
            task_context=task_context,
            gui_meta=gui_meta,
            fail_event=fail_event,
            failed_substep_index=failed_substep_index,
            total_substeps=total_substeps,
            signal_flags=signal_flags,
        )

        # Build messages (English prompt as requested)
        messages = self._build_messages(
            high_value_context=high_value_context,
            start=start,
            end=end,
        )

        # Call LLM with strong JSON-only instruction
        model_json = self._call_llm_json(messages)

        # Validate & coerce output to the contract
        decision = self._coerce_decision(model_json, start, end)

        # Attach diagnostics for debugging/telemetry
        decision.setdefault("diagnostics", {})
        decision["diagnostics"]["signals"] = signal_flags
        decision["diagnostics"]["used_metadata"] = gui_meta

        return decision

    # ---------------------
    # Internals
    # ---------------------
    def _build_high_value_context(
        self,
        task_context: Dict[str, Any],
        gui_meta: Dict[str, Any],
        fail_event: Dict[str, Any],
        failed_substep_index: int,
        total_substeps: int,
        signal_flags: Dict[str, Optional[bool]],
    ) -> Dict[str, Any]:
        # CRITICAL INFO
        critical = {
            "task_goal": task_context.get("description") or f"{task_context.get('env','')}::{task_context.get('act','')}",
            "current_window_title": gui_meta.get("active_window_title"),
            "top_actionable_components": "(omitted: not provided here)",  # leave summary-level slot open
            "last_exception": self._last_exception_snippet(task_context.get("execution_logs") or []),
        }

        # CONTEXTUAL INFO
        contextual = {
            "component_count_total": gui_meta.get("total_component_count"),
            "visible_component_count": gui_meta.get("visible_component_count"),
            "hidden_component_count": gui_meta.get("hidden_component_count"),
            "recent_3_steps": self._tail(task_context.get("execution_logs") or [], k=3),
            "retry_count": self._extract_retry_count(task_context.get("execution_logs") or []),
            "parsing_method": gui_meta.get("parsing_method"),
            "timestamp": gui_meta.get("timestamp"),
        }

        # Failure locus
        locus = {
            "failed_substep_index": failed_substep_index,
            "total_substeps": total_substeps,
            "fail_event": fail_event or {},
        }

        return {
            "critical": critical,
            "contextual": contextual,
            "signals": signal_flags,
            "all_windows": gui_meta.get("all_windows") or [],
        }

    def _build_messages(self, high_value_context: Dict[str, Any], start: int, end: int) -> List[Dict[str, str]]:
        """
        Build English prompts with:
        - Stepwise reasoning scaffold
        - Priority-ordered signal guidance
        - Context compression (CRITICAL/CONTEXTUAL/NOISE)
        - Strict JSON output instruction
        """
        system_msg = {
            "role": "system",
            "content": (
                "You are a strict desktop UI execution critic and micro-planner for a Windows 10 GUI automation agent. "
                "Your job is to: (1) judge the outcome of the latest executed substep using robust UI signals, and "
                "(2) if retryable, propose a replacement sub-plan to splice into the main plan between the provided indices. "
                "Always respond with a SINGLE VALID JSON OBJECT that matches the required schema. Do NOT include code fences."
            ),
        }

        user_payload = {
            "CRITICAL_INFO": high_value_context.get("critical"),
            "CONTEXTUAL_INFO": high_value_context.get("contextual"),
            "UI_SIGNALS_PRIORITY": [
                "PRIORITY 1: Window Title & URL changes (most reliable).",
                "PRIORITY 2: Presence/absence of key actionable components (Buttons/Links/Forms).",
                "PRIORITY 3: Sudden component-count drift (may indicate parser error or DOM churn).",
                "PRIORITY 4: Error/Alert dialogs visible.",
                "PRIORITY 5: Exceptions/Timeouts in execution logs.",
                "If lower-priority info conflicts with higher-priority, ignore the lower-priority info."
            ],
            "COMPONENT_NOISE_POLICY": (
                "Ignore long raw lists (>500 items). Use only summary stats and high-signal cues."
            ),
            "ALL_WINDOWS_SAMPLE": high_value_context.get("all_windows")[:10],
            "REPLACE_RANGE_HINT": [start, end],
            "REQUIRED_OUTPUT_JSON_SCHEMA": {
                "status": "success | failure | retryable",
                "reason": "short string",
                "confidence": "float in [0,1]",
                "plan_bundle_patch": {
                    "replace_range": [start, end],
                    "substeps": [
                        {
                            "index": "int (local index within replace_range)",
                            "env": "optional string",
                            "act": "required string",
                            "description": "optional string",
                            "arguments": "optional dict/list/string",
                            "rationale": "optional string"
                        }
                    ]
                },
                "diagnostics": {
                    "signals": "object",
                    "used_metadata": "object",
                    "notes": "optional string"
                }
            },
            "REASONING_STEPS": [
                "1) TASK GOAL ANALYSIS: derive the end-goal of the task.",
                "2) CURRENT STATE EVALUATION: judge how close the GUI state is to the goal.",
                "3) EXECUTION TRACE REVIEW: assess whether the recent steps were moving in the right direction.",
                "4) SUCCESS/FAILURE SIGNALS: cite concrete signals per the priority order.",
                "5) FINAL JUDGMENT: decide success/failure/retryable. If retryable, provide *actionable* replacement substeps.",
            ]
        }

        user_msg = {
            "role": "user",
            "content": json.dumps(user_payload, ensure_ascii=False),
        }

        return [system_msg, user_msg]

    def _call_llm_json(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Call the LLM and try to parse a single JSON object."""
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=messages,
            response_format={"type": "json_object"},  # enforce JSON if supported
        )
        content = (resp.choices[0].message.content or "").strip()
        # Fallback: in case the model ignored response_format (rare), try to load anyway
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        # If all else fails, return a conservative failure
        return {
            "status": "failure",
            "reason": "Model did not return valid JSON.",
            "confidence": 0.3,
            "plan_bundle_patch": None,
            "diagnostics": {"notes": content[:500]},
        }

    def _coerce_decision(self, data: Dict[str, Any], start: int, end: int) -> Dict[str, Any]:
        """Validate schema and coerce to safe defaults."""
        status = str(data.get("status") or "failure").lower()
        if status not in ("success", "failure", "retryable"):
            status = "failure"

        reason = data.get("reason") or ""
        try:
            confidence = float(data.get("confidence", 0.5))
            if not (0.0 <= confidence <= 1.0):
                confidence = 0.5
        except Exception:
            confidence = 0.5

        patch = data.get("plan_bundle_patch")
        if status == "retryable" and isinstance(patch, dict):
            rr = patch.get("replace_range") or [start, end]
            if not (isinstance(rr, list) and len(rr) == 2):
                rr = [start, end]
            r0, r1 = _safe_int(rr[0], start), _safe_int(rr[1], end)
            substeps = patch.get("substeps") or []
            # normalize substeps
            norm_substeps = []
            for i, s in enumerate(substeps):
                if not isinstance(s, dict):
                    continue
                act = s.get("act")
                if not act or not isinstance(act, str):
                    continue
                norm_substeps.append({
                    "index": _safe_int(s.get("index"), start + i),
                    "env": s.get("env"),
                    "act": act,
                    "description": s.get("description"),
                    "arguments": s.get("arguments"),
                    "rationale": s.get("rationale"),
                })
            patch = {
                "replace_range": [r0, r1],
                "substeps": norm_substeps,
            }
        else:
            patch = None

        decision = {
            "status": status,
            "reason": reason,
            "confidence": confidence,
            "plan_bundle_patch": patch,
        }

        # preserve optional diagnostics if any
        diag = data.get("diagnostics")
        if isinstance(diag, dict):
            decision["diagnostics"] = diag

        return decision

    # ---------------------
    # Small utilities
    # ---------------------
    @staticmethod
    def _tail(xs: List[Any], k: int = 3) -> List[Any]:
        k = max(0, int(k))
        return xs[-k:] if xs else []

    @staticmethod
    def _last_exception_snippet(logs: List[str]) -> Optional[str]:
        for line in reversed(logs):
            ll = (line or "").lower()
            if any(tok in ll for tok in ("traceback", "exception", "error", "timeout")):
                return line[-300:]
        return None

    @staticmethod
    def _extract_retry_count(logs: List[str]) -> int:
        # Heuristic: count lines containing a retry marker
        count = 0
        for line in logs or []:
            ll = (line or "").lower()
            if "retry" in ll or "re-try" in ll:
                count += 1
        return count