# =========================
# main.py (unified launcher • PlanBundle-first)
# =========================
from typing import Any, Dict, List, Optional

# UI_Control는 이제 plan_bundle을 받습니다
from UI_control import UI_Control  # UI_Control(plan_bundle)


# -------------------------------
# Helpers: normalize PlanBundle
# -------------------------------

def _task_list_from_bundle(bundle: Dict[str, Any]) -> List[List[str]]:
    """PlanBundle(dict) -> legacy task_list (optional: 디버깅/로깅용)"""
    if not isinstance(bundle, dict):
        return []
    out: List[List[str]] = []
    for s in bundle.get("steps", []) or []:
        out.append([str(s.get("assist", 0)), s.get("event", ""), s.get("object", "")])
    return out


def _bundle_from_task_list(task_list: List[List[str]]) -> Dict[str, Any]:
    """레거시 globalPlanner(...)가 리스트만 줄 때, 최소한의 PlanBundle로 감싼다."""
    steps = []
    for i, t in enumerate(task_list, 1):
        try:
            assist, event, obj = (int(t[0]), t[1], t[2])
        except Exception:
            # 안전 폴백
            assist, event, obj = (0, t[0] if t else "", t[1] if len(t) > 1 else "")
        steps.append({
            "id": i, "assist": assist, "event": event, "object": obj, "task_id": None
        })
    return {"steps": steps, "tasks": {}, "task_order": []}


def _try_import_globals() -> Dict[str, Any]:
    mods: Dict[str, Any] = {"GP": None, "GP_old": None}
    try:
        import GlobalPlanner as GP_old
        mods["GP_old"] = GP_old
    except Exception:
        pass
    try:
        import GlobalPlanner_new as GP
        mods["GP"] = GP
    except Exception:
        pass
    return mods


def _resolve_api(mod) -> Dict[str, Any]:
    return {
        # 신버전: PlanBundle 반환
        "global_planner": getattr(mod, "global_planner", None),
        # 구버전 패치: PlanBundle 반환
        "globalPlanner_bundle": getattr(mod, "globalPlanner_bundle", None),
        # 구버전 원형: task_list 반환
        "globalPlanner": getattr(mod, "globalPlanner", None),
        "process_command": getattr(mod, "process_command", None),
    }


def _get_plan_bundle(command: str) -> Dict[str, Any]:
    mods = _try_import_globals()

    for tag in ("GP", "GP_old"):
        mod = mods.get(tag)
        if not mod:
            continue
        api = _resolve_api(mod)

        # (A) 신버전: PlanBundle 바로 반환
        if callable(api["global_planner"]):
            try:
                bundle = api["global_planner"](command)
                if isinstance(bundle, dict) and "steps" in bundle:
                    return bundle
            except Exception as e:
                print(f"[WARN] {tag}.global_planner failed: {e}")

        # (B) 구버전 패치: PlanBundle 반환
        if callable(api["globalPlanner_bundle"]):
            try:
                content = api["process_command"](command) if callable(api["process_command"]) else command
                bundle = api["globalPlanner_bundle"](command, content)
                if isinstance(bundle, dict) and "steps" in bundle:
                    return bundle
            except Exception as e:
                print(f"[WARN] {tag}.globalPlanner_bundle failed: {e}")

        # (C) 구버전 원형: task_list만 반환 → 최소 번들로 감싸기
        if callable(api["globalPlanner"]):
            try:
                content = api["process_command"](command) if callable(api["process_command"]) else command
                task_list = api["globalPlanner"](command, content) or []
                return _bundle_from_task_list(task_list)
            except Exception as e:
                print(f"[WARN] {tag}.globalPlanner failed: {e}")
    # 전부 실패 시 빈 번들
    return {"steps": [], "tasks": {}, "task_order": []}

def main():
    command = input("Automation Task: ").strip()
    if not command:
        print("Empty command. Abort.")
        return

    plan_bundle = _get_plan_bundle(command)

    # (선택) 레거시 task_list가 필요하면 변환해서 출력만
    legacy_task_list = _task_list_from_bundle(plan_bundle)
    print("Generated task_list (for debug):", legacy_task_list)
    print("글로벌 플래너 끝끝")

    # ─ 실행 ─
    incomplete_tasks, observations = UI_Control(plan_bundle)

    if incomplete_tasks:
        print("\n[Step 4] Handling Incomplete Tasks...")
        # TODO: newPlan(...) 연동


if __name__ == "__main__":
    main()