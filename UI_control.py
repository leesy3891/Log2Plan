# UI_control.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import json
import datetime
from pathlib import Path
import traceback

# Import functions instead of classes
from LocalPlanner import localPlan
from GUI_Parser import GUI_Parser, get_components_from_localenv
from Execution import executeAutomation
from Critic import Critic, AlternativeSubplan, CriticDecision

LOG_DIR = Path("./logs"); LOG_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_JSON = Path("./local_feedback.json")
EXEC_LOG_DIR = Path("./execution_logs"); EXEC_LOG_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class ExecutionLog:
    """실행 로그 구조: LocalPlanner 출력 -> Execution 완료까지의 전체 과정"""
    task_id: int
    substep_index: int
    input_step: Dict[str, Any]  # LocalPlanner 입력 step
    local_plan: List[List[str]]  # LocalPlanner 출력: [action, target] 리스트
    execution_result: Dict[str, Any]  # Execution 결과
    timestamp: str
    success: bool

@dataclass
class TaskSpec:
    tid: int
    env: List[str]
    act: List[str]
    description: str
    step_indices: List[int]             # 이 Task에 속한 Global step index
    task_list: List[Dict[str, Any]]     # 이 Task의 substeps (로컬 실행 단위)

@dataclass
class PlanBundle:
    command: str
    main_plan: List[TaskSpec]
    # 인덱스 관리를 위한 메타데이터
    total_substeps: int = 0
    step_to_task_mapping: Dict[int, int] = field(default_factory=dict)

@dataclass
class UIControlState:
    task_idx: int = 0
    substep_idx: int = 0
    total_steps: int = 0
    final_plan: List[Dict[str, Any]] = field(default_factory=list)
    execution_logs: List[ExecutionLog] = field(default_factory=list)  # 실제 실행 로그
    system_logs: List[str] = field(default_factory=list)  # 시스템 로그 (타임스탬프 등)
    retry_hint: Optional[str] = None
    # 무한 루프 방지
    retry_count_per_task: Dict[int, int] = field(default_factory=dict)
    max_retries_per_task: int = 3

class IndexManager:
    """인덱스 관리 및 plan bundle 조작을 담당하는 유틸리티 클래스"""
    @staticmethod
    def reindex_plan_bundle(bundle: PlanBundle) -> PlanBundle:
        """Plan bundle의 모든 인덱스를 재정렬"""
        total_substeps = 0
        step_to_task_mapping = {}
        
        for task in bundle.main_plan:
            # 각 task의 step_indices 업데이트
            task.step_indices = list(range(total_substeps, total_substeps + len(task.task_list)))
            
            # step to task mapping 업데이트
            for step_idx in task.step_indices:
                step_to_task_mapping[step_idx] = task.tid
                
            total_substeps += len(task.task_list)
        
        bundle.total_substeps = total_substeps
        bundle.step_to_task_mapping = step_to_task_mapping
        return bundle
    
    @staticmethod
    def splice_task_substeps(
        task: TaskSpec, 
        replace_range: Tuple[int, int], 
        new_substeps: List[Dict[str, Any]]
    ) -> TaskSpec:
        """Task 내의 substeps을 안전하게 교체"""
        start_idx, end_idx = replace_range
        
        # 유효성 검사
        if start_idx < 0 or end_idx >= len(task.task_list) or start_idx > end_idx:
            raise ValueError(f"Invalid replace range: {replace_range} for task with {len(task.task_list)} substeps")
        
        # substeps 교체
        new_task_list = (
            task.task_list[:start_idx] + 
            new_substeps + 
            task.task_list[end_idx + 1:]
        )
        
        # 새 TaskSpec 생성
        return TaskSpec(
            tid=task.tid,
            env=task.env,
            act=task.act,
            description=task.description,
            step_indices=[],  # reindex_plan_bundle에서 재계산됨
            task_list=new_task_list
        )

class ExecutionLogManager:
    """실행 로그 관리 클래스"""
    
    def __init__(self, log_dir: Path = EXEC_LOG_DIR):
        self.log_dir = log_dir
    
    def add_execution_log(
        self, 
        logs: List[ExecutionLog], 
        task_id: int,
        substep_index: int,
        input_step: Dict[str, Any],
        local_plan: List[List[str]],
        execution_result: Dict[str, Any],
        success: bool
    ) -> None:
        """실행 로그 추가"""
        log_entry = ExecutionLog(
            task_id=task_id,
            substep_index=substep_index,
            input_step=input_step,
            local_plan=local_plan,
            execution_result=execution_result,
            timestamp=datetime.datetime.now().isoformat(),
            success=success
        )
        logs.append(log_entry)
    
    def get_logs_by_task(self, logs: List[ExecutionLog], task_id: int) -> List[ExecutionLog]:
        """특정 Task의 실행 로그만 반환"""
        return [log for log in logs if log.task_id == task_id]
    
    def get_logs_by_range(
        self, 
        logs: List[ExecutionLog], 
        task_id: int, 
        start_substep: int, 
        end_substep: int
    ) -> List[ExecutionLog]:
        """특정 Task의 특정 substep 범위 실행 로그 반환"""
        return [
            log for log in logs 
            if log.task_id == task_id and start_substep <= log.substep_index <= end_substep
        ]
    
    def save_execution_logs(self, logs: List[ExecutionLog], run_id: str) -> None:
        """실행 로그를 파일로 저장"""
        log_file = self.log_dir / f"execution_{run_id}.json"
        log_data = [
            {
                "task_id": log.task_id,
                "substep_index": log.substep_index,
                "input_step": log.input_step,
                "local_plan": log.local_plan,
                "execution_result": log.execution_result,
                "timestamp": log.timestamp,
                "success": log.success
            }
            for log in logs
        ]
        log_file.write_text(json.dumps(log_data, ensure_ascii=False, indent=2), encoding="utf-8")

class UIController:
    def __init__(self, use_llm_critic: bool=True, critic_model: str="gpt-4o"):
        # Functions are imported directly, no need to instantiate
        self.use_llm_critic = use_llm_critic
        self.critic = Critic(model=critic_model)
        self.state = UIControlState()
        self.index_manager = IndexManager()
        self.exec_log_manager = ExecutionLogManager()
        # 스냅샷 중복 호출 방지를 위한 카운터
        self._snapshot_call_count = 0

    # ----------------- 유틸 -----------------
    def _log(self, line: str):
        """시스템 로그 (타임스탬프 등)"""
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"[{ts}] {line}"
        print(msg)
        self.state.system_logs.append(msg)

    def _save_run_log(self, run_id: str, payload: Dict[str, Any]):
        out = LOG_DIR / f"run_{run_id}.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _append_feedback_json(self, item: Dict[str, Any]):
        prev = []
        if FEEDBACK_JSON.exists():
            try:
                prev = json.loads(FEEDBACK_JSON.read_text(encoding="utf-8"))
                if not isinstance(prev, list):
                    prev = []
            except Exception:
                prev = []
        prev.append(item)
        FEEDBACK_JSON.write_text(json.dumps(prev, ensure_ascii=False, indent=2), encoding="utf-8")

    # GUI/관찰 래퍼 - 함수 호출로 수정하고 중복 호출 방지
    def _snapshot_gui(self, caller_tag: str = "unknown") -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """GUI 스냅샷 생성 - 호출자 태그 포함하여 중복 방지"""
        self._snapshot_call_count += 1
        
        try:
            # 수정된 GUI_Parser 호출 (메타데이터 포함)
            comp_dict, next_page_dict, comp_type, meta_data = GUI_Parser("./current_snapshot.png")
            
            # 로깅에 caller tag 추가
            total_count = meta_data.get("total_component_count", len(comp_dict))
            self._log(f"GUI Snapshot #{self._snapshot_call_count} [by={caller_tag}]: {total_count} components")
            
            return comp_dict, next_page_dict, meta_data
            
        except Exception as e:
            self._log(f"GUI Snapshot failed [by={caller_tag}]: {e}")
            return {}, {}, {
                "active_window_title": "Snapshot Failed",
                "total_component_count": 0,
                "error": str(e),
                "caller_tag": caller_tag
            }
    
    def _before_obs(self, caller_tag: str = "before_obs") -> Tuple[str, Dict[str, Any]]:
        """실행 전 관찰 - 간단한 메타데이터만"""
        try:
            _, _, meta_data = self._snapshot_gui(caller_tag)
            window_title = meta_data.get("active_window_title", "Unknown")
            obs_data = {
                "component_count": meta_data.get("total_component_count", 0),
                "timestamp": meta_data.get("timestamp", "")
            }
            return window_title, obs_data
        except Exception as e:
            self._log(f"Before observation failed: {e}")
            return "Error", {"error": str(e)}
    
    def _after_obs(self, caller_tag: str = "after_obs") -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """실행 후 관찰 - 상세 컴포넌트 정보 포함"""
        try:
            comp_dict, _, meta_data = self._snapshot_gui(caller_tag)
            
            # 컴포넌트를 리스트 형태로 변환 (최대 50개만)
            components = []
            for k, v in list(comp_dict.items())[:50]:  # 너무 많으면 일부만
                try:
                    components.append({
                        "id": k, 
                        "name": v[1] if len(v) > 1 else "Unknown", 
                        "type": v[2] if len(v) > 2 else "Unknown",
                        "coords": v[3] if len(v) > 3 else None
                    })
                except Exception:
                    continue
            
            window_title = meta_data.get("active_window_title", "Unknown")
            obs_summary = {
                "component_count": len(comp_dict),
                "meta_data": meta_data
            }
            
            return window_title, components, obs_summary
            
        except Exception as e:
            self._log(f"After observation failed: {e}")
            return "Error", [], {"error": str(e)}

    def _format_execution_logs_for_critic(self, task_id: int, range_start: int = None, range_end: int = None) -> List[Dict[str, Any]]:
        """Critic에 전달할 실행 로그 포맷팅"""
        if range_start is not None and range_end is not None:
            logs = self.exec_log_manager.get_logs_by_range(
                self.state.execution_logs, task_id, range_start, range_end
            )
        else:
            logs = self.exec_log_manager.get_logs_by_task(self.state.execution_logs, task_id)
        
        return [
            {
                "substep_index": log.substep_index,
                "input": log.input_step,
                "local_plan": log.local_plan,
                "execution_result": log.execution_result,
                "success": log.success,
                "timestamp": log.timestamp
            }
            for log in logs
        ]

    # LocalPlanner 호출을 위한 헬퍼 - 견고성 강화
    def _call_local_planner(self, idx: int, task_list: List[List[str]], current_task: List[str], comp_dict: Dict) -> Tuple[List[List[str]], str, str]:
        """LocalPlanner 함수 호출 - 예외 처리 강화"""
        try:
            # localPlan 함수 호출 (comp_type은 'pwa'로 고정)
            localplan_list, exc, obsv = localPlan(idx, task_list, current_task, comp_dict, 'pwa')
            
            # None 체크 및 기본값 설정
            if localplan_list is None:
                localplan_list = []
            if exc is None:
                exc = "unknown"
            if obsv is None:
                obsv = "No observation provided"
                
            return localplan_list, exc, obsv
            
        except Exception as e:
            self._log(f"LocalPlanner error: {e}")
            traceback.print_exc()
            # 안전한 기본값 반환
            return [], "no", f"LocalPlanner failed: {e}"

    # Execution 호출을 위한 헬퍼
    def _call_executor(self, length: int, tasks: List[List[str]], comp_dict: Dict) -> int:
        """Execution 함수 호출"""
        try:
            return executeAutomation(length, tasks, comp_dict)
        except Exception as e:
            self._log(f"Execution error: {e}")
            raise e

    # ----------------- 실행 본체 -----------------
    def run(self, plan_bundle: PlanBundle) -> Dict[str, Any]:
        self.state = UIControlState()
        self._snapshot_call_count = 0  # 스냅샷 카운터 초기화
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        command = plan_bundle.command
        
        # 초기 인덱스 정리
        plan_bundle = self.index_manager.reindex_plan_bundle(plan_bundle)
        self.state.total_steps = plan_bundle.total_substeps
        
        self._log(f"Start run: '{command}', total_substeps={self.state.total_steps}")

        try:
            for t_idx, task in enumerate(plan_bundle.main_plan):
                self.state.task_idx = t_idx
                self.state.substep_idx = 0
                
                # Task별 재시도 횟수 초기화
                if task.tid not in self.state.retry_count_per_task:
                    self.state.retry_count_per_task[task.tid] = 0
                
                self._log(f"== Task#{task.tid} BEGIN | ENV={task.env} | ACT={task.act} ==")

                # Task 단위 루프
                while self.state.substep_idx < len(task.task_list):
                    sub_i = self.state.substep_idx
                    current_sub = task.task_list[sub_i]

                    # ========== 1회만 스냅샷 생성 (substep 시작 시) ==========
                    comp_dict, next_page_dict, gui_meta_data = self._snapshot_gui(f"Task{task.tid}.substep{sub_i}")
                    
                    # 실행 전 관찰 (스냅샷 재사용)
                    window_title = gui_meta_data.get("active_window_title", "Unknown")
                    b_obs = {
                        "component_count": gui_meta_data.get("total_component_count", 0),
                        "timestamp": gui_meta_data.get("timestamp", "")
                    }

                    # LocalPlanner에 전달할 task_list 형태로 변환
                    task_list_for_lp = []
                    for step in task.task_list:
                        # step이 dict이면 필요한 형태로 변환
                        if isinstance(step, dict):
                            assist = step.get('assist', '0')
                            event = step.get('event', '')
                            obj = step.get('object', '')
                            task_list_for_lp.append([str(assist), event, obj])
                        else:
                            # 이미 리스트 형태인 경우
                            task_list_for_lp.append(step)
                    
                    # current_task도 적절한 형태로 변환
                    if isinstance(current_sub, dict):
                        current_task = [
                            str(current_sub.get('assist', '0')),
                            current_sub.get('event', ''),
                            current_sub.get('object', '')
                        ]
                    else:
                        current_task = current_sub

                    # LocalPlanner 호출 (견고성 강화)
                    try:
                        local_plan, exc, obsv = self._call_local_planner(
                            sub_i, task_list_for_lp, current_task, comp_dict
                        )
                        
                        # exc 값 정규화 (None 안전 처리)
                        exc_norm = (exc or "").strip().lower()
                        
                        # LocalPlanner 결과를 final_plan에 추가
                        local_plan_result = {
                            "substep_index": sub_i,
                            "task_id": task.tid,
                            "input_step": current_sub,
                            "local_plan": local_plan,
                            "executable": exc,
                            "observation": obsv,
                            "gui_meta_data": gui_meta_data  # GUI 메타데이터 포함
                        }
                        self.state.final_plan.append(local_plan_result)
                        
                        # Executable이 'no'인 경우 처리 (None 안전)
                        if exc_norm == 'no' or not local_plan:
                            raise Exception(f"LocalPlanner marked step as non-executable: {obsv}")
                        
                        # Execution 실행 (executeAutomation은 length, tasks, comp_dict를 받음)
                        # length는 현재까지의 누적 실행 수
                        length = len(self.state.final_plan) - 1
                        
                        # executeAutomation이 기대하는 형태로 tasks 변환
                        if local_plan:
                            execution_result = self._call_executor(length, local_plan, comp_dict)
                            execution_result_dict = {"length_after_execution": execution_result}
                        else:
                            execution_result_dict = {"error": "Empty local plan"}
                        
                        # 실행 로그 기록
                        self.exec_log_manager.add_execution_log(
                            logs=self.state.execution_logs,
                            task_id=task.tid,
                            substep_index=sub_i,
                            input_step=current_sub,
                            local_plan=local_plan,
                            execution_result=execution_result_dict,
                            success=True
                        )
                        
                        self._log(f"Executed substep {sub_i} (Task#{task.tid})")

                    except Exception as e:
                        self._log(f"Exception at substep {sub_i}: {e}")
                        traceback.print_exc()

                        # 실행 실패 로그 기록
                        self.exec_log_manager.add_execution_log(
                            logs=self.state.execution_logs,
                            task_id=task.tid,
                            substep_index=sub_i,
                            input_step=current_sub,
                            local_plan=[],
                            execution_result={"error": str(e), "traceback": traceback.format_exc()},
                            success=False
                        )

                        # 예외는 blocked와 동일 처리
                        a_title, a_comps, a_obs = self._after_obs(f"Task{task.tid}.substep{sub_i}.error")
                        
                        # Critic에 전달할 실행 로그 포맷팅
                        exec_logs_for_critic = self._format_execution_logs_for_critic(task.tid)
                        
                        fail_event = {
                            "error_type": "exception",
                            "error_message": str(e),
                            "failed_at_substep": sub_i,
                            "before_obs": {"title": window_title, "state": b_obs},
                            "after_obs": {"title": a_title, "components": a_comps, "state": a_obs},
                            "gui_meta_data": gui_meta_data  # GUI 메타데이터 추가
                        }

                        if self.use_llm_critic:
                            decision = self.critic.judge_and_replan(
                                fail_event=fail_event,
                                plan_bundle={"main_plan": [task.__dict__], "total_substeps": len(task.task_list)},
                                task_context={
                                    "tid": task.tid, "env": task.env, "act": task.act,
                                    "description": task.description,
                                    "execution_logs": exec_logs_for_critic,
                                    "gui_meta_data": gui_meta_data  # GUI 메타데이터 추가
                                },
                                failed_substep_index=sub_i,
                                replace_range=(sub_i, sub_i)
                            )

                            if not decision.is_retryable:
                                # 사용자 피드백 저장 후 종료
                                fb = {"category": "exception", "note": str(e), "ts": datetime.datetime.now().isoformat()}
                                self._append_feedback_json({
                                    "command": command,
                                    "final_plan": self.state.final_plan,
                                    "total_steps": self.state.total_steps,
                                    "blocked": True,
                                    "blocked_task_context": {"tid": task.tid, "substep": sub_i},
                                    "execution_logs": exec_logs_for_critic,
                                    "user_feedback": fb,
                                    "system_logs": self.state.system_logs[-200:],
                                    "ts": datetime.datetime.now().isoformat(),
                                })

                                result = {
                                    "command": command, "final_plan": self.state.final_plan,
                                    "total_steps": self.state.total_steps, "blocked": True,
                                    "halt_task": task.tid, "halt_substep_index": sub_i
                                }
                                self._save_run_log(run_id, result)
                                return result
                        else:
                            # Critic 사용하지 않는 경우 바로 종료
                            result = {
                                "command": command, "final_plan": self.state.final_plan,
                                "total_steps": self.state.total_steps, "blocked": True,
                                "halt_task": task.tid, "halt_substep_index": sub_i
                            }
                            self._save_run_log(run_id, result)
                            return result

                    # 실행 후 관찰 (필요한 경우에만 - 검증용)
                    if self.use_llm_critic:  # Critic이 활성화된 경우에만 사후 관찰
                        a_title, a_comps, a_obs = self._after_obs(f"Task{task.tid}.substep{sub_i}.postcheck")
                    
                    # substep 소진
                    self.state.substep_idx += 1

                # ===== Task의 모든 substep 완료 → Critic 평가 =====
                if self.use_llm_critic:
                    # 최종 상태 스냅샷 (Task 완료 검증용)
                    final_title, final_comps, final_obs = self._after_obs(f"Task{task.tid}.completion")
                    exec_logs_for_critic = self._format_execution_logs_for_critic(task.tid)
                    
                    fail_event = {
                        "error_type": "task_completion_check",
                        "task_completed": True,
                        "after_obs": {"title": final_title, "components": final_comps, "state": final_obs}
                    }

                    decision = self.critic.judge_and_replan(
                        fail_event=fail_event,
                        plan_bundle={"main_plan": [task.__dict__], "total_substeps": len(task.task_list)},
                        task_context={
                            "tid": task.tid, "env": task.env, "act": task.act,
                            "description": task.description,
                            "execution_logs": exec_logs_for_critic
                        },
                        failed_substep_index=len(task.task_list) - 1,
                        replace_range=(len(task.task_list) - 1, len(task.task_list) - 1)
                    )
                    
                    self._log(f"Critic decision for Task#{task.tid}: retryable={decision.is_retryable} ({decision.decision_reason})")

                    if not decision.is_retryable:
                        # Task 성공으로 간주, 다음 Task 진행
                        self.state.retry_hint = None
                        continue

                    # retryable인 경우 처리
                    if decision.is_retryable:
                        # 무한 루프 방지
                        if self.state.retry_count_per_task[task.tid] >= self.state.max_retries_per_task:
                            self._log(f"Max retries exceeded for Task#{task.tid}")
                            fb = {"category": "max_retries", "note": f"Task#{task.tid} exceeded max retries", 
                                  "ts": datetime.datetime.now().isoformat()}
                            self._append_feedback_json({
                                "command": command, "final_plan": self.state.final_plan,
                                "total_steps": self.state.total_steps, "blocked": True,
                                "blocked_task_context": {"tid": task.tid}, "execution_logs": exec_logs_for_critic,
                                "user_feedback": fb, "system_logs": self.state.system_logs[-200:],
                                "ts": datetime.datetime.now().isoformat(),
                            })
                            return {"command": command, "blocked": True, "reason": "max_retries_exceeded"}

                        self.state.retry_count_per_task[task.tid] += 1
                        
                        # 대체 플랜이 있다면 적용
                        if decision.alt_subplan and decision.alt_subplan.replacement_substeps:
                            self._log(f"Apply replacement plan: range={decision.alt_subplan.replace_range}, "
                                      f"len_new={len(decision.alt_subplan.replacement_substeps)}")
                            
                            # 현재 task에 splice 적용
                            try:
                                modified_task = self.index_manager.splice_task_substeps(
                                    task, 
                                    decision.alt_subplan.replace_range,
                                    decision.alt_subplan.replacement_substeps
                                )
                                
                                # plan_bundle 업데이트
                                plan_bundle.main_plan[t_idx] = modified_task
                                plan_bundle = self.index_manager.reindex_plan_bundle(plan_bundle)
                                
                                # 현재 task 참조 업데이트
                                task = plan_bundle.main_plan[t_idx]
                                
                                # 롤백 지점 설정
                                self.state.substep_idx = decision.alt_subplan.replace_range[0]
                                
                            except ValueError as e:
                                self._log(f"Splice failed: {e}")
                                return {"command": command, "blocked": True, "reason": f"splice_failed: {e}"}

                        # LocalPlanner 프롬프트용 힌트 구성
                        self.state.retry_hint = " ".join(decision.task_context_additions) if decision.task_context_additions else None

                        # while 루프 재진입 → 같은 Task 재실행
                        continue

            # ===== 모든 Task 성공 =====
            # 실행 로그 저장
            self.exec_log_manager.save_execution_logs(self.state.execution_logs, run_id)
            
            result = {
                "command": command,
                "final_plan": self.state.final_plan,
                "total_steps": self.state.total_steps,
                "success": True,
                "execution_logs_file": f"execution_{run_id}.json",
                "total_gui_snapshots": self._snapshot_call_count
            }
            self._save_run_log(run_id, result)
            return result

        except Exception as e:
            self._log(f"FATAL: {e}")
            tb = traceback.format_exc()
            
            # 실행 로그 저장
            self.exec_log_manager.save_execution_logs(self.state.execution_logs, run_id)
            
            fb = {"category": "fatal", "note": str(e), "ts": datetime.datetime.now().isoformat()}
            self._append_feedback_json({
                "command": plan_bundle.command, "final_plan": self.state.final_plan,
                "total_steps": self.state.total_steps, "blocked": True,
                "blocked_task_context": {"fatal": True}, "execution_logs": [],
                "user_feedback": fb, "system_logs": self.state.system_logs[-200:] + [tb],
                "ts": datetime.datetime.now().isoformat(),
            })
            
            result = {
                "command": plan_bundle.command, "final_plan": self.state.final_plan,
                "total_steps": self.state.total_steps, "blocked": True,
                "halt_task": self.state.task_idx, "halt_substep_index": self.state.substep_idx,
                "execution_logs_file": f"execution_{run_id}.json",
                "total_gui_snapshots": self._snapshot_call_count
            }
            self._save_run_log(run_id, result)
            return result


# === 레거시 호환용 함수 ===
def UI_Control(plan_bundle: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[ExecutionLog]]:
    """레거시 인터페이스 호환용 함수"""
    controller = UIController()
    
    # Dict를 PlanBundle로 변환
    if isinstance(plan_bundle, dict):
        main_plan = []
        
        # plan_bundle에서 steps를 tasks로 변환
        steps = plan_bundle.get("steps", [])
        tasks = plan_bundle.get("tasks", {})
        task_order = plan_bundle.get("task_order", [])
        
        if not task_order and tasks:
            task_order = list(tasks.keys())
        
        # 기본 task가 없으면 모든 steps를 하나의 task로 처리
        if not tasks:
            task = TaskSpec(
                tid=0,
                env=["local"],
                act=["automation"],
                description="Automated task execution",
                step_indices=list(range(len(steps))),
                task_list=steps
            )
            main_plan.append(task)
        else:
            # 각 task를 TaskSpec으로 변환
            for task_id in task_order:
                if task_id in tasks:
                    task_data = tasks[task_id]
                    task_steps = [step for step in steps if step.get("task_id") == task_id]
                    
                    task = TaskSpec(
                        tid=task_data.get("tid", task_id),
                        env=task_data.get("env", []),
                        act=task_data.get("act", []),
                        description=task_data.get("description", ""),
                        step_indices=task_data.get("step_ids", []),
                        task_list=task_steps
                    )
                    main_plan.append(task)
        
        bundle = PlanBundle(
            command=plan_bundle.get("command", ""),
            main_plan=main_plan
        )
    else:
        bundle = plan_bundle
    
    result = controller.run(bundle)
    
    incomplete_tasks = []
    if result.get("blocked", False):
        incomplete_tasks = [{"reason": result.get("reason", "unknown")}]
    
    return incomplete_tasks, controller.state.execution_logs