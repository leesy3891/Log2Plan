from LocalPlanner import localPlan
from Execution import executeAutomation
from GUI_Parser import GUI_Parser
import pyautogui as pag
import time
import re

def UI_Control(task_list):
    print("\n[Step 1] Generating Local Plan...")
    comp_dict = {}  # UI 요소 매칭 딕셔너리
    final_plan = []
    incomplete_tasks = []
    observations = []
    length = 0
    # ANSI escape codes for colors
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    RESET = "\033[0m"

    start_time = time.time()  # 전체 루프 시작 시간

    for idx, task in enumerate(task_list):
        print("Processing task:", task)
        if task[0] == '1': 
                print("Popup !! ")
                pag.alert(text=f"{task[1]}, {task[2]}", title='USER_ASSIST_CHECK', button='OK')
                time.sleep(3)
                final_plan.append(task)
                length +=1
        else:
            if is_simple_command(task):
                print(f"Directly Executed: {task}")
                final_plan.append(task)  # 단순 명령어는 바로 실행
                length = executeAutomation(length, [task[1:]], comp_dict)
            else:
                print("\n[Step 2.5] Using GUI Parser...")
                comp_dict, next_page_dict, _ = GUI_Parser(f"./case0/{idx}.png")  # GUI 요소 탐색
                comp_dict_prompt = {
                    k: v[1:3] for k, v in comp_dict.items()
                }
                #print("comp_dict: ", comp_dict_prompt)
                localplan, exc, obsv = localPlan(idx, task_list, task, comp_dict_prompt, comp_type='pwa')


                # next_page_dict를 염두에 둔 설계 추가
                # 추가 탐색 시작작
                if exc=='no':
                    try:
                        pag.scroll(1200)
                        next_comp_dict, next_page_dict, _ = GUI_Parser(f"./case0/{idx}.png")
                        next_comp_dict_prompt = {
                            k: v[1:3] for k, v in next_comp_dict.items()
                        }
                        localplan, exc, obsv = localPlan(idx, task_list, task, next_comp_dict_prompt, comp_type='pwa') 
                        print("\n[Step 3] Executing Automation...")
                        length = executeAutomation(length, localplan, next_comp_dict)
                    except Exception as e:
                        incomplete_tasks.append(task[idx:])
                        observations.append(obsv) 
                else:
                    #final_plan.extend(localplan)
                    final_plan.append(localplan)
                    print("\n[Step 3] Executing Automation...")
                    length = executeAutomation(length, localplan, comp_dict)

        print(f"Total length = {length}") 

    end_time = time.time()
   # Your original code with colored prints
    print(f"{CYAN}Final Plan: {RESET}{final_plan}")
    print(f"\n{GREEN}[Summary]{RESET} Total processing time: {YELLOW}{end_time - start_time:.2f} seconds{RESET}")
    return incomplete_tasks, observations

def is_simple_command(task):
    if not isinstance(task, list) or len(task) < 2:
        return False  # task가 비정상적이면 단순 명령이 아님
    return task[1] in {"press", "save", "wait", "type", "switch focus"}
