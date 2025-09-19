import pyautogui as pag
import time
import re
from pywinauto import mouse, Desktop
import pygetwindow as gw
import os
import openai
from dotenv import load_dotenv
from GUI_Parser import GUI_Parser
import Detectchange

def parse_local_plan(input_string):
    # 'localPlan:' 이후 문자열 추출
    match = re.search(r'localPlan:(.*)', input_string)
    if not match:
        return None, None, None
    3
    local_plan_str = match.group(1)

    # 1. 대괄호 안의 문자열 추출
    matches = re.findall(r"\[([^\[\]]+?)\]", local_plan_str)

    # 2. 각 항목을 분리 및 후처리
    localplan_list = []
    for item in matches:
        parts = item.strip().split(maxsplit=1)
        if len(parts) == 2:
            # 내용1 처리: 공백, ',' 제거 + 소문자 변환
            part1 = parts[0].replace(",", "").lower()
            # 내용2 처리: 앞뒤 공백 제거 + '', "" 제거
            part2 = parts[1].strip().strip("'\"")
            localplan_list.append([part1, part2])
        else:
            # 예외 처리 (예: 하나밖에 없는 경우 등)
            localplan_list.append([parts[0].replace(",", "").lower(), ""])

    print(f"localplan_list = {localplan_list}")
    return localplan_list

def return_win_dict():
    windows = Desktop(backend="uia").windows()
    
    # 인덱스를 키로, [윈도우 이름, 객체]를 값으로 하는 딕셔너리 생성
    win_dict = {
        idx: [w.window_text(), w]
        for idx, w in enumerate(windows)
        if w.window_text().strip()
    }
    
    return win_dict


def executeAutomation(length, tasks, comp_dict):
    """Executes a list of tasks using PyAutoGUI and Pywinauto."""
    print(">>>> Starting Task Execution...")
    task_num = 0
    comp_index = 0    # 이 값은 click/ rightclick ... 등등 컴포넌트 선택 시에 선택된 element의 인덱스
    # tasks 가 문자열임. 파싱해서 리스트로 바꿔야함 > LocalPlanner 출력 보고,,,,
    
    length += len(tasks)

    for task in tasks:
        action = task[0].lower() if len(task) > 0 else None
        argument = task[1].strip() if len(task) > 1 and task[1].strip().strip('\'"') else None
        argument = argument.strip().strip("'\"")
        BLUE = "\033[94m"
        RESET = "\033[0m"
        print(f"{BLUE}Executing task: {action} - {argument}{RESET}")

        # 현재 활성 윈도우 가져오기
        top_window = gw.getActiveWindow() 
        if not top_window:
            print("No active window found.")
            break
        window_left, window_top = top_window.left, top_window.top

        if action.startswith('press'):
            argument = argument.lower()
            if '+' in argument:
                keys = argument.split('+')
                pag.hotkey(*keys)
                time.sleep(1)
            elif argument:
                pag.press(argument)
                time.sleep(4)

        elif action.startswith('type'):
            if 'coupang' in argument:
                pag.typewrite('https://coupan', interval=0.05)
                time.sleep(1)
            elif 'chat' in argument:
                pag.typewrite('https://chat', interval=0.05)
                time.sleep(1)
            elif argument:
                pag.typewrite(argument, interval=0.05)
                time.sleep(1)

        elif action in ['click', 'rightclick', 'doubleclick']:
            if task_num== 0 :   
                match = re.search(r'(\d+)', argument)
                if match:
                    comp_index = int(match.group(1))

                    # 각 key에 대해 value[3] (좌표)만 추출
                    comp_dict_coord = {
                        k: v[3] for k, v in comp_dict.items()
                    }

                    # comp_index 키에 해당하는 좌표값 가져오기
                    coord = comp_dict_coord.get(comp_index)
                    print("comp_coordinates: ", coord)
                    if coord:
                        x, y = coord
                        executeMouseAction(action, x, y, window_left, window_top)
                        task_num +=1
                    else:
                        print(f"Invalid coordinates for {action}: {argument}")
                else:
                    print(f"Invalid argument for {action}: {argument}")
            else:
                load_dotenv('./openaikey.env')
                openai_api_key = os.getenv("OPENAI_API_KEY")
                if openai.api_key:
                    print("API 키가 성공적으로 설정되었습니다.")

                from openai import OpenAI

                client = openai.OpenAI()

                comp_dict, _, _ = GUI_Parser(f"./case0/1.png")  # GUI 요소 탐색
                comp_dict_prompt = {
                    k: v[1:3] for k, v in comp_dict.items()
                }
                
               #  task_options = "click/rightclick/doubleclick, target object"
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a Desktop Automation Agent. Your goal is to generate an optimal automation command: localPlan for a given task. You will be given current task, and component dictionary extracted from Windows GUI env."},
                        {"role": "system", "content": "Each task follows the format ['Event', 'object'] which is in [click/rightclick/doubleclick, target object]. The object specifies where the action should take place."},
                        {"role": "system", "content": "Determine the target object in the component dictionary with no prepositions. The generated task should take form of [click/rightclick/doubleclick, index. component name(component_type)] with no prepositions."},
                        {"role": "system", "content": "If component is in component dict(index: [component name, component type] form), select the most semantically similar component to the object in task.  ex) [click, index. component name(component type)]"},
                        {"role": "system", "content": "To match a task object with a UI component, normalize text (remove quotes, lowercase, strip UI-specific suffixes), account for translation or paraphrasing (e.g., 'images' ≈ '이미지'), and select the component that best matches the task intent."},
                        {"role": "system", "content": "For documents or image, pdf, etc.., match using known titles or domain knowledge (e.g., 'Multihead Attention' → 'Attention is all you need')."},
                        {"role": "user", "content": f"Task list (context): {tasks}, Current task: {task}, Component dictionary: {comp_dict_prompt}."},
                        {"role": "user", "content": "Provide the selected task list with the chosen component."},
                        {"role": "user", "content": "Always format your response as follows -> [click/rightclick/doubleclick, target object]"}
                    ],
                    max_tokens=3000,
                    temperature=0.5
                )
                response = response.choices[0].message.content

                # 1. 대괄호 안의 문자열 추출
                matches = re.findall(r"\[([^\[\]]+?)\]", response)

                # 2. 각 항목을 분리 및 후처리
                tasks = []
                for item in matches:
                    parts = item.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        # 내용1 처리: 공백, ',' 제거 + 소문자 변환
                        part1 = parts[0].replace(",", "").lower()
                        # 내용2 처리: 앞뒤 공백 제거 + '', "" 제거
                        part2 = parts[1].strip().strip("'\"")
                        tasks.append([part1, part2])
                    else:
                        # 예외 처리 (예: 하나밖에 없는 경우 등)
                        tasks.append([parts[0].replace(",", "").lower(), ""])

                print(f"Modified Task = {tasks}")

                action = task[0].lower() if len(task) > 0 else None
                argument = task[1].lower() if len(task) > 1 else None

                match = re.search(r'(\d+)', argument)
                if match:
                    comp_index = int(match.group(1))

                    # 각 key에 대해 value[3] (좌표)만 추출
                    comp_dict_coord = {
                        k: v[3] for k, v in comp_dict.items()
                    }

                    # comp_index 키에 해당하는 좌표값 가져오기
                    coord = comp_dict_coord.get(comp_index)
                    print("comp_coordinates: ", coord)
                    if coord:
                        x, y = coord
                        executeMouseAction(action, x, y, window_left, window_top)
                        task_num +=1
                    else:
                        print(f"Invalid coordinates for {action}: {argument}")
                else:
                    print(f"Invalid argument for {action}: {argument}")

        elif action.startswith('drag'):
            match = re.search(r'(\d+)', argument)
            if match:
                comp_index = int(match.group(1))

                # 각 key에 대해 value[3] (좌표)만 추출
                comp_dict_coord = {
                    k: v[3] for k, v in comp_dict.items()
                }

                # comp_index 키에 해당하는 좌표값 가져오기
                coord = comp_dict_coord.get(comp_index)
                print("comp_coordinates: ", coord)
                if coord:
                    x, y = coord
                    executeDragAction(action, x, y, window_left, window_top)
                    task_num +=1
                else:
                    print(f"Invalid coordinates for {action}: {argument}")
            else:
                print(f"Invalid argument for {action}: {argument}")
        elif action.startswith('save'):
            pag.hotkey('ctrl', 's')
        elif action.startswith('wait'):
            Detectchange.detect_changes()

        elif action.startswith('switch focus'):
            print("this is switch focus")

            load_dotenv('./openaikey.env')
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai.api_key:
                print("API 키가 성공적으로 설정되었습니다.")

            from openai import OpenAI
            client = openai.OpenAI()

            win_dict = return_win_dict()
            win_dict_prompt = {
                    k: v[0] for k, v in win_dict.items()
            }
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a Desktop Automation Agent. Your goal is to choose the right window to switch focus to. You will be given a part of the task list, current task, and Open Window dictionary extracted from Windows GUI env."},
                    {"role": "system", "content": "Current task follows the format ['switch focus', 'key. win_name']. The object gives information on which window you you should switch focus to."},
                    {"role": "system", "content": "Event is 'switch focus'. you should choose the right window to switch focus to. "},
                    {"role": "system", "content": "Choose the best matching Window using the available Window dictionary (index: [window name] form). Select the window which best matches the given object including the index. ex) [switch focus, index. window name]"},
                    {"role": "system", "content": "Example:\n[Switch Focus, back to Excel\nSelected Window: 7: budget2024 - Excel"},
                    {"role": "user", "content": f"Task list (context): {tasks}, Current task: {task}, Window dictionary: {win_dict_prompt}."},
                    {"role": "user", "content": "Provide the selected task list with the chosen component. Always format your response as follows -> Selected Window: index: [window name]"}
                ],
                max_tokens=1000,
                temperature=0.5
            )
            response = response.choices[0].message.content

            match = re.search(r'(\d+)', response)
            if match:
                win_index = int(match.group(1))
                print(f"Selected Window index: {win_index} in ({response})")
                
                win_dict = return_win_dict()
                
                if win_index in win_dict:
                    _, win_spec = win_dict[win_index]
                    win_spec.set_focus()
                else:
                    raise IndexError(f"인덱스 {win_index}가 윈도우 목록에 존재하지 않음")
            else:
                raise ValueError("올바른 숫자 범위를 찾을 수 없음")
        else:    
            print("Action not recognized.")

        time.sleep(0.5)
    print(">>>> Task Execution Completed!")
    return length

def executeMouseAction(action, x, y, window_left, window_top):
    """좌표 기반 마우스 동작 수행"""
    print(f"좌표 기반 마우스 동작 수")
    adjusted_x = x 
    adjusted_y = y
    pag.moveTo(adjusted_x, adjusted_y)
    
    if action == "click":
        mouse.click(coords=(int(adjusted_x), int(adjusted_y)))
    elif action == "rightclick":
        mouse.right_click(coords=(int(adjusted_x), int(adjusted_y)))
    elif action == "doubleclick":
        mouse.double_click(coords=(int(adjusted_x), int(adjusted_y)))

    print(f"{action} performed at ({adjusted_x}, {adjusted_y})")

def executeDragAction(x, y, window_left, window_top):
    """드래그 동작 수행"""
    adjusted_x = x
    adjusted_y = y
    pag.moveTo(int(adjusted_x), int(adjusted_y))
    pag.mouseDown()
    pag.moveTo(1250, 800, duration=1)
    pag.mouseUp()
    print(f"Dragged from ({adjusted_x}, {adjusted_y})")
