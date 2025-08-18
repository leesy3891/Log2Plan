import os
import re
import pandas as pd
from glob import glob
import ast

def get_unprocessed_log_files(log_dir="action_logs"):
    # 변환되지 않은 action_log_*.txt 파일 목록을 가져옴
    log_files = glob(os.path.join(log_dir, "action_log_*.txt"))
    converted_files = glob(os.path.join(log_dir, "converted_action_log_*.txt"))
    
    converted_timestamps = {re.search(r"converted_action_log_(\d+)\.txt", os.path.basename(f)).group(1) 
                            for f in converted_files if re.search(r"converted_action_log_(\d+)\.txt", os.path.basename(f))}
    unprocessed_logs = [f for f in log_files if re.search(r"action_log_(\d+)\.txt", os.path.basename(f)) and 
                         re.search(r"action_log_(\d+)\.txt", os.path.basename(f)).group(1) not in converted_timestamps]
    return unprocessed_logs
    
def generate_output_filename(raw_log_file, output_dir="action_logs"):
    #원본 로그 파일명을 기반으로 변환된 파일명 생성
    base_name = os.path.basename(raw_log_file)
    match = re.search(r"action_log_(\d+)\.txt", base_name)
    
    if match:
        timestamp = match.group(1)
        return os.path.join(output_dir, f"converted_action_log_{timestamp}.txt")
    else:
        return os.path.join(output_dir, "converted_action_log.txt")  # 기본값

def load_event_dictionary(excel_path="task_dictionary.xlsx"):
    #엑셀 파일에서 정의된 이벤트 딕셔너리를 로드 (task1~task6 포함)
    try:
        if not os.path.exists(excel_path):
            print(f"❌ Event dictionary file not found: {excel_path}")
            return {}
            
        df = pd.read_excel(excel_path)
        event_dict = {}

        for _, row in df.iterrows():
            tasks = [str(row[col]).strip().lower() for col in ["task1", "task2", "task3", "task4", "task5", "task6"] 
                    if pd.notna(row[col]) and str(row[col]).strip()]
            if tasks:  # 비어있지 않은 경우만 딕셔너리에 추가
                event_dict[row['Event']] = tasks
        
        if not event_dict:
            print("⚠️ Warning: Event dictionary is empty")
        return event_dict
    except Exception as e:
        print(f"❌ Error loading event dictionary: {e}")
        return {}

def translate_control_key(key):
    # 컨트롤 키 조합을 읽기 쉬운 형식으로 변환
    ctrl_key_map = {
        "ctrl_l+\\x03": "ctrl+c",
        "ctrl_l+\\x16": "ctrl+v",
        "ctrl_l+\\x13": "ctrl+s",
        "ctrl_l+\\x01": "ctrl+a",
        "ctrl_r+\\x03": "ctrl+c",
        "ctrl_r+\\x16": "ctrl+v",
        "ctrl_r+\\x13": "ctrl+s",
        "ctrl_r+\\x01": "ctrl+a",
    }
    if key in ctrl_key_map:
        return ctrl_key_map[key]
    
    match = re.match(r"ctrl_([lr])\+(.+)", key)
    if match:
        return f"ctrl+{match.group(2)}"
    
    return key

def post_process_events(events):
    # 이벤트 목록을 후처리하여 연속적인 클릭을 더블클릭으로 변환
    processed_events = []
    i = 0
    
    while i < len(events):
        current_event = events[i]
        
        # 연속된 스크롤 패턴 감지
        if (i < len(events) - 1 and 
            current_event["event"] == "Scroll" and 
            events[i+1]["event"] == "Scroll"):
            
            # 시작 스크롤 값과 끝 스크롤 값 추출
            start_scroll = current_event["target"]
            end_scroll = events[i+1]["target"]
            
            # 스크롤 그룹 이벤트 추가
            processed_events.append({
                "event": "ScrollGroup",
                "target": f"from {start_scroll} to {end_scroll}",
                "window": current_event["window"],
                "timestamp": current_event["timestamp"]
            })
            
            # 다음 스크롤 이벤트도 스킵
            i += 2
            continue
        
        # 연속된 클릭 패턴 감지
        elif (i < len(events) - 1 and 
              current_event["event"] == "Click" and 
              events[i+1]["event"] == "Click"):
            
            # 같은 요소에 대한 연속 클릭인지 확인
            if (current_event["target"] == events[i+1]["target"] and
                current_event.get("control_type", "") == events[i+1].get("control_type", "") and
                current_event["window"] == events[i+1]["window"]):
                
                # 창 전환이 발생하는지 확인
                window_switch_after = False
                for j in range(i+2, min(i+5, len(events))):  # 최대 3개 이벤트 앞으로 확인
                    if events[j]["event"] == "Switch Focus" or events[j]["event"] == "Go To":
                        window_switch_after = True
                        break
                
                # 창 전환이 있으면 더블클릭으로 처리
                if window_switch_after:
                    processed_events.append({
                        "event": "Doubleclick",
                        "target": current_event["target"],
                        "control_type": current_event.get("control_type", ""),
                        "window": current_event["window"],
                        "timestamp": current_event["timestamp"]
                    })
                    # 다음 이벤트도 스킵 (이미 더블클릭으로 처리했으므로)
                    i += 2
                    continue
                # 같은 요소에 대한 다중 클릭(2회 이상)이 연속될 경우에도 더블클릭으로 처리
                elif i < len(events) - 2 and events[i+2]["event"] == "Click" and current_event["target"] == events[i+2]["target"]:
                    processed_events.append({
                        "event": "Doubleclick",
                        "target": current_event["target"],
                        "control_type": current_event.get("control_type", ""),
                        "window": current_event["window"],
                        "timestamp": current_event["timestamp"]
                    })
                    # 다음 이벤트들도 스킵 (이미 더블클릭으로 처리했으므로)
                    next_target = current_event["target"]
                    next_idx = i + 3
                    
                    # 같은 타겟에 대한 모든 연속 클릭 스킵
                    while next_idx < len(events) and events[next_idx]["event"] == "Click" and events[next_idx]["target"] == next_target:
                        next_idx += 1
                    
                    i = next_idx
                    continue
        
        # 이벤트를 그대로 추가
        processed_events.append(current_event)
        i += 1
    
    return processed_events

def parse_raw_logs(raw_log_file):
    """
    원시 로그 파일을 파싱하여 이벤트 목록을 추출
    """
    events = []
    current_input = ""
    current_window = "Unknown"
    current_window_time = ""
    text_input_mode = False
    # 특수 문자 매핑 정의
    special_char_map = {
        "shift_r+@": "@", "shift_l+@": "@",
        "shift_r+!": "!", "shift_l+!": "!",
        "shift_r+#": "#", "shift_l+#": "#",
        "shift_r+$": "$", "shift_l+$": "$",
        "shift_r+%": "%", "shift_l+%": "%",
        "shift_r+^": "^", "shift_l+^": "^",
        "shift_r+&": "&", "shift_l+&": "&",
        "shift_r+*": "*", "shift_l+*": "*",
        "shift_r+(": "(", "shift_l+(": "(",
        "shift_r+)": ")", "shift_l+)": ")",
        "shift_r+_": "_", "shift_l+_": "_",
        "shift_r++": "+", "shift_l++": "+",
        "shift_r+{": "{", "shift_l+{": "{",
        "shift_r+}": "}", "shift_l+}": "}",
        "shift_r+|": "|", "shift_l+|": "|",
        "shift_r+:": ":", "shift_l+:": ":",
        "shift_r+\"": "\"", "shift_l+\"": "\"",
        "shift_r+<": "<", "shift_l+<": "<",
        "shift_r+>": ">", "shift_l+>": ">",
        "shift_r+?": "?", "shift_l+?": "?",
        "shift_r+~": "~", "shift_l+~": "~",
        "period": ".", "comma": ",", "semicolon": ";",
        "slash": "/", "backslash": "\\", "quote": "'",
        "minus": "-", "plus": "+", "equal": "=",
        "leftbracket": "[", "rightbracket": "]",
        "grave": "`"
    }

    # cmd+enter, cmd+tab, cmd+esc 등의 특수 키 조합 처리
    special_keys = ["enter", "tab", "esc", "backspace", "space", "up", "down", "left", "right", 
                    "home", "end", "pageup", "pagedown", "delete", "insert"]
                   
    
    # 전체 로그 라인 읽기
    with open(raw_log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 로그 처리 시작
    for i, line in enumerate(lines):
        try:
            match = re.search(r'\[(.*?)\] (.*?): (.*)$', line.strip())
            if not match:
                continue
            timestamp, event_type, details = match.groups()
            # 문자열을 안전하게 변환
            try:
                details = ast.literal_eval(details)
            except (SyntaxError, ValueError):
                print(f"⚠️ Warning: Unable to parse details in line: {line.strip()}")
                continue  # 해당 라인은 무시하고 넘어감

            # 에러 메시지나 불필요한 이벤트는 무시
            if isinstance(details, (list, tuple)) and any(isinstance(item, str) and "이벤트에서 가입자를 불러낼 수 없습니다" in item for item in details):
                continue
            if event_type == "Error":
                continue

            # 키보드 이벤트 처리
            if event_type == "Keyboard" and isinstance(details, (list, tuple)) and len(details) >= 2:
                key = details[1]
                
                if key == "cmd":
                    if current_input:
                        events.append({
                            "event": "Text input",
                            "target": current_input,
                            "window": current_window,
                            "timestamp": current_window_time
                        })
                        current_input = ""
                        text_input_mode = False
                    
                    # Win 키 이벤트 추가
                    events.append({
                        "event": "Press",
                        "target": "win",
                        "window": current_window,
                        "timestamp": timestamp
                    })
                    continue 

                # cmd+ 조합 처리 - 텍스트 입력 또는 특수 키 처리
                if isinstance(key, str) and key.startswith("cmd+"): 
                    # cmd+ 이후의 문자열 추출
                    after_cmd = key[4:]
                    
                    # cmd+shift+a와 같은 복합 조합 처리
                    if '+' in after_cmd:
                        parts = after_cmd.split('+')
                        last_part = parts[-1]
                        
                        # 마지막 부분이 특수 키인 경우 (cmd+shift+enter 등)
                        if last_part.lower() in special_keys:
                            if current_input:
                                events.append({
                                    "event": "Text input",
                                    "target": current_input,
                                    "window": current_window,
                                    "timestamp": current_window_time
                                })
                                current_input = ""
                                text_input_mode = False
                            
                            # 특수 키 이벤트 추가
                            events.append({
                                "event": "Press",
                                "target": f"cmd+{after_cmd}",
                                "window": current_window,
                                "timestamp": timestamp
                            })
                        # 마지막 부분이 단일 문자인 경우 (cmd+shift+a 등)
                        elif len(last_part) == 1:
                            # 텍스트 입력 모드 시작
                            if not text_input_mode:
                                current_window_time = timestamp
                                text_input_mode = True
                                
                            if any(p.startswith('shift') for p in parts):  # shift 포함 여부 확인
                                current_input += last_part.upper()  # 대문자로 변환
                            else:
                                current_input += last_part.lower()  # 소문자로 변환
                    # 단순 cmd+key 조합 처리
                    else:
                        # 특수 키인 경우 (cmd+enter, cmd+tab 등)
                        if after_cmd.lower() in special_keys:
                            if current_input:
                                events.append({
                                    "event": "Text input",
                                    "target": current_input,
                                    "window": current_window,
                                    "timestamp": current_window_time
                                })
                                current_input = ""
                                text_input_mode = False
                            
                            # 특수 키 이벤트 추가
                            events.append({
                                "event": "Press",
                                "target": key,  # 전체 조합 유지 (cmd+enter 등)
                                "window": current_window,
                                "timestamp": timestamp
                            })

                        # 일반 문자인 경우 (cmd+a, cmd+b 등)
                        elif len(after_cmd) == 1:
                            # 텍스트 입력 모드 시작
                            if not text_input_mode:
                                current_window_time = timestamp
                                text_input_mode = True
                                
                            current_input += after_cmd.lower()  # 소문자로 추가
                    
                    continue

                # 컨트롤 키 조합 감지 (간단한 문자열 검사 사용)
                if isinstance(key, str) and ("ctrl_l+" in key or "ctrl_r+" in key):
                    if current_input:
                        events.append({
                            "event": "Text input",
                            "target": current_input,
                            "window": current_window,
                            "timestamp": current_window_time
                        })
                        current_input = ""
                        text_input_mode = False
                    # Ctrl+C (복사) 감지
                    if "\\x03" in key or key.endswith("+c"):
                        events.append({
                            "event": "Copy",
                            "target": "Selected content",
                            "window": current_window,
                            "timestamp": timestamp
                        })
                    # Ctrl+V (붙여넣기) 감지
                    elif "\\x16" in key or key.endswith("+v"):
                        events.append({
                            "event": "Paste",
                            "target": "Clipboard content",
                            "window": current_window,
                            "timestamp": timestamp
                        })
                    # Ctrl+S (저장) 감지
                    elif "\\x13" in key or key.endswith("+s"):
                        events.append({
                            "event": "Save",
                            "target": "",
                            "window": current_window,
                            "timestamp": timestamp
                        })
                    # 다른 컨트롤 키 조합은 직접 추가
                    else:
                        translated_key = translate_control_key(key)
                        events.append({
                            "event": "Press",
                            "target": translated_key,
                            "window": current_window,
                            "timestamp": timestamp
                        })
                # 입력 확정 키 (enter, tab, esc) 처리
                elif key in ["enter", "tab", "esc"]:
                    if current_input:
                        events.append({
                            "event": "Text input",
                            "target": current_input,
                            "window": current_window,
                            "timestamp": current_window_time
                        })
                        current_input = ""
                        text_input_mode = False
                    events.append({
                        "event": "Press",
                        "target": key,
                        "window": current_window,
                        "timestamp": timestamp
                    })
                # 일반 문자 입력 및 특수키 처리
                else:
                    # 텍스트 입력 모드 시작
                    if not text_input_mode:
                        current_window_time = timestamp
                        text_input_mode = True
                    # 백스페이스 처리 - 현재 입력 문자열만 수정
                    if key == "backspace":
                        if current_input:
                            current_input = current_input[:-1]
                    # 스페이스바 처리
                    elif key == "space":
                        current_input += " "
                    # 특수 문자 매핑 처리
                    elif key in special_char_map:
                        current_input += special_char_map[key]
                    # Shift 키 조합 처리
                    elif isinstance(key, str) and key.startswith("shift_") and "+" in key:
                        parts = key.split("+")
                        if len(parts) > 1:
                            # 알파벳이면 대문자로
                            if len(parts[1]) == 1 and parts[1].isalpha():
                                current_input += parts[1].upper()
                            # 숫자나 다른 문자는 그대로 (로그에 실제 문자가 기록됨)
                            else:
                                second_part = parts[1]
                                current_input += second_part
                    # 일반 문자 처리 (알파벳, 숫자, 특수문자)
                    elif isinstance(key, str) and len(key) == 1:
                        current_input += key
                    # 다른 키 처리
                    elif isinstance(key, str):
                        # 특수 키 이름이 있으면 변환
                        if key in special_char_map:
                            current_input += special_char_map[key]
                        # 일반 키는 그대로 추가
                        elif not key.startswith(("ctrl_", "alt_", "shift_")):
                            current_input += key
                continue  # 키보드 이벤트 처리 후 다음 로그라인으로

            # Mouse 이벤트 처리
            if event_type == "Mouse" and isinstance(details, (list, tuple)):
                # 마우스 이벤트가 있을 때 현재 입력이 있으면 확정
                if current_input:
                    events.append({
                        "event": "Text input",
                        "target": current_input,
                        "window": current_window,
                        "timestamp": current_window_time
                    })
                    current_input = ""
                    text_input_mode = False
                
                # Scroll 이벤트 먼저 처리 - 창 전환 로직 실행 전에 처리
                if len(details) > 0 and details[0] == "Scroll":
                    try:
                        # Scroll 정보를 그대로 저장
                        scroll_info = details[1:]  # 'Scroll' 다음의 모든 값
                        
                        events.append({
                            "event": "Scroll",
                            "target": str(scroll_info),  # 배열을 문자열로 변환하여 저장
                            "window": current_window,
                            "timestamp": timestamp
                        })
                    except Exception as e:
                        print(f"⚠️ Warning: Error processing scroll event: {e}")
                    continue
                
                # 마우스 드래그 처리 - 창 전환 로직 실행 전에 처리
                elif len(details) > 2 and details[1] == "Dragged":
                    try:
                        # UI 요소가 있는 경우 클릭 이벤트로 처리
                        if len(details) >= 3 and isinstance(details[2], str) and details[2] and details[2] != "''":
                            ui_element = details[2]
                            control_type = details[3] if len(details) >= 4 and details[3] else "Unknown"
                            
                            # 클릭 이벤트로 처리
                            events.append({
                                "event": "Click",
                                "target": ui_element,
                                "control_type": control_type,
                                "window": current_window,
                                "timestamp": timestamp
                            })
                        # 좌표만 있는 일반 드래그 이벤트 처리
                        elif len(details) >= 5:
                            # details[3]은 첫 번째 좌표 (시작점), details[4]는 두 번째 좌표 (끝점)
                            if isinstance(details[3], tuple) and isinstance(details[4], tuple):
                                start_pos = details[3]
                                end_pos = details[4]
                                
                                # 즉시 드래그 이벤트 추가 - 좌표 형식 유지
                                events.append({
                                    "event": "Drag",
                                    "target": f"from {start_pos} to {end_pos}",
                                    "window": current_window,
                                    "timestamp": timestamp
                                })
                    except Exception as e:
                        print(f"⚠️ Warning: Error processing drag event: {e}, details: {details}")
                    continue
                
                # 윈도우 이름 확인 - 문자열인 경우에만 처리
                window_name = "Unknown"
                if len(details) > 4 and isinstance(details[4], str):
                    window_name = details[4]
                
                # 창 전환 이벤트 추가 (중복 방지)
                if window_name != "Unknown" and window_name != current_window:
                    if not events or events[-1]["event"] != "Switch Focus" or events[-1]["target"] != window_name:
                        try:
                            # 마지막 구분자 이후의 부분을 추출하는 함수
                            def extract_last_part(window_title):
                                if " - " in window_title:
                                    return window_title.split(" - ")[-1].strip()
                                return window_title
                            
                            # 각 창 제목에서 마지막 부분 추출
                            last_part_old = extract_last_part(current_window)
                            last_part_new = extract_last_part(window_name)
                            
                            # 마지막 부분이 같고 창 제목이 다르면 "Go To"
                            if last_part_old == last_part_new and current_window != window_name:
                                # Go To (Navigation) 이벤트 추가
                                events.append({
                                    "event": "Go To",
                                    "target": window_name,
                                    "window": window_name,
                                    "timestamp": timestamp
                                })
                            else:
                                # Switch Focus 이벤트 추가
                                events.append({
                                    "event": "Switch Focus",
                                    "target": window_name,
                                    "window": window_name,
                                    "timestamp": timestamp
                                })
                            
                            current_window = window_name
                            current_window_time = timestamp
                        except Exception as e:
                            print(f"⚠️ Warning: Error processing window change: {e}")
                
                if len(details) >= 5 and details[0] in ["Left", "Right"] and details[1] in ["Pressed", "Released"]:
                    # 클릭 이벤트 추가 (Pressed만 처리)
                    if details[1] == "Pressed":
                        ui_element = details[2]
                        control_type = details[3]
                        if ui_element and not isinstance(ui_element, tuple) and not "Error" in str(ui_element):
                            # 에러 이벤트나 빈 UI 요소는 무시
                            click_type = "Click" if details[0] == "Left" else "Rightclick"
                            events.append({
                                "event": click_type,
                                "target": ui_element,
                                "control_type": control_type,
                                "window": current_window,
                                "timestamp": timestamp
                            })
                
                if len(details) > 3 and details[0] == "Left" and details[1] == "Pressed" and details[3] == "Text":
                    selected_text = details[2]  # 선택된 텍스트 저장                    
                    # 즉시 선택 이벤트 추가
                    events.append({
                        "event": "Select",
                        "target": selected_text,
                        "window": current_window,
                        "timestamp": timestamp
                    })
                    
        except Exception as e:
            print(f"Error parsing line: {line.strip()}, Error: {e}")

    # 마지막 current_input 처리
    if current_input:
        events.append({
            "event": "Text input",
            "target": current_input,
            "window": current_window,
            "timestamp": current_window_time
        })
    # 이벤트 후처리 - 연속 클릭을 더블클릭으로 변환
    events = post_process_events(events)
    return events
                
def generate_command_list(events, event_dict):
    """
    파싱된 이벤트를 기반으로 명령 목록 생성
    """
    commands = []
    
    # 이벤트를 순회하며 명령어 생성
    i = 0
    while i < len(events):
        event = events[i]
        event_type = event["event"]
        
        # Select 다음에 Copy가 오는 패턴 감지
        if event_type == "Select" and i + 1 < len(events) and events[i+1]["event"] == "Copy":
            selected_text = event["target"]
            if isinstance(selected_text, str) and len(selected_text) > 50:
                selected_text = selected_text[:47] + "..."
            
            # 선택 명령 추가
            command = f"0#select, {selected_text}"
            commands.append(command)
            
            # Copy 명령 추가
            command = f"0#copy,"
            commands.append(command)
            
            # Select와 Copy 이벤트 둘 다 처리했으므로 인덱스 2 증가
            i += 2
            continue
        
        # 일반 Copy 이벤트 처리
        elif event_type == "Copy":
            command = f"0#copy,"
            commands.append(command)
            i += 1
            continue
        
        # Paste 이벤트 처리
        elif event_type == "Paste":
            command = f"0#paste,"
            commands.append(command)
            i += 1
            continue
        
        # 저장 이벤트 처리
        elif event_type == "Save":
            command = f"0#save,"
            commands.append(command)
            i += 1
            continue
        
        # 스크롤 그룹 이벤트 처리
        elif event_type == "ScrollGroup":
            target = event["target"]
            command = f"0#scroll, {target}"
            commands.append(command)
            i += 1
            continue
        
        # 일반 스크롤 이벤트 처리
        elif event_type == "Scroll":
            target = event["target"]
            command = f"0#scroll, {target}"
            commands.append(command)
            i += 1
            continue
        
        # 창 전환 이벤트 처리
        elif event_type == "Switch Focus":
            command = f"0#Switch Focus, {event['target']}"
            commands.append(command)
            i += 1
            continue
            
        # Go To (Navigation) 이벤트 처리
        elif event_type == "Go To":
            command = f"0#Go To (Navigation), {event['target']}"
            commands.append(command)
            i += 1
            continue
        
        # 텍스트 입력 이벤트 처리
        elif event_type == "Text input":
            target = event["target"]
            if isinstance(target, str) and len(target) > 50:
                target = target[:47] + "..."
            command = f"0#Text Input, {target}"
            commands.append(command)
            i += 1
            continue
        
        # 텍스트 선택 이벤트 처리
        elif event_type == "Select":
            target = event["target"]
            if isinstance(target, str) and len(target) > 50:
                target = target[:47] + "..."
            command = f"0#select, {target}"
            commands.append(command)
            i += 1
            continue
        
        # 드래그 이벤트 처리
        elif event_type == "Drag":
            target = event["target"]
            command = f"0#drag, {target}"
            commands.append(command)
            i += 1
            continue
        
        # 키 입력 이벤트 처리
        elif event_type == "Press" and isinstance(event["target"], str):
            key = event["target"].lower()

            special_keys = ["enter", "tab", "esc", "backspace", "space", "up", "down", "left", "right", 
                           "home", "end", "pageup", "pagedown", "delete", "insert"]
            
            if key.startswith("cmd+"):
                # cmd+enter, cmd+tab 등의 조합에서 cmd+ 제거하고 기본 키만 사용
                after_cmd = key[4:]
                
                # 복합 조합 (cmd+shift+...) 처리
                if '+' in after_cmd:
                    parts = after_cmd.split('+')
                    last_part = parts[-1].lower()
                    
                    if last_part in special_keys:
                        key = last_part  # cmd+shift+enter -> enter
                    else:
                        key = after_cmd  # 다른 복합 키는 원래대로 유지
                else:
                    # 단순 조합 (cmd+enter)
                    if after_cmd.lower() in special_keys:
                        key = after_cmd.lower() 

            if key == "win":
                command = f"0#press, win"
                commands.append(command)
                i += 1
                continue

            if "ctrl+" in key:
                if key == "ctrl+c" or "ctrl+\x03" in key:
                    command = f"0#copy,"
                    commands.append(command)
                elif key == "ctrl+v" or "ctrl+\x16" in key:
                    command = f"0#paste,"
                    commands.append(command)
                elif key == "ctrl+s" or "ctrl+\x13" in key:
                    command = f"0#save,"
                    commands.append(command)
                else:
                    # ETX/SYN 같은 컨트롤 문자가 표시되지 않도록 명시적으로 처리
                    ctrl_key = key.split('+')[1] if '+' in key else key
                    # 이스케이프 시퀀스 제거
                    if '\\x' in ctrl_key:
                        if '\\x03' in ctrl_key:
                            ctrl_key = 'c'
                        elif '\\x16' in ctrl_key:
                            ctrl_key = 'v'
                        elif '\\x13' in ctrl_key:
                            ctrl_key = 's'
                        elif '\\x01' in ctrl_key:
                            ctrl_key = 'a'
                        else:
                            # 기타 알려지지 않은 이스케이프 시퀀스는 그냥 글자로 표현
                            ctrl_key = ctrl_key.replace('\\x', '')
                    command = f"0#press, ctrl+{ctrl_key}"
                    commands.append(command)
            else:
                # 엔터, 탭 등의 키 처리
                if key == "enter":
                    command = f"0#press, enter"
                elif key == "tab":
                    command = f"0#press, tab"
                else:
                    command = f"0#press, {key}"
                
                commands.append(command)
            i += 1
            continue

        elif event_type in ["Click", "Rightclick", "Doubleclick"]:
            target = event["target"]
            control_type = event.get("control_type", "")
            
            if target and not "Error" in str(target):
                assist_bit = "0"
                
                command_name = event_type.lower()
                
                if event_type == "Rightclick":
                    if control_type and control_type != "Unknown" and not isinstance(control_type, tuple):
                        command = f"{assist_bit}#{command_name}, {target} ({control_type}) - rightclick, target object > move focus, to child window(menu)"
                    else:
                        command = f"{assist_bit}#{command_name}, {target} - rightclick, target object > move focus, to child window(menu)"
                else:
                    if not control_type or control_type == "Unknown" or isinstance(control_type, tuple):
                        command = f"{assist_bit}#{command_name}, {target}"
                    else:
                        command = f"{assist_bit}#{command_name}, {target} ({control_type})"
                
                commands.append(command)
            
            i += 1
            continue
        
        # 기타 이벤트 처리
        else:
            target = event["target"]
            assist_bit = "0"
            command_name = event_type.lower()
            
            # control_type이 있는 경우에만 포함
            control_type = event.get("control_type", "")
            if control_type and control_type != "Unknown":
                command = f"{assist_bit}#{command_name}, {target} ({control_type})"
            else:
                command = f"{assist_bit}#{command_name}, {target}"
            
            commands.append(command)
            i += 1
    
    return commands


def convert_logs(raw_log_file, output_file, dictionary_path="task_dictionary.xlsx"):
    """
    로그 파일을 변환하여 GlobalPlanner 형식으로 출력
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    event_dict = load_event_dictionary(dictionary_path)
    
    # 로그 파싱
    print(f"로그 파일 파싱 중...")
    events = parse_raw_logs(raw_log_file)
    
    # 명령어 생성
    commands = generate_command_list(events, event_dict)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Subtasks:\n")
            for i, command in enumerate(commands, 1):
                f.write(f"{i}. {command}\n")
        print(f"✅ 변환된 로그 저장 완료: '{output_file}'")
        
        return True
    except Exception as e:
        print(f"❌ 로그 저장 오류: {e}")
        return False

def process_logs(dictionary_path="task_dictionary.xlsx"):
    """
    변환되지 않은 모든 로그 파일을 변환
    """
    unprocessed_logs = get_unprocessed_log_files()

    if not unprocessed_logs:
        print("✅ 모든 로그 파일이 이미 변환되었습니다!")
        return True

    for raw_log_file in unprocessed_logs:
        output_file = generate_output_filename(raw_log_file)

        print(f"🔍 로그 파일 처리 중: {raw_log_file} -> {output_file}")

        if convert_logs(raw_log_file, output_file, dictionary_path):
            print(f"✅ 로그 변환 완료: {raw_log_file} -> {output_file}")
        else:
            print(f"❌ 변환 실패: {raw_log_file}")

    print("모든 변환되지 않은 로그 파일이 변환되었습니다!")
    return True

if __name__ == "__main__":
    process_logs()