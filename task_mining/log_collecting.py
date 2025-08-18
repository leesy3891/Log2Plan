import os
import time
from pynput import mouse, keyboard
from pywinauto import Application
from pygetwindow import getActiveWindow
import tkinter as tk
import pygetwindow as gw
import psutil

# 설정 가능한 옵션
TYPING_THRESHOLD = 0.5  # 연속 타이핑 문자열 종료 간격(초)

# Modifier Key 및 현재 눌려 있는 키 상태 추적
pressed_keys = set()
click_positions = {}  # 클릭한 좌표를 저장
stop_tracking = False  # ESC 키 입력 감지 변수
selected_window = None  # 전역 변수로 사용

# 초기화
current_scroll_x = 0
current_scroll_y = 0

# 로그 저장 디렉토리 설정
action_logs_dir = "action_logs"
os.makedirs(action_logs_dir, exist_ok=True)

# 화면 크기 가져오기
def get_screen_dimensions():
    root = tk.Tk()
    root.withdraw()
    return root.winfo_screenwidth(), root.winfo_screenheight()

screen_width, screen_height = get_screen_dimensions()

def get_ui_component_from_click(x, y, button):
    global selected_window
    try:
        active_windows = gw.getWindowsWithTitle('')  # 모든 활성 윈도우 가져오기
        active_window = gw.getActiveWindow()
        if not active_window:
            return None, "Unknown UI Elements", "Unknown"

        app = Application(backend="uia").connect(handle=active_window._hWnd)
        new_selected_window = app.window(handle=active_window._hWnd)

        window_name = new_selected_window.window_text()
        change_type = None

        # 활성 윈도우 리스트에서 현재 창의 인덱스 가져오기
        window_indices = {win.title: i for i, win in enumerate(active_windows) if win.title}
        current_index = window_indices.get(window_name, -1)
        previous_index = window_indices.get(selected_window[0], -1) if selected_window else -1

        if selected_window is None:
            change_type = "windowchange"  # 첫 실행 시 윈도우 변경으로 간주
        elif current_index != previous_index:
            change_type = "windowchange"  # 다른 애플리케이션 창으로 변경됨 (인덱스 기준)
        elif new_selected_window.window_text() != selected_window[0]:
            change_type = "screenchange"  # 같은 앱 내에서 화면 제목 변경됨
            
        selected_window = [window_name, change_type or "Unknown"]

        new_selected_window.set_focus()

        if button == mouse.Button.right:
            selected_window = [window_name, change_type or "Unknown"]
            new_selected_window = new_selected_window.child_window(control_type="Menu", found_index=0)

        all_elements = list(new_selected_window.descendants(depth=None))
        for child_window in new_selected_window.children():
            all_elements.extend(child_window.descendants(depth=None))

        matching_elements = []
        closest_element = None
        closest_distance = float('inf')

        for element in all_elements:
            rect = element.rectangle()
            if rect.left <= x <= rect.right and rect.top <= y <= rect.bottom:
                element_name = element.element_info.name or "Unknown UI Element"
                matching_elements.append(element_name)

                center_x = (rect.left + rect.right) / 2
                center_y = (rect.top + rect.bottom) / 2
                distance = ((center_x - x) ** 2 + (center_y - y) ** 2) ** 0.5

                if distance < closest_distance:
                    closest_distance = distance
                    closest_element = element

        control_type = closest_element.element_info.control_type if closest_element else "Unknown"
        return closest_element, closest_element.element_info.name, control_type
    except Exception as e:
        return None, str(e), "Error"


def track_user_actions():
    global stop_tracking
    
    print("Tracking user actions... (Press ESC to stop)")
    action_log = []
    
    def log_event(event_type, details):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"[{timestamp}] {event_type}: {details}")
        action_log.append(f"[{timestamp}] {event_type}: {details}\n")

    def on_click(x, y, button, pressed): 
        if stop_tracking:
            return False

        try:
            event_type = "Mouse"
            element, ui_name, control_type = get_ui_component_from_click(x, y, button)
            btn_type = "Right" if button == mouse.Button.right else "Left"

            if pressed:
                click_positions[button] = (x, y)
                log_event(event_type, [btn_type, "Pressed", ui_name, control_type, selected_window[0] if selected_window else "Unknown Window", selected_window[1] if selected_window else "Unknown"])
            else:
                release_pos = (x, y)
                press_pos = click_positions.get(button, release_pos)
                
                # 드래그 이벤트로 인식할 조건 추가
                if press_pos != release_pos and hasattr(element, "drag_mouse_input"):
                    log_event(event_type, [btn_type, "Dragged", ui_name, press_pos, release_pos])  # Pressed - Drag가 한 쌍
                else:
                    log_event(event_type, [btn_type, "Released", ui_name, control_type, selected_window[0] if selected_window else "Unknown Window", selected_window[1] if selected_window else "Unknown"]) # Pressed - Release가 한 쌍 = Click

        except Exception as e:
            log_event("Error", f"Error during mouse click: {e}")

    def on_scroll(x, y, dx, dy):
        global current_scroll_x, current_scroll_y

        if stop_tracking: 
            return False

        current_scroll_x += dx
        current_scroll_y += dy
        log_event("Mouse", ["Scroll", current_scroll_x, current_scroll_y, dx, dy])

    def on_press(key):
        global stop_tracking
        
        try:
            key_name = key.char if isinstance(key, keyboard.KeyCode) else key.name if isinstance(key, keyboard.Key) else None
            if key_name:
                pressed_keys.add(key_name)
            if key_name == "esc":
                print("ESC key detected. Stopping tracking...")
                stop_tracking = True
                return False
        except Exception as e:
            log_event("Error", f"Error on press: {e}")

    def on_release(key):
        if stop_tracking:
            return False
        try:
            key_name = key.char if isinstance(key, keyboard.KeyCode) else key.name if isinstance(key, keyboard.Key) else None
            if key_name and key_name in pressed_keys:
                pressed_keys.remove(key_name)
                valid_keys = '+'.join(sorted(pressed_keys))
                log_event("Keyboard", ["pressed", f"{valid_keys}+{key_name}" if valid_keys else f"{key_name}", selected_window[0] if selected_window else "Unknown Window", selected_window[1] if selected_window else "Unknown"])
                selected_window[1]= "Unknown"
        except Exception as e:
            log_event("Error", f"Error on release: {e}")

    with mouse.Listener(on_click=on_click, on_scroll=on_scroll) as mouse_listener, keyboard.Listener(on_press=on_press, on_release=on_release) as keyboard_listener:
        keyboard_listener.join()
        mouse_listener.join()
    
    return action_log 

def save_log_file(log):
    filename = os.path.join(action_logs_dir, f"action_log_{int(time.time())}.txt")
    try:
        with open(filename, "w", encoding="utf-8") as file:
            file.writelines(log)
        print(f"Logs saved to '{filename}'")
    except Exception as e:
        print(f"Error saving log file: {e}")

def main():
    print("Event tracking started. (Press ESC to stop)")
    action_log = track_user_actions()
    if action_log:
        save_log_file(action_log)
    else:
        print("No actions recorded.")

if __name__ == "__main__":
    main()
