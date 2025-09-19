# GUI_Parser.py
import os
import pygetwindow as gw
from pywinauto.application import Application

                                
os.getenv("GOOGLE_API_KEY")

def get_top_window_resolution():
    """Get the resolution of the top active window.""" 
    top_window = gw.getActiveWindow()

    if top_window is not None:
        window_width = top_window.width
        window_height = top_window.height
        return window_width, window_height
    else:
        return None


def get_components_from_localenv():
    # 현재 가장 최근 실행된 윈도우에 포커스
    try:
        window = gw.getActiveWindow()
        if not window:
            raise Exception("No active window found.")
        window.activate()
    except Exception as e:
        print(f"Error activating window: {e}")
        return {}, {}

    import win32gui
    import time

    hwnd = win32gui.GetForegroundWindow()

    try:
        app = Application(backend="uia").connect(handle=hwnd)
        selected_window = app.window(handle=hwnd)
        selected_window.set_focus()
    except Exception as e:
        print(f"Error connecting to window: {e}")
        return {}, {}

    def get_descendants(element):
        descendants = []
        try:
            children = element.children()
        except Exception as e:
            print(f"Error getting children: {e}")
            children = []

        for child in children:
            # 하위 요소가 보이든 아니든 모두 수집
            descendants.append(child)
            if child.element_info.control_type in ["ListItem", "MenuItem"]:
                continue
            descendants.extend(get_descendants(child))
        return descendants

    max_retries = 2
    wait_interval = 1.0

    for attempt in range(max_retries):
        try:
            if selected_window.exists() and selected_window.is_visible():
                elements = get_descendants(selected_window)
                print(f"Found {len(elements)} elements")
                break
        except Exception as inner_e:
            print(f"Retry {attempt + 1}: {inner_e}")
            time.sleep(wait_interval)
    else:
        print("Error: Could not fetch elements after retries.")
        return {}, {}

    components = {}
    next_page_dict = {}
    idx_visible = 0
    idx_hidden = 0

    for index, element in enumerate(elements):
        if element.element_info.name:
            try:
                info = element.element_info
                name = element.element_info.name or "(No Name)"
                if len(name) > 20:
                    name = name[:10] + "..." + name[-10:]
                type_ = info.control_type or "(No Type)"
                rect = element.rectangle()
                if rect:
                    W = (rect.left + rect.right) / 2
                    H = (rect.top + rect.bottom) / 2
                    coords = (W, H)
                else:
                    coords = "(No Coordinates)"

                # 분기: 현재 화면에 보이는지 여부
                if info.visible:
                    components[idx_visible] = [element, name, type_, coords]
                    idx_visible += 1
                else:
                    next_page_dict[idx_hidden] = [element, name, type_, coords]
                    idx_hidden += 1

            except Exception as e:
                print(f"Error processing element {index}: {e}")

    return components, next_page_dict


def GUI_Parser(path):
    try:
        components, next_page_dict = get_components_from_localenv()
        comp_type= 'pwa'
    except Exception as e:
        #components, next_page_dict = get_components_from_ocr(path)
        components = None
        next_page_dict = None
        print("Failed to extract component dictionary using PyWinAuto...")
        comp_type= 'ocr'
    return components, next_page_dict, comp_type