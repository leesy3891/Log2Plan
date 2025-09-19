# GUI_Parser.py
import os
import pygetwindow as gw
from pywinauto.application import Application
import win32gui
import time

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

def get_all_windows_info():
    """현재 열린 모든 윈도우 정보 수집"""
    try:
        windows = gw.getAllWindows()
        return [{"title": w.title, "visible": w.visible, "size": (w.width, w.height)} 
                for w in windows if w.title.strip()]
    except Exception:
        return []

def is_browser_window(hwnd):
    """브라우저 윈도우인지 확인"""
    try:
        class_name = win32gui.GetClassName(hwnd)
        window_text = win32gui.GetWindowText(hwnd)
        
        browser_classes = [
            "Chrome_WidgetWin_1",  # Chrome
            "ApplicationFrameWindow",  # Edge
            "MozillaWindowClass",  # Firefox
        ]
        
        return any(cls in class_name for cls in browser_classes)
    except Exception:
        return False

def get_safe_root_window():
    """안전한 루트 윈도우 확보"""
    try:
        # 1. 현재 포그라운드 윈도우 시도
        hwnd = win32gui.GetForegroundWindow()
        if hwnd and win32gui.IsWindowVisible(hwnd):
            window_text = win32gui.GetWindowText(hwnd)
            if window_text.strip():  # 빈 제목이 아니면
                return hwnd, window_text
        
        # 2. 브라우저 윈도우 우선 탐색
        def enum_windows_proc(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd) and is_browser_window(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title.strip():
                    windows.append((hwnd, title))
            return True
        
        browser_windows = []
        win32gui.EnumWindows(enum_windows_proc, browser_windows)
        
        if browser_windows:
            # 가장 최근 활성화된 브라우저 윈도우 반환
            return browser_windows[0]
        
        # 3. 일반 윈도우 중 제목이 있는 것
        active_window = gw.getActiveWindow()
        if active_window and active_window.title.strip():
            return active_window._hWnd, active_window.title
            
        return None, None
        
    except Exception as e:
        print(f"Error in get_safe_root_window: {e}")
        return None, None

def get_components_from_localenv():
    MAX_ELEMENTS = 5000  # 안전 상한선
    MAX_DEPTH = 6  # 깊이 제한
    
    # 안전한 루트 윈도우 확보
    hwnd, window_title = get_safe_root_window()
    
    if not hwnd:
        print("No valid window found.")
        return {}, {}

    try:
        app = Application(backend="uia").connect(handle=hwnd)
        selected_window = app.window(handle=hwnd)
        selected_window.set_focus()
        
        print(f"Root window: {window_title} (HWND: {hwnd})")
        
    except Exception as e:
        print(f"Error connecting to window: {e}")
        return {}, {}

    def is_actionable_element(element):
        """액션 가능한 요소인지 필터링"""
        try:
            info = element.element_info
            
            # 타입 화이트리스트 - 실제 액션 가능한 요소들만
            actionable_types = {
                "Button", "Edit", "Hyperlink", "MenuItem", "Image", 
                "ListItem", "ComboBox", "CheckBox", "RadioButton",
                "TabItem", "TreeItem", "Text", "Document", "Pane"
            }
            
            if info.control_type not in actionable_types:
                return False
            
            # 가시성 체크
            if not info.visible:
                return False
                
            # 좌표 유효성 체크
            try:
                rect = element.rectangle()
                if not rect or rect.width() <= 1 or rect.height() <= 1:
                    return False
                    
                # 화면 범위 체크 (음수 좌표나 너무 큰 좌표 제외)
                if rect.left < -100 or rect.top < -100 or rect.left > 5000 or rect.top > 5000:
                    return False
                    
            except Exception:
                return False
                
            return True
            
        except Exception:
            return False

    def get_descendants_safe(element, current_depth=0):
        """안전한 하위 요소 탐색 (깊이 제한 포함)"""
        if current_depth >= MAX_DEPTH:
            return []
            
        descendants = []
        try:
            children = element.children()
        except Exception:
            return descendants

        for child in children:
            # 중복 방지를 위한 기본 체크
            if is_actionable_element(child):
                descendants.append(child)
                
            # 재귀 호출 (깊이 제한)
            if current_depth < MAX_DEPTH - 1:
                # 특정 컨테이너 타입에서만 더 깊이 들어감
                container_types = {"Pane", "Group", "List", "Tree", "Tab", "Document"}
                if child.element_info.control_type in container_types:
                    descendants.extend(get_descendants_safe(child, current_depth + 1))
                    
            # 안전 상한선 체크
            if len(descendants) >= MAX_ELEMENTS:
                print(f"WARNING: Reached MAX_ELEMENTS ({MAX_ELEMENTS}), stopping collection")
                break
                
        return descendants

    max_retries = 2
    wait_interval = 1.0

    for attempt in range(max_retries):
        try:
            if selected_window.exists() and selected_window.is_visible():
                elements = get_descendants_safe(selected_window)
                print(f"Found {len(elements)} elements (depth limit: {MAX_DEPTH})")
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
    
    # 중복 제거를 위한 세트
    seen_elements = set()

    for element in elements:
        try:
            info = element.element_info
            name = info.name or "(No Name)"
            
            # 중복 체크 (이름, 타입, 대략적 위치로)
            try:
                rect = element.rectangle()
                element_signature = (name, info.control_type, 
                                   int(rect.left/10)*10, int(rect.top/10)*10)  # 10픽셀 단위로 반올림
                if element_signature in seen_elements:
                    continue
                seen_elements.add(element_signature)
            except Exception:
                # 좌표를 얻을 수 없으면 이름과 타입으로만 중복 체크
                element_signature = (name, info.control_type)
                if element_signature in seen_elements:
                    continue
                seen_elements.add(element_signature)
            
            # 이름 길이 제한
            if len(name) > 20:
                name = name[:10] + "..." + name[-10:]
                
            type_ = info.control_type or "(No Type)"
            
            try:
                rect = element.rectangle()
                if rect:
                    W = (rect.left + rect.right) / 2
                    H = (rect.top + rect.bottom) / 2
                    coords = (W, H)
                else:
                    coords = "(No Coordinates)"
            except Exception:
                coords = "(No Coordinates)"

            # 가시성에 따른 분류
            if info.visible:
                components[idx_visible] = [element, name, type_, coords]
                idx_visible += 1
            else:
                next_page_dict[idx_hidden] = [element, name, type_, coords]
                idx_hidden += 1

        except Exception as e:
            print(f"Error processing element: {e}")
            continue

    return components, next_page_dict

def GUI_Parser(path):
    """GUI 파싱 메인 함수 - 메타데이터 포함"""
    try:
        components, next_page_dict = get_components_from_localenv()
        comp_type = 'pwa'
        
        # 메타데이터 수집
        active_window = gw.getActiveWindow()
        active_window_title = active_window.title if active_window else "Unknown"
        
        all_windows = get_all_windows_info()
        
        meta_data = {
            "active_window_title": active_window_title,
            "visible_component_count": len(components),
            "hidden_component_count": len(next_page_dict),
            "total_component_count": len(components) + len(next_page_dict),
            "all_windows_count": len(all_windows),
            "all_windows": all_windows[:10],  # 최대 10개만 저장
            "parsing_method": comp_type,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        print(f"GUI Parser - Active Window: {active_window_title}")
        print(f"GUI Parser - Components: {len(components)} visible, {len(next_page_dict)} hidden")
        
    except Exception as e:
        print("Failed to extract component dictionary using PyWinAuto...")
        print(f"Error: {e}")
        components = {}
        next_page_dict = {}
        comp_type = 'ocr'
        
        # 오류 시 기본 메타데이터
        meta_data = {
            "active_window_title": "Parse Failed",
            "visible_component_count": 0,
            "hidden_component_count": 0,
            "total_component_count": 0,
            "all_windows_count": 0,
            "all_windows": [],
            "parsing_method": comp_type,
            "error": str(e),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
    return components, next_page_dict, comp_type, meta_data