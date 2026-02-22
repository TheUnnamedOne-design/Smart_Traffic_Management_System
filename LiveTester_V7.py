import cv2
import numpy as np
import keyboard
import time
import threading
import pygetwindow as gw
import win32gui
import win32ui
import win32con
import ctypes
from ultralytics import YOLO

# Configuration
# Path to the specialized V7 Native YOLO model
MODEL_PATH = 'vehicle_detector_v7_native.pt' 
RUNNING = False
LOCK = threading.Lock()

def capture_window_ctypes(window_title):
    hwnd = win32gui.FindWindow(None, window_title)
    if not hwnd:
        windows = gw.getWindowsWithTitle(window_title)
        if windows: hwnd = windows[0]._hWnd
        else: return None
    try:
        rect = win32gui.GetWindowRect(hwnd)
        w, h = rect[2] - rect[0], rect[3] - rect[1]
        hwndDC = win32gui.GetWindowDC(hwnd)
        mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()
        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
        saveDC.SelectObject(saveBitMap)
        ctypes.windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 2)
        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)
        img = np.frombuffer(bmpstr, dtype='uint8')
        img.shape = (bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4)
        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwndDC)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    except Exception: return None

def detection_loop():
    global RUNNING
    try:
        import traceback
        print("\n--- V7 NATIVE YOLO TINYML DETECTOR ---")
        
        # Check if model exists
        if not os.path.exists(MODEL_PATH):
            print(f"Error: {MODEL_PATH} not found. Running training first?")
            return

        # Load weights
        model = YOLO(MODEL_PATH, task='detect')
        
        win_name = "V7 NATIVE TinyML Detector"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1.0)

        while True:
            with LOCK:
                if not RUNNING: break
            
            frame = capture_window_ctypes('Google Chrome')
            if frame is None: frame = capture_window_ctypes('YouTube')
            if frame is None:
                time.sleep(1)
                continue
                
            # Run Native Inference
            results = model(frame, verbose=False, conf=0.3)[0]
            
            display_frame = frame.copy()
            count = 0
            
            for box in results.boxes:
                b = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0]
                label = f"V7 {conf:.2f}"
                
                # Draw tight boxes
                cv2.rectangle(display_frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cv2.putText(display_frame, label, (b[0], b[1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                count += 1

            # HUD
            cv2.rectangle(display_frame, (0, 0), (280, 60), (0,0,0), -1)
            cv2.putText(display_frame, f"VEHICLES: {count}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow(win_name, display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
                    
        cv2.destroyAllWindows()
        print("[V7 Stopped]")
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR in Detection Loop]: {e}")
        import traceback
        traceback.print_exc()
        RUNNING = False

def toggle():
    global RUNNING
    with LOCK:
        if RUNNING:
            RUNNING = False
        else:
            RUNNING = True
            threading.Thread(target=detection_loop, daemon=True).start()

import os
print("--- Native YOLO TinyML V7 Live Tester ---")
print("Press 'Ctrl + Shift + 1' to START/STOP.")
keyboard.add_hotkey('ctrl+shift+1', toggle)
keyboard.wait('esc')
RUNNING = False
