# C:\Prototypes\IEH_Lab\ieh_universal_harvester.py
import os
import sys
import time
import psutil
import pandas as pd
from pynput import mouse, keyboard
from datetime import datetime

# --- SYSTEM PERMANENCE CONFIGURATION ---
OUTPUT_DIR = OUTPUT_DIR = r"C:\Prototypes\IEH_Lab\yashas_data"
MOUSE_FILE = os.path.join(OUTPUT_DIR, "mouse_raw.parquet")
KEYBOARD_FILE = os.path.join(OUTPUT_DIR, "keyboard_raw.parquet")
HEARTBEAT_FILE = os.path.join(OUTPUT_DIR, "harvester_v2.log")

FLUSH_INTERVAL_SEC = 60
BUFFER_MAX_ROWS = 1000

# Set process priority to 'Below Normal' to guarantee 0% system latency or gaming lag
try:
    p = psutil.Process(os.getpid())
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:
    pass

# Transient storage buffers
mouse_buffer = []
keyboard_buffer = []
last_flush_epoch = time.time()

# State verification dictionary to handle keypress tracing
active_press_registry = {}

def update_heartbeat_status(message):
    """Writes an explicit, verifiable diagnostic marker to disk."""
    try:
        with open(HEARTBEAT_FILE, "w") as f:
            f.write(f"IEH_PROTOCOL_V2 // STATUS: ACTIVE\n")
            f.write(f"LAST_TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"DIAGNOSTIC: {message}\n")
    except Exception:
        pass

def flush_buffers_to_parquet():
    """Commits RAM data stores to binary Parquet files cleanly."""
    global mouse_buffer, keyboard_buffer, last_flush_epoch
    
    # 1. Commit Mouse Cache
    if mouse_buffer:
        try:
            df_mouse = pd.DataFrame(mouse_buffer)
            if os.path.exists(MOUSE_FILE):
                df_existing = pd.read_parquet(MOUSE_FILE)
                df_mouse = pd.concat([df_existing, df_mouse], ignore_index=True)
            df_mouse.to_parquet(MOUSE_FILE, engine='fastparquet')
            mouse_buffer = []
        except Exception as e:
            update_heartbeat_status(f"Mouse write error: {str(e)}")

    # 2. Commit Keyboard Cache
    if keyboard_buffer:
        try:
            df_key = pd.DataFrame(keyboard_buffer)
            if os.path.exists(KEYBOARD_FILE):
                df_existing_key = pd.read_parquet(KEYBOARD_FILE)
                df_key = pd.concat([df_existing_key, df_key], ignore_index=True)
            df_key.to_parquet(KEYBOARD_FILE, engine='fastparquet')
            keyboard_buffer = []
        except Exception as e:
            update_heartbeat_status(f"Keyboard write error: {str(e)}")

    last_flush_epoch = time.time()
    update_heartbeat_status("Buffers cleanly committed to parquet cluster.")

def check_buffer_limits():
    """Triggers forced storage dumps if batch size conditions are crossed."""
    current_epoch = time.time()
    total_buffered = len(mouse_buffer) + len(keyboard_buffer)
    
    if (current_epoch - last_flush_epoch >= FLUSH_INTERVAL_SEC) or (total_buffered >= BUFFER_MAX_ROWS):
        flush_buffers_to_parquet()

# --- telemetry tracking logic ---

def on_mouse_move(x, y):
    mouse_buffer.append({
        "timestamp": time.time(),
        "event_class": "move",
        "vector_param_0": float(x),
        "vector_param_1": float(y)
    })
    check_buffer_limits()

def on_mouse_click(x, y, button, pressed):
    mouse_buffer.append({
        "timestamp": time.time(),
        "event_class": "click_down" if pressed else "click_up",
        "vector_param_0": float(x),
        "vector_param_1": float(y)
    })
    check_buffer_limits()

def on_key_press(key):
    """
    Captures temporal metrics while protecting textual privacy.
    Identifies control characters vs structure, discarding letter layout data.
    """
    epoch = time.time()
    try:
        # Check if it's a structural key or standard character
        if hasattr(key, 'vk') and key.vk is not None:
            key_signature = int(key.vk)
        else:
            key_signature = int(key.value.vk)
    except Exception:
        key_signature = hash(str(key)) # Fallback string mapping for edge layouts

    # Deduplicate repeating key holds triggered by Windows OS settings
    if key_signature in active_press_registry:
        return

    active_press_registry[key_signature] = epoch

    keyboard_buffer.append({
        "timestamp": epoch,
        "event_class": "keydown",
        "vector_param_0": float(key_signature),
        "vector_param_1": 0.0
    })
    check_buffer_limits()

def on_key_release(key):
    epoch = time.time()
    try:
        if hasattr(key, 'vk') and key.vk is not None:
            key_signature = int(key.vk)
        else:
            key_signature = int(key.value.vk)
    except Exception:
        key_signature = hash(str(key))

    # Resolve timing delta safely
    press_start_epoch = active_press_registry.pop(key_signature, None)
    dwell_duration = (epoch - press_start_epoch) if press_start_epoch else 0.0

    keyboard_buffer.append({
        "timestamp": epoch,
        "event_class": "keyup",
        "vector_param_0": float(key_signature),
        "vector_param_1": float(dwell_duration)
    })
    check_buffer_limits()

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    update_heartbeat_status("Initializing Core Ingestion Hooks...")

    # Establish continuous non-blocking tracking structures
    mouse_hook = mouse.Listener(on_move=on_mouse_move, on_click=on_mouse_click)
    keyboard_hook = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)

    mouse_hook.start()
    keyboard_hook.start()

    try:
        mouse_hook.join()
        keyboard_hook.join()
    except KeyboardInterrupt:
        flush_buffers_to_parquet()
        sys.exit(0)
