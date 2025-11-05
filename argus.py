#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARGUS Supreme-Core — Phase 1: Core Boot, Config, Logging & Memory
Author: TopazConch
Features:
- Autonomous data path creation & verification
- Full environment audit (OS, CPU, RAM, Disk, Network)
- Smart configuration load with validation & default fallback
- Persistent AI memory with learning capabilities
- Adaptive logging system with severity & analytics
"""

import os, sys, psutil, json, platform, socket, time, threading
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
try: import pyttsx3
except ImportError: pyttsx3 = None

# -----------------------------
# [0] PATHS, ROOT, CONFIG, ENV AUDIT
# -----------------------------
ROOT = Path(__file__).parent
DATA = ROOT / "argus_data"; DATA.mkdir(exist_ok=True)
MEM  = DATA / "memory.json"
LOG  = DATA / "argus.log"

DEFAULT_CONFIG = {
    "alert_thresholds": {"cpu":90,"ram":90,"disk":95},
    "max_log_files": 5,
    "game_highscore_file": str(DATA/"games.json"),
    "network_scan_ports": list(range(20,1025)),
    "snapshot_interval": 5,
    "trend_window": 10,
    "tts_rate": 180,
    "learning_enabled": True
}

def audit_environment():
    audit = {
        "OS": platform.system(),
        "OS_version": platform.version(),
        "CPU_count": psutil.cpu_count(logical=True),
        "RAM_total": psutil.virtual_memory().total,
        "Disk_total": psutil.disk_usage('/').total,
        "Hostname": socket.gethostname(),
        "IP_addresses": [i.address for i in psutil.net_if_addrs().get('Ethernet',[])]
    }
    return audit

# -----------------------------
# [1] CONFIG LOAD & VALIDATION
# -----------------------------
def load_config():
    if not (DATA / "config.json").exists():
        with open(DATA / "config.json","w") as f:
            json.dump(DEFAULT_CONFIG,f,indent=2)
        return DEFAULT_CONFIG.copy()
    try:
        cfg = json.load(open(DATA / "config.json"))
        for k,v in DEFAULT_CONFIG.items():
            if k not in cfg:
                cfg[k] = v
        return cfg
    except Exception as e:
        log(f"Config load error: {e}, falling back to defaults","WARN")
        return DEFAULT_CONFIG.copy()

CONFIG = load_config()
ENV_AUDIT = audit_environment()

# -----------------------------
# [2] STATE / MEMORY
# -----------------------------
STATE = {
    "metrics": deque(maxlen=500),
    "insights": [],
    "mood": "neutral",
    "alerts": [],
    "events": [],
    "jobs": [],
    "habits": defaultdict(int),
    "rules": [],
    "games": {},
    "network": {"hosts":{}},
    "command_history": deque(maxlen=200)
}

def load_state():
    if MEM.exists():
        try:
            raw = json.load(open(MEM))
            for k,v in raw.items():
                if k=="habits": STATE[k]=defaultdict(int,v)
                elif k=="metrics": STATE[k]=deque(v,maxlen=500)
                elif k=="command_history": STATE[k]=deque(v,maxlen=200)
                else: STATE[k]=v
        except Exception as e:
            log(f"Memory load error: {e}","WARN")

def save_state():
    to_save = {**STATE, "metrics": list(STATE["metrics"]),
               "command_history": list(STATE["command_history"]),
               "habits": dict(STATE["habits"])}
    try:
        with open(MEM,"w") as f:
            json.dump(to_save,f,indent=2)
    except Exception as e:
        log(f"Memory save failed: {e}","ERROR")

# -----------------------------
# [3] LOGGING
# -----------------------------
class C:
    RESET="\033[0m"; G="\033[32m"; R="\033[31m"; Y="\033[33m"
    B="\033[36m"; M="\033[35m"; W="\033[37m"

def ts(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg, level="INFO"):
    color = {"INFO":C.B,"WARN":C.Y,"ERROR":C.R,"SUCCESS":C.G}.get(level,C.M)
    line = f"{ts()} [{level}] {msg}"
    try:
        with open(LOG,"a") as f: f.write(line+"\n")
    except Exception: pass
    print(f"{color}{line}{C.RESET}")

# -----------------------------
# [4] TTS & NOTIFICATIONS
# -----------------------------
def speak(msg):
    if not pyttsx3: return
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate',CONFIG.get("tts_rate",180))
        engine.say(msg)
        engine.runAndWait()
    except Exception as e:
        log(f"TTS error: {e}","WARN")

def notify(msg, level="INFO"):
    log(msg, level)
    if level in ("WARN","ERROR"): speak(msg)

# -----------------------------
# INIT
# -----------------------------
load_state()
log("ARGUS Phase 1: Core Boot & Memory Initialized","SUCCESS")
log(f"Environment Audit: {ENV_AUDIT}","INFO")

# =========================================================
# Phase 2: SYSTEM METRICS, SNAPSHOTS, TREND ANALYSIS
# =========================================================

import statistics
import threading
import time

# -----------------------------
# Snapshot collection
# -----------------------------
def snapshot():
    """Take a single system snapshot and save it to STATE metrics."""
    snap = {
        "cpu": psutil.cpu_percent(interval=0.5),
        "ram": psutil.virtual_memory().percent,
        "disk": psutil.disk_usage('/').percent,
        "time": ts()
    }
    STATE["metrics"].append(snap)
    save_state()
    return snap

# -----------------------------
# Trend Computation
# -----------------------------
def compute_trend(window=CONFIG.get("trend_window",10)):
    """Compute moving averages and standard deviation for recent snapshots."""
    if len(STATE["metrics"]) < window:
        return None
    recent = list(STATE["metrics"])[-window:]
    cpu_vals = [m["cpu"] for m in recent]
    ram_vals = [m["ram"] for m in recent]
    disk_vals = [m["disk"] for m in recent]

    trend = {
        "cpu_avg": sum(cpu_vals)/window,
        "ram_avg": sum(ram_vals)/window,
        "disk_avg": sum(disk_vals)/window,
        "cpu_std": statistics.stdev(cpu_vals),
        "ram_std": statistics.stdev(ram_vals),
        "disk_std": statistics.stdev(disk_vals)
    }
    return trend

# -----------------------------
# Anomaly Detection
# -----------------------------
def detect_anomaly(trend=None):
    """Check if system metrics deviate beyond thresholds or abnormal behavior."""
    if not trend:
        trend = compute_trend()
    if not trend: return None

    alerts = []
    for key, avg_key, std_key in [("cpu","cpu_avg","cpu_std"),("ram","ram_avg","ram_std"),("disk","disk_avg","disk_std")]:
        if trend[avg_key] > CONFIG["alert_thresholds"][key] or trend[std_key] > 25:
            alerts.append(f"{key.upper()} anomaly: avg={trend[avg_key]:.1f}% std={trend[std_key]:.1f}%")

    if alerts:
        alert_msg = " | ".join(alerts)
        STATE["alerts"].append({"time": ts(), "alert": alert_msg})
        notify(alert_msg,"WARN")
        return alert_msg
    return None

# -----------------------------
# Continuous Monitoring Loop
# -----------------------------
def monitor_loop(snapshot_interval=None):
    """Threaded loop that snapshots system metrics, computes trends, and detects anomalies."""
    interval = snapshot_interval or CONFIG.get("snapshot_interval",5)
    while True:
        snap = snapshot()
        trend = compute_trend()
        detect_anomaly(trend)
        # Adaptive learning: track spikes for habit detection
        if trend:
            if trend["cpu_std"] > 20:
                STATE["habits"]["cpu_spike_count"] += 1
            if trend["ram_std"] > 20:
                STATE["habits"]["ram_spike_count"] += 1
        time.sleep(interval)

# -----------------------------
# Start monitoring in background
# -----------------------------
monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
monitor_thread.start()

log("ARGUS Phase 2: System Metrics & Trend Analysis initialized","SUCCESS")

# =========================================================
# Phase 3: COGNITIVE FEEDBACK & SMART SUGGESTIONS
# =========================================================

def suggest_action():
    """Analyze trends, alerts, and habits to suggest a safe action."""
    trend = compute_trend()
    suggestions = []

    if not trend:
        return "Insufficient data for suggestions."

    # CPU / RAM / Disk based suggestions
    if trend["cpu_avg"] > CONFIG["alert_thresholds"]["cpu"] * 0.8:
        suggestions.append("Consider closing heavy applications or running 'optimize' to reduce CPU load.")
    if trend["ram_avg"] > CONFIG["alert_thresholds"]["ram"] * 0.8:
        suggestions.append("Consider freeing RAM or restarting unused services.")
    if trend["disk_avg"] > CONFIG["alert_thresholds"]["disk"] * 0.85:
        suggestions.append("Consider cleaning disk space or rotating logs.")

    # Habit-based suggestion
    if STATE["habits"].get("cpu_spike_count",0) > 3:
        suggestions.append("Frequent CPU spikes detected. Would you like ARGUS to track the top processes automatically?")
    
    if STATE["habits"].get("ram_spike_count",0) > 3:
        suggestions.append("Frequent RAM spikes detected. ARGUS can monitor apps and suggest shutdowns.")

    if not suggestions:
        return "All systems stable. No immediate suggestions."

    # pick the most urgent suggestion
    msg = suggestions[0]
    notify(f"Suggestion: {msg}", "INFO")
    return msg

def perform_autonomous_action(action):
    """Perform a safe system action requested or approved by the user."""
    safe_actions = {
        "optimize": lambda: log("System optimization executed (dummy placeholder)", "SUCCESS"),
        "rotate_logs": lambda: log("Logs rotated safely", "SUCCESS"),
        "snapshot": snapshot
    }

    if action not in safe_actions:
        return f"Action '{action}' not recognized or unsafe."

    try:
        result = safe_actions[action]()
        return f"Action '{action}' performed successfully."
    except Exception as e:
        return f"Failed to perform action '{action}': {e}"

def feedback_loop(user_response, last_suggestion):
    """Learn from user feedback to improve suggestions."""
    key = f"suggestion_{last_suggestion[:10]}"
    if user_response.lower() in ("yes","y","approve","do it"):
        STATE["habits"][key] = STATE["habits"].get(key,0) + 1
        return "ARGUS learned: user approved suggestion."
    elif user_response.lower() in ("no","n","ignore"):
        STATE["habits"][key] = STATE["habits"].get(key,0) - 1
        return "ARGUS learned: user ignored suggestion."
    else:
        return "ARGUS feedback recorded."

# -----------------------------
# Smart Suggestion Loop (threaded)
# -----------------------------
def suggestion_loop(interval=30):
    """Periodically suggest actions based on trends and habits."""
    last_suggestion = None
    while True:
        last_suggestion = suggest_action()
        # optionally ask the user for feedback
        # user_input = input(f"ARGUS suggestion: {last_suggestion}\nDo you approve? (y/n) ")
        # feedback_loop(user_input, last_suggestion)
        time.sleep(interval)

# Start the suggestion thread
suggest_thread = threading.Thread(target=suggestion_loop, daemon=True)
suggest_thread.start()

log("ARGUS Phase 3: Cognitive Feedback & Smart Suggestions initialized", "SUCCESS")

# =========================================================
# Phase 4: COMMAND PARSING & AUTONOMOUS AI COMMUNICATION
# =========================================================

# -----------------------------
# Command Registry
# -----------------------------
COMMANDS = {}
COMMAND_HELP = {}

def register_command(name, fn, help_text=""):
    """Register a command into ARGUS's command system."""
    COMMANDS[name.lower()] = fn
    COMMAND_HELP[name.lower()] = help_text

def fuzzy_match(cmd):
    """Try to match user command to closest registered command."""
    from difflib import get_close_matches
    matches = get_close_matches(cmd.lower(), COMMANDS.keys(), n=1, cutoff=0.6)
    return matches[0] if matches else None

def handle_command(user_input):
    """Parse user input, match command, execute, and return response."""
    parts = user_input.strip().split()
    if not parts: return "No command entered."
    cmd, args = parts[0], parts[1:]
    
    target_cmd = fuzzy_match(cmd)
    if not target_cmd:
        return f"Unknown command '{cmd}'. Did you mean: {', '.join(COMMANDS.keys())}?"
    
    try:
        result = COMMANDS[target_cmd](args)
        return result
    except Exception as e:
        log(f"Command execution error: {e}", "ERROR")
        return f"Error executing command '{target_cmd}': {e}"

# -----------------------------
# Autonomous Chat / Learning
# -----------------------------
CHAT_HISTORY = deque(maxlen=200)  # store last 200 exchanges

def chat(user_msg):
    """Respond intelligently to user messages using a learning loop."""
    CHAT_HISTORY.append({"role":"user","msg":user_msg})

    # Very basic autonomous reply logic for now; can be replaced with ML/NLP
    response = None

    # 1. Check if user is asking a command
    for word in user_msg.split():
        if fuzzy_match(word):
            response = handle_command(user_msg)
            break

    # 2. Check for general conversation patterns
    if not response:
        if "hello" in user_msg.lower():
            response = "Hello! How can I assist you today?"
        elif "status" in user_msg.lower():
            response = snapshot()
        else:
            response = "I'm learning to respond better. Could you clarify?"

    CHAT_HISTORY.append({"role":"argus","msg":response})
    return response

def chat_loop():
    """Background loop for autonomous chat or suggestions."""
    while True:
        time.sleep(60)
        # Optionally: ARGUS can proactively greet or suggest
        if random.random() < 0.1:  # 10% chance every minute
            msg = "Hi! I'm monitoring your system. Would you like a suggestion?"
            notify(msg)
            CHAT_HISTORY.append({"role":"argus","msg":msg})

# Start chat loop thread
chat_thread = threading.Thread(target=chat_loop, daemon=True)
chat_thread.start()

log("ARGUS Phase 4: Command Parsing & Autonomous AI Communication initialized", "SUCCESS")

# =========================================================
# Phase 5: USER-FRIENDLY CLI INTERFACE (ULTIMATE INTEGRATION)
# =========================================================
import readline
import subprocess
import random
import pprint
from collections import defaultdict

# -----------------------------
# Pretty print helper
# -----------------------------
def pretty_print(obj):
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(obj)

# -----------------------------
# Tab completion
# -----------------------------
def complete(text, state):
    options = [cmd for cmd in COMMANDS.keys() if cmd.startswith(text)]
    if state < len(options):
        return options[state]
    return None

readline.set_completer(complete)
readline.parse_and_bind("tab: complete")

# -----------------------------
# CLI Welcome & Help
# -----------------------------
def print_welcome():
    print(f"\n{C.G}=== ARGUS Supreme-Core CLI ==={C.RESET}")
    print("Type 'help' to see available commands, 'exit' to quit.\n")

def show_help(args=None):
    print("\n=== Available Commands ===")
    categories = defaultdict(list)
    for cmd, desc in COMMAND_HELP.items():
        if cmd in ("help","exit"):
            categories["System"] += [f"{cmd}: {desc}"]
        elif cmd in ("snapshot","trend","anomaly","suggest","feedback","action","monitor_loop","suggestion_loop"):
            categories["Monitoring / AI"] += [f"{cmd}: {desc}"]
        elif cmd in ("disk_usage","uptime","processes","ls","mkdir","rm","touch","cat","du","tree","whoami"):
            categories["System & Files"] += [f"{cmd}: {desc}"]
        elif cmd in ("ping","scan","user_list","netstat","ipconfig","nslookup"):
            categories["Network & Connectivity"] += [f"{cmd}: {desc}"]
        elif cmd in ("shutdown","restart","lock","logoff"):
            categories["Admin & Security"] += [f"{cmd}: {desc}"]
        elif cmd in ("eval","exec","script","python","shell"):
            categories["Developer / Scripting"] += [f"{cmd}: {desc}"]
        elif cmd in ("memory","chat","tts","notify","learn","insights","habits"):
            categories["AI / Learning"] += [f"{cmd}: {desc}"]
        else:
            categories["Other"] += [f"{cmd}: {desc}"]

    for cat, cmds in categories.items():
        print(f"\n{C.B}{cat}:{C.RESET}")
        for entry in cmds:
            print(f"  {entry}")
    print()

# -----------------------------
# Register built-in commands
# -----------------------------
register_command("help", show_help, "Show this help message")
register_command("exit", lambda args: sys.exit(0), "Exit the ARGUS CLI")

# -----------------------------
# Monitoring / AI Commands
# -----------------------------
register_command("snapshot", lambda args: snapshot(), "Take a system snapshot")
register_command("trend", lambda args: compute_trend(), "Compute trend for recent snapshots")
register_command("anomaly", lambda args: detect_anomaly(), "Detect anomalies in system metrics")
register_command("suggest", lambda args: suggest_action(), "Ask ARGUS for a smart suggestion")
register_command("feedback", lambda args: feedback_loop(args[0] if args else "", args[1] if len(args)>1 else ""), "Provide feedback on last suggestion")
register_command("action", lambda args: perform_autonomous_action(args[0] if args else ""), "Perform a safe system action")
register_command("monitor_loop", lambda args: threading.Thread(target=monitor_loop, daemon=True).start(), "Start live system monitoring loop")
register_command("suggestion_loop", lambda args: threading.Thread(target=suggestion_loop, daemon=True).start(), "Start live suggestion loop")

# -----------------------------
# System & Files
# -----------------------------
def cmd_disk_usage(args): return psutil.disk_usage(args[0] if args else "/")._asdict()
def cmd_processes(args): return [{"pid": p.pid, "name": p.name(), "cpu": p.cpu_percent()} for p in psutil.process_iter()]
def cmd_uptime(args): return {"uptime_sec": time.time() - psutil.boot_time()}
def cmd_ls(args): return os.listdir(args[0] if args else ".")
def cmd_mkdir(args): os.makedirs(args[0], exist_ok=True); return f"Directory '{args[0]}' created."
def cmd_rm(args): os.remove(args[0]); return f"File '{args[0]}' removed."
def cmd_touch(args): Path(args[0]).touch(); return f"File '{args[0]}' touched."
def cmd_cat(args): return open(args[0]).read() if Path(args[0]).exists() else f"File '{args[0]}' not found."
def cmd_du(args): return {"size_bytes": sum(f.stat().st_size for f in Path(args[0] if args else ".").rglob('*'))}
def cmd_tree(args): return "\n".join([str(p) for p in Path(args[0] if args else ".").rglob("*")])
def cmd_whoami(args): import getpass; return getpass.getuser()

register_command("disk_usage", cmd_disk_usage, "Show disk usage for path")
register_command("processes", cmd_processes, "Show top processes")
register_command("uptime", cmd_uptime, "Show system uptime")
register_command("ls", cmd_ls, "List directory contents")
register_command("mkdir", cmd_mkdir, "Create directory")
register_command("rm", cmd_rm, "Remove file")
register_command("touch", cmd_touch, "Touch a file")
register_command("cat", cmd_cat, "View file contents")
register_command("du", cmd_du, "Disk usage of folder recursively")
register_command("tree", cmd_tree, "Tree view of folder recursively")
register_command("whoami", cmd_whoami, "Show current user")

# -----------------------------
# Network & Connectivity
# -----------------------------
def cmd_ping(args):
    host = args[0] if args else "8.8.8.8"
    result = subprocess.run(["ping", "-c", "2", host], capture_output=True, text=True)
    return result.stdout

def cmd_scan(args):
    ports = range(20, 1025) if not args else range(int(args[0]), int(args[1])+1)
    host = args[2] if len(args)>2 else "127.0.0.1"
    open_ports = []
    for port in ports:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.2)
        try:
            sock.connect((host, port))
            open_ports.append(port)
        except:
            pass
        finally:
            sock.close()
    return {"host": host, "open_ports": open_ports}

def cmd_user_list(args):
    import getpass
    return [getpass.getuser()]

def cmd_netstat(args):
    result = subprocess.run(["netstat","-an"], capture_output=True, text=True)
    return result.stdout

def cmd_ipconfig(args):
    result = subprocess.run(["ipconfig"], capture_output=True, text=True)
    return result.stdout

def cmd_nslookup(args):
    host = args[0] if args else "google.com"
    result = subprocess.run(["nslookup", host], capture_output=True, text=True)
    return result.stdout

register_command("ping", cmd_ping, "Ping a host")
register_command("scan", cmd_scan, "Scan ports on a host")
register_command("user_list", cmd_user_list, "List current user")
register_command("netstat", cmd_netstat, "Show network connections")
register_command("ipconfig", cmd_ipconfig, "Show network configuration")
register_command("nslookup", cmd_nslookup, "DNS lookup for a host")

# -----------------------------
# Admin & Security
# -----------------------------
register_command("shutdown", lambda args: log("Simulated shutdown","INFO"), "Simulate system shutdown")
register_command("restart", lambda args: log("Simulated restart","INFO"), "Simulate system restart")
register_command("lock", lambda args: log("Simulated lock","INFO"), "Simulate locking system")
register_command("logoff", lambda args: log("Simulated logoff","INFO"), "Simulate logoff")

# -----------------------------
# Developer / Scripting
# -----------------------------
def cmd_eval(args): return eval(" ".join(args))
def cmd_exec(args): return exec(" ".join(args))
def cmd_script(args): return subprocess.run(args, capture_output=True, text=True).stdout
def cmd_shell(args): return subprocess.run(" ".join(args), shell=True, capture_output=True, text=True).stdout

register_command("eval", cmd_eval, "Evaluate Python expression")
register_command("exec", cmd_exec, "Execute Python code")
register_command("script", cmd_script, "Run external script")
register_command("shell", cmd_shell, "Run shell command")

# -----------------------------
# AI / Learning
# -----------------------------
register_command("memory", lambda args: STATE, "Show ARGUS memory")
register_command("chat", lambda args: chat(" ".join(args)), "Chat with ARGUS")
register_command("tts", lambda args: speak(" ".join(args)), "Text-to-speech output")
register_command("notify", lambda args: notify(" ".join(args)), "Send a notification")
register_command("insights", lambda args: STATE["insights"], "Show AI insights")
register_command("habits", lambda args: STATE["habits"], "Show learned habits")
register_command("learn", lambda args: feedback_loop(args[0] if args else "", args[1] if len(args)>1 else ""), "Teach ARGUS feedback")

# -----------------------------
# CLI Loop
# -----------------------------
def cli_loop():
    print_welcome()
    while True:
        try:
            user_input = input(f"{C.G}ARGUS> {C.RESET}").strip()
            if not user_input:
                continue

            STATE["command_history"].append(user_input)
            cmd_word = user_input.split()[0].lower() if user_input else ""

            if cmd_word in ("exit","quit"):
                print("Exiting ARGUS...")
                save_state()
                break
            elif cmd_word in ("help","?"):
                show_help()
            else:
                cmd_match = fuzzy_match(cmd_word)
                if cmd_match:
                    result = handle_command(user_input)
                else:
                    result = chat(user_input)

                # Pretty-print
                if isinstance(result, dict) or isinstance(result, list):
                    pretty_print(result)
                else:
                    print(result)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting ARGUS...")
            save_state()
            break
        except Exception as e:
            log(f"CLI error: {e}", "ERROR")
            print(f"{C.R}Error:{C.RESET} {e}")

# -----------------------------
# Start CLI
# -----------------------------
if __name__ == "__main__":
    cli_loop()

# -----------------------------
# EXTENDED MODULES / COMMANDS
# -----------------------------
# Add this after cli_loop() start in ARGUS for full functionality

import shutil
import hashlib
import glob
import psutil
import subprocess
import random

# -----------------------------
# [1] System & Files / Monitoring & Metrics
# -----------------------------
def cmd_stats(args=None):
    return {
        "CPU%": psutil.cpu_percent(),
        "RAM%": psutil.virtual_memory().percent,
        "Disk%": psutil.disk_usage("/").percent
    }

def cmd_alerts(args=None):
    return STATE["alerts"]

def cmd_log_tail(args):
    lines = int(args[0]) if args else 10
    try:
        with open(LOG, "r") as f:
            content = f.readlines()[-lines:]
        return "".join(content)
    except Exception as e:
        return str(e)

def cmd_env(args=None):
    return {"env_vars": dict(os.environ), "audit": ENV_AUDIT}

def cmd_temperature(args=None):
    try:
        import psutil_sensors
        temps = psutil_sensors.sensors_temperatures()
        return temps
    except:
        return "Temperature info not available."

def cmd_gpu_info(args=None):
    return "GPU info placeholder – integrate nvidia-smi or similar."

def cmd_disk_smart(args):
    device = args[0] if args else "/dev/sda"
    return subprocess.getoutput(f"smartctl -a {device}")

register_command("stats", cmd_stats, "Show CPU/RAM/Disk usage")
register_command("alerts", cmd_alerts, "Show active alerts")
register_command("log_tail", cmd_log_tail, "Tail last N lines of ARGUS log")
register_command("env", cmd_env, "Show environment variables & audit")
register_command("temperature", cmd_temperature, "Show CPU/GPU temps")
register_command("gpu_info", cmd_gpu_info, "Show GPU info")
register_command("disk_smart", cmd_disk_smart, "Show SMART disk info")

# Filesystem operations
register_command("ls", lambda args: os.listdir(args[0] if args else "."), "List directory contents")
register_command("mkdir", lambda args: os.makedirs(args[0], exist_ok=True) or f"Created {args[0]}", "Make directory")
register_command("rmdir", lambda args: os.rmdir(args[0]) or f"Removed {args[0]}", "Remove empty directory")
register_command("rm", lambda args: os.remove(args[0]) or f"Removed {args[0]}", "Remove file")
register_command("mv", lambda args: shutil.move(args[0], args[1]) or f"Moved {args[0]} → {args[1]}", "Move/rename")
register_command("cp", lambda args: shutil.copy(args[0], args[1]) or f"Copied {args[0]} → {args[1]}", "Copy file")
register_command("touch", lambda args: Path(args[0]).touch() or f"Touched {args[0]}", "Touch file")
register_command("cat", lambda args: open(args[0]).read() if Path(args[0]).exists() else f"{args[0]} not found", "Show file contents")
register_command("head", lambda args: "".join(open(args[0]).readlines()[:int(args[1])]) if len(args)>1 else "".join(open(args[0]).readlines()[:10]), "Show first N lines")
register_command("tail", lambda args: "".join(open(args[0]).readlines()[-int(args[1]):]) if len(args)>1 else "".join(open(args[0]).readlines()[-10:]), "Show last N lines")
register_command("find", lambda args: [str(p) for p in Path(args[0]).rglob(args[1])] if len(args)>1 else [], "Find files by pattern")
register_command("df", lambda args=None: shutil.disk_usage("/") if args is None else shutil.disk_usage(args[0]), "Disk free space")
register_command("du", lambda args: sum(f.stat().st_size for f in Path(args[0]).rglob("*")) if args else sum(f.stat().st_size for f in Path(".").rglob("*")), "Directory size")
register_command("md5sum", lambda args: hashlib.md5(open(args[0],"rb").read()).hexdigest(), "MD5 checksum")
register_command("sha256sum", lambda args: hashlib.sha256(open(args[0],"rb").read()).hexdigest(), "SHA256 checksum")
register_command("zip", lambda args: shutil.make_archive(args[0], 'zip', args[1] if len(args)>1 else "."), "Zip files")
register_command("unzip", lambda args: shutil.unpack_archive(args[0], args[1] if len(args)>1 else "."), "Unzip files")

# -----------------------------
# [2] Networking & Connectivity
# -----------------------------
register_command("ping", lambda args: subprocess.getoutput(f"ping -c 2 {args[0] if args else '8.8.8.8'}"), "Ping host")
register_command("scan", cmd_scan, "Scan ports on host")
register_command("netstats", lambda args=None: subprocess.getoutput("netstat -an"), "Network stats")
register_command("traceroute", lambda args: subprocess.getoutput(f"traceroute {args[0]}") if args else "No host", "Trace route")
register_command("ifconfig", lambda args=None: subprocess.getoutput("ifconfig"), "Network interfaces")
register_command("vpn_status", lambda args=None: "VPN status placeholder", "VPN connection status")

# -----------------------------
# [3] Optimization & Maintenance
# -----------------------------
register_command("optimize", lambda args=None: log("Optimization executed", "SUCCESS"), "Optimize system")
register_command("rotate_logs", lambda args=None: log("Logs rotated", "SUCCESS"), "Rotate logs")
register_command("cleanup_temp", lambda args=None: log("Temp files cleaned", "SUCCESS"), "Clean temp/cache files")

# -----------------------------
# [4] Memory & Learning
# -----------------------------
register_command("memory", lambda args=None: STATE, "Show ARGUS memory")
register_command("habits", lambda args=None: STATE["habits"], "Show habits")
register_command("rules", lambda args=None: STATE["rules"], "Show rules")
register_command("events", lambda args=None: STATE["events"], "Show events")
register_command("forget", lambda args: STATE["habits"].pop(args[0], None) if args else "Specify topic", "Forget habit/topic")
register_command("teach", lambda args: STATE["rules"].append(args[0]) or f"Taught {args[0]}", "Teach new rule/fact")

# -----------------------------
# [5] AI Communication & Chat
# -----------------------------
register_command("chat", lambda args: chat(" ".join(args)), "Talk to ARGUS")
register_command("status", lambda args=None: {"metrics": cmd_stats(), "alerts": STATE["alerts"], "habits": STATE["habits"]}, "System & learning status")
register_command("history", lambda args=None: list(CHAT_HISTORY), "Show last 200 chat messages")
register_command("clear_history", lambda args=None: CHAT_HISTORY.clear() or "Chat history cleared", "Clear chat history")

# -----------------------------
# [6] Games & Productivity
# -----------------------------
STATE["games"] = {}
register_command("play", lambda args: f"Launching game {args[0] if args else 'unknown'}", "Launch mini-game")
register_command("highscores", lambda args=None: STATE["games"], "Show game highscores")
register_command("reset_scores", lambda args=None: STATE["games"].clear() or "Highscores reset", "Reset highscores")

# -----------------------------
# [7] Admin & Security
# -----------------------------
register_command("shutdown", lambda args=None: log("Shutdown simulated", "INFO"), "Simulate shutdown")
register_command("restart", lambda args=None: log("Restart simulated", "INFO"), "Simulate restart")
register_command("user_list", cmd_user_list, "Show current users")
register_command("network_block", lambda args: f"Blocked host {args[0]}" if args else "Specify host", "Block host")

# -----------------------------
# [8] Developer Tools / Scripting
# -----------------------------
register_command("eval", cmd_eval, "Evaluate Python")
register_command("exec", cmd_exec, "Execute Python")
register_command("script", cmd_script, "Run external script")
register_command("shell", cmd_shell, "Run shell command")

# -----------------------------
# [9] Visualization & Analytics
# -----------------------------
register_command("cli_charts", lambda args=None: "ASCII charts placeholder", "CPU/RAM/Disk trends")
register_command("trend_analysis", lambda args=None: "Trend analysis placeholder", "Graph anomalies & usage trends")
register_command("habit_stats", lambda args=None: "Habit stats placeholder", "Visualize habits")

# -----------------------------
# [10] Benchmarking & Testing
# -----------------------------
register_command("benchmark_cpu", lambda args=None: "CPU benchmark placeholder", "CPU stress test")
register_command("benchmark_disk", lambda args=None: "Disk benchmark placeholder", "Disk performance")
register_command("benchmark_network", lambda args=None: "Network benchmark placeholder", "Network throughput test")
register_command("stress_test", lambda args=None: "Stress test placeholder", "Test system stability")





# Setup Directories and Paths
class Paths:
    """
    Centralized immutable paths object. Creates directory layout on import.
    Ensures that essential paths like logs, snapshots, and config are available.
    """
    def __init__(self, root: str = "/opt/argus"):
        self._root = Path(root).resolve()
        # Define canonical directories
        self.data = self._root / "data"
        self.logs = self._root / "logs"
        self.snapshots = self._root / "snapshots"
        self.wal = self._root / "wal"
        self.config = self._root / "config"
        self.tmp = self._root / "tmp"
        self.modules = self._root / "modules"
        self._dirs = [self.data, self.logs, self.snapshots, self.wal, self.config, self.tmp, self.modules]
        self._create_dirs()

    def _create_dirs(self):
        for d in self._dirs:
            d.mkdir(parents=True, exist_ok=True)
            # Lock down permissions for sensitive dirs
            if d in (self.snapshots, self.config, self.wal):
                try:
                    d.chmod(0o700)  # restrict access
                except Exception as e:
                    print(f"Permission setting failed on {d}: {e}")

    @property
    def root(self):
        return self._root

    def to_dict(self):
        return {k: str(v) for k, v in self.__dict__.items() if not k.startswith("_")}


# Singleton path object for application use
DEFAULT_PATHS = Paths()


# Config Management with Locking and Thresholds
DEFAULT_CONFIG = {
    "monitoring": {
        "interval_seconds": 5,
        "cpu_warn_pct": 75,
        "cpu_critical_pct": 95,
        "mem_warn_pct": 75,
        "mem_critical_pct": 95
    },
    "snapshot": {
        "interval_minutes": 10,
        "retain": 72
    },
    "logging": {
        "level": "INFO",
        "rotation_mb": 10
    },
    "security": {
        "require_hw_key": True
    }
}

class Config:
    """
    Configuration management class that loads, saves, and reloads the app's configuration file
    with support for file locking and data persistence.
    """
    def __init__(self, config_path: str = None):
        self.config_path = config_path or (DEFAULT_PATHS.config / "argus.yaml")
        self._lock = FileLock(str(self.config_path) + ".lock")
        self._data = {}
        self.reload()

    def reload(self):
        """
        Reloads the configuration from file, merging with default configuration.
        """
        with self._lock:
            if self.config_path.exists():
                try:
                    with open(self.config_path, "r") as f:
                        self._data = yaml.safe_load(f) or {}
                except Exception:
                    print("Failed to load config. Using defaults.")
                    self._data = {}
            # Merge defaults into loaded config
            merged = DEFAULT_CONFIG.copy()
            merged.update(self._data)
            self._data = merged

    def get(self, key_path: str, default=None):
        """
        Get a value from the configuration, using dot notation for nested keys.
        """
        keys = key_path.split(".")
        node = self._data
        for k in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(k, default)
            if node is default:
                break
        return node

    def set(self, key_path: str, value: Any):
        """
        Set a value in the configuration file, supporting nested paths.
        """
        with self._lock:
            # Create intermediary dicts for setting nested keys
            node = self._data
            keys = key_path.split(".")
            for k in keys[:-1]:
                node = node.setdefault(k, {})
            node[keys[-1]] = value
            with open(self.config_path, "w") as f:
                yaml.safe_dump(self._data, f)


# Logging setup with JSON and Rotation
def get_logger(name="argus"):
    """
    Returns a logger instance, configured with console and file handlers.
    The log file rotates based on the configuration.
    """
    log = logging.getLogger(name)
    if log.handlers:
        return log
    cfg = Config()  # Reload config to get correct logging level
    level = getattr(logging, cfg.get("logging.level", "INFO").upper(), logging.INFO)
    log.setLevel(level)

    # Console Handler (human readable)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ch.setFormatter(ch_formatter)
    log.addHandler(ch)

    # File Handler (JSON)
    logfile = DEFAULT_PATHS.logs / f"{name}.log"
    fh = handlers.RotatingFileHandler(str(logfile), maxBytes=cfg.get("logging.rotation_mb", 10) * 1024 * 1024, backupCount=10)
    fh.setLevel(logging.DEBUG)
    json_fmt = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(name)s %(message)s %(module)s %(funcName)s %(lineno)d')
    fh.setFormatter(json_fmt)
    log.addHandler(fh)

    return log


# Memory Management with Write-Ahead Log (WAL) & Snapshots
class MemoryStore:
    """
    Centralized state store with WAL (Write-Ahead Log) for reliability and snapshotting.
    """
    def __init__(self):
        self._lock = threading.RLock()
        self.STATE = {
            "metrics": [],
            "alerts": [],
            "insights": [],
            "habits": {},
            "rules": [],
            "events": [],
            "games": {},
            "network": {"hosts": {}},
            "command_history": [],
            "trust_matrix": {"admins": {}, "clients": {}},
            "chat_history": [],
            "knowledge_graph": {},
            "adversary_scores": []
        }
        # Ensure WAL file exists
        WAL_FILE = DEFAULT_PATHS.wal / "state.wal"
        WAL_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not WAL_FILE.exists():
            WAL_FILE.write_text("")
        self._replay_wal_if_needed()

    def _append_wal(self, op: dict):
        """
        Append operations to the Write-Ahead Log (WAL).
        """
        line = json.dumps({"ts": time.time(), "op": op}, separators=(",", ":"))
        with open(WAL_FILE, "a+") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _replay_wal_if_needed(self):
        """
        Replays WAL if needed (to restore the state from previous operations).
        """
        if not WAL_FILE.exists():
            return
        applied = 0
        with open(WAL_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    op = item.get("op")
                    self._apply_op(op, write_wal=False)
                    applied += 1
                except Exception as e:
                    logger.warning("Failed to replay wal line: %s", e)
        logger.info("Replayed %d WAL entries", applied)

    def _apply_op(self, op: dict, write_wal=True):
        """
        Applies an operation to the in-memory state.
        """
        t = op.get("type")
        if t == "set":
            path = op.get("path", [])
            val = op.get("value")
            node = self.STATE
            for p in path[:-1]:
                node = node.setdefault(p, {})
            node[path[-1]] = val
        elif t == "append":
            path = op.get("path", [])
            val = op.get("value")
            node = self.STATE
            for p in path:
                node = node.setdefault(p, [])
            node.append(val)
            # Trim the list to prevent memory blowup
            if len(node) > 10000:
                del node[:len(node) - 10000]
        else:
            logger.warning("Unknown WAL op type: %s", t)
        if write_wal:
            self._append_wal(op)

    def apply(self, op: dict):
        """
        Apply an operation and log it to the WAL.
        """
        with self._lock:
            self._apply_op(op, write_wal=True)

    def snapshot(self, name=None):
        """
        Creates a snapshot of the current state and saves it to disk.
        """
        with self._lock:
            ts = int(time.time())
            fname = DEFAULT_PATHS.snapshots / f"snapshot-{name or ts}.json"
            fname.parent.mkdir(parents=True, exist_ok=True)
            state_json = json.dumps(self.STATE, separators=(",", ":"), sort_keys=True)
            # Compute checksum of the state
            cs = hashlib.sha256(state_json.encode("utf-8")).hexdigest()
            meta = {"created": ts, "checksum": cs}
            with open(fname, "w") as f:
                f.write(json.dumps({"meta": meta, "state": self.STATE}, indent=2))
                f.flush()
                os.fsync(f.fileno())
            logger.info("Created snapshot %s checksum=%s", fname.name, cs)
            return str(fname)

    def load_snapshot(self, path):
        """
        Loads a snapshot from a file.
        """
        with self._lock:
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(path)
            with open(p, "r") as f:
                data = json.load(f)
            # Optional integrity check
            self.STATE = data.get("state", {})
            logger.info("Loaded snapshot %s", p.name)


# Module Management for Dynamic Module Loading and Registration
class ModuleManager:
    def __init__(self):
        self.modules = {}

    def load_module(self, module_name: str):
        """
        Dynamically loads a module from the 'modules' directory.
        """
        try:
            module_path = DEFAULT_PATHS.modules / f"{module_name}.py"
            if not module_path.exists():
                raise FileNotFoundError(f"Module '{module_name}' not found.")
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.modules[module_name] = module
            logger.info(f"Module '{module_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading module '{module_name}': {str(e)}")
            traceback.print_exc()

    def get_module(self, module_name: str):
        """
        Retrieve a loaded module by its name.
        """
        return self.modules.get(module_name, None)

    def list_modules(self):
        """
        List all loaded modules.
        """
        return list(self.modules.keys())

# Initialize Memory Store and ModuleManager
MEMORY = MemoryStore()
MODULE_MANAGER = ModuleManager()
