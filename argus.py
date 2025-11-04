#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARGUS Supreme-Core Full Suite — Ultimate vFull
Author: TopazConch
Phase: 0-15 Full Integration
Features:
- System Monitoring (CPU/RAM/Disk/Network)
- Reflexive Optimization & Predictive Alerts
- Behavioral Learning & NLP Parsing
- Job Scheduler & Smart Suggestions
- AI / NLP Command Handling
- Plex / Media Integration
- Network Monitoring
- Mini-Games CLI Engine
- Autonomous Actions & Reporting
- Trend Analysis, Correlation, Insights
"""

import os, sys, psutil, json, time, statistics, threading, platform, socket, random, re
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque, defaultdict

try: import pyttsx3
except ImportError: pyttsx3 = None

# =========================================================
# [0] PATHS, ROOT, CONFIG
# =========================================================
ROOT = Path(__file__).parent
DATA = ROOT / "argus_data"; DATA.mkdir(exist_ok=True)
MEM  = DATA / "memory.json"
LOG  = DATA / "argus.log"
REPORT = DATA / "report.txt"
CONFIG = {
    "alert_thresholds": {"cpu":90, "ram":90, "disk":95},
    "max_log_files": 5,
    "game_highscore_file": DATA/"games.json",
    "network_scan_ports": list(range(20,1025)),
    "snapshot_interval": 5,
    "trend_window": 10
}

# =========================================================
# [1] COLORS & UTILITIES
# =========================================================
class C:
    RESET="\033[0m"; G="\033[32m"; R="\033[31m"; Y="\033[33m"
    B="\033[36m"; M="\033[35m"; W="\033[37m"
def ts(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def log(msg, level="INFO"):
    color = {"INFO":C.B,"WARN":C.Y,"ERROR":C.R,"SUCCESS":C.G}.get(level,C.M)
    line = f"{ts()} [{level}] {msg}"
    with open(LOG,"a") as f: f.write(line+"\n")
    print(f"{color}{line}{C.RESET}")

# =========================================================
# [2] MEMORY / STATE
# =========================================================
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
    "network": {"hosts":{}}
}

if MEM.exists():
    try:
        raw = json.load(open(MEM))
        for k,v in raw.items(): 
            if k == "habits": STATE[k] = defaultdict(int,v)
            elif k == "metrics": STATE[k] = deque(v, maxlen=500)
            else: STATE[k]=v
    except Exception as e:
        log(f"Memory load error {e}","WARN")

def save_state():
    to_save = {**STATE, "metrics": list(STATE["metrics"])}
    to_save["habits"] = dict(STATE["habits"])
    json.dump(to_save, open(MEM,"w"), indent=2)

# =========================================================
# [3] TTS & NOTIFICATIONS
# =========================================================
def speak(msg):
    if not pyttsx3: return
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate',180)
        engine.say(msg)
        engine.runAndWait()
    except Exception as e:
        log(f"TTS error: {e}","WARN")

def notify(msg, level="INFO"):
    log(msg, level)
    try:
        if platform.system()=="Windows":
            os.system(f'powershell -Command "Add-Type -AssemblyName PresentationFramework;[System.Windows.MessageBox]::Show(\'{msg}\')"')
        elif platform.system()=="Darwin":
            os.system(f'''osascript -e 'display notification "{msg}" with title "ARGUS"' ''')
        else:
            print("\a") # beep for Linux
    except Exception: pass
    if level in ("WARN","ERROR"): speak(msg)

# =========================================================
# [4] SYSTEM SNAPSHOT & METRICS
# =========================================================
def snapshot():
    snap = {
        "cpu": psutil.cpu_percent(0.5),
        "ram": psutil.virtual_memory().percent,
        "disk": psutil.disk_usage('/').percent,
        "time": ts()
    }
    STATE["metrics"].append(snap)
    save_state()
    return snap

def detect_anomaly():
    """Check recent metrics for anomalies using standard deviation."""
    if len(STATE["metrics"]) < CONFIG["trend_window"]: return None
    recent = list(STATE["metrics"])[-CONFIG["trend_window"]:]
    cpu_vals = [m["cpu"] for m in recent]
    ram_vals = [m["ram"] for m in recent]
    cpu_dev = statistics.stdev(cpu_vals)
    ram_dev = statistics.stdev(ram_vals)
    if cpu_dev > 25 or ram_dev > 25:
        return f"Anomaly detected: CPU dev {cpu_dev:.1f}, RAM dev {ram_dev:.1f}"
    return None

# =========================================================
# [5] RULES & REFLEX ENGINE
# =========================================================
DEFAULT_RULES = [
    {"metric":"cpu","op":">","value":90,"action":"optimize","last_trigger":0},
    {"metric":"ram","op":">","value":90,"action":"optimize","last_trigger":0},
    {"metric":"disk","op":">","value":95,"action":"alert","last_trigger":0}
]

if not STATE["rules"]: STATE["rules"]=DEFAULT_RULES; save_state()

def eval_rule(rule, metrics):
    v = metrics.get(rule["metric"],0)
    op = rule["op"]; val = rule["value"]
    return (op==">" and v>val) or (op=="<" and v<val) or (op=="=" and v==val)

def reflex_loop():
    """Continuously monitor metrics and apply rules."""
    while True:
        m = snapshot()
        for rule in STATE["rules"]:
            if eval_rule(rule, m) and (time.time()-rule.get("last_trigger",0) > 60):
                rule["last_trigger"] = time.time()
                if rule["action"] == "optimize":
                    log(f"Auto-optimization triggered by {rule['metric']}", "WARN")
                    auto_optimize()
                elif rule["action"] == "alert":
                    notify(f"ALERT: {rule['metric']} exceeded {rule['value']}%", "WARN")
        adjust_mood()
        save_state()
        time.sleep(CONFIG["snapshot_interval"])

# =========================================================
# [6] MOOD ADJUSTMENT
# =========================================================
def adjust_mood():
    if len(STATE["metrics"]) < CONFIG["trend_window"]: return
    recent = list(STATE["metrics"])[-CONFIG["trend_window"]:]
    avg_cpu = statistics.mean([x["cpu"] for x in recent])
    old_mood = STATE["mood"]
    if avg_cpu > 90: STATE["mood"] = "stressed"
    elif avg_cpu > 70: STATE["mood"] = "alert"
    elif avg_cpu < 30: STATE["mood"] = "calm"
    else: STATE["mood"] = "focused"
    if old_mood != STATE["mood"]:
        log(f"Mood changed: {old_mood} → {STATE['mood']}", "INFO")

# =========================================================
# [7] AUTONOMOUS OPTIMIZER
# =========================================================
def auto_optimize():
    """Perform safe cleanup and log rotation."""
    tmp_dir = DATA / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    for f in tmp_dir.iterdir():
        if f.is_file(): f.unlink()
    log("Temporary files cleared.", "SUCCESS")
    rotate_logs()

def rotate_logs():
    """Rotate ARGUS log files safely."""
    try:
        for i in range(CONFIG["max_log_files"]-1, 0, -1):
            f1 = DATA / f"argus.log.{i}"
            f2 = DATA / f"argus.log.{i+1}"
            if f1.exists():
                f1.rename(f2)
        log("Log rotation complete.", "SUCCESS")
    except Exception as e:
        log(f"Log rotation failed: {e}", "ERROR")

# =========================================================
# [8] START REFLEX THREAD
# =========================================================
reflex_thread = threading.Thread(target=reflex_loop, daemon=True)
reflex_thread.start()
log("Reflex Engine online.", "SUCCESS")

# =========================================================
# [9] BEHAVIORAL LEARNING & JOB SCHEDULER
# =========================================================
import re
from collections import defaultdict
from datetime import timedelta

# Initialize habits tracking
if "habits" not in STATE: STATE["habits"] = defaultdict(int)
if "jobs" not in STATE: STATE["jobs"] = []

# Pattern to parse natural language times
time_pattern = re.compile(r"(\d+)\s*(sec|secs|second|seconds|min|minute|minutes|hour|hours)")

def schedule_job(delay_sec, command):
    """Schedule a command to run after delay_sec seconds."""
    run_at = (time.time() + delay_sec)
    job = {"time": run_at, "cmd": command}
    STATE["jobs"].append(job)
    save_state()
    log(f"Scheduled job '{command}' in {delay_sec} seconds", "INFO")

def job_loop():
    """Continuously check for scheduled jobs and execute them."""
    while True:
        now = time.time()
        due_jobs = [j for j in STATE["jobs"] if j["time"] <= now]
        for job in due_jobs:
            log(f"Running scheduled job: {job['cmd']}", "SUCCESS")
            handle_command(job["cmd"], scheduled=True)
        # Keep only future jobs
        STATE["jobs"] = [j for j in STATE["jobs"] if j["time"] > now]
        save_state()
        time.sleep(2)

job_thread = threading.Thread(target=job_loop, daemon=True)
job_thread.start()

# =========================================================
# [10] NATURAL LANGUAGE PARSER
# =========================================================
def parse_natural_language(text):
    """Convert natural sentences into ARGUS commands."""
    txt = text.lower().strip()
    
    # Schedule status check
    if "status" in txt or "check" in txt:
        m = time_pattern.search(txt)
        if m:
            num, unit = int(m.group(1)), m.group(2)
            multiplier = {"sec":1,"secs":1,"second":1,"seconds":1,
                          "min":60,"minute":60,"minutes":60,
                          "hour":3600,"hours":3600}[unit]
            schedule_job(num*multiplier, "status")
            return f"Scheduled status in {num} {unit}"
        return "status"

    # Remind to say something
    if "remind" in txt and "say" in txt:
        phrase = re.findall(r"say\s+(.+)", txt)
        msg = phrase[0] if phrase else "hello"
        schedule_job(60, f"say:{msg}")
        return f"Will say '{msg}' in 1 minute"

    # Direct say command
    if txt.startswith("say"):
        return f"say:{txt.split('say',1)[1].strip()}"
    
    return None

# =========================================================
# [11] EVENT CORRELATION
# =========================================================
if "events" not in STATE: STATE["events"] = []

def log_event(event_name):
    """Attach system context to user-triggered events."""
    snap = snapshot()
    entry = {
        "event": event_name,
        "time": ts(),
        "cpu": snap["cpu"],
        "ram": snap["ram"],
        "disk": snap["disk"]
    }
    STATE["events"].append(entry)
    # Keep last 200 events
    STATE["events"] = STATE["events"][-200:]
    save_state()
    log(f"Event logged: {event_name} @ CPU {snap['cpu']}% RAM {snap['ram']}%", "INFO")

def analyze_event_correlations():
    """Return events that occurred during high system load."""
    anomalies = []
    for e in STATE["events"][-50:]:
        if e["cpu"] > 80 or e["ram"] > 85:
            anomalies.append(e)
    return anomalies

# =========================================================
# [12] COMMAND HANDLER
# =========================================================
def handle_command(cmd, scheduled=False):
    """Handle ARGUS commands and update habits."""
    if not scheduled:
        STATE["habits"][cmd] += 1
        save_state()

    if cmd.startswith("say:"):
        msg = cmd.split("say:",1)[1]
        speak(msg)
        log(f"Said: {msg}", "INFO")
        return

    if cmd == "status":
        s = snapshot()
        print(f"CPU {s['cpu']}% RAM {s['ram']}% DISK {s['disk']}% Mood {STATE['mood']}")
        return

    if cmd == "jobs":
        for j in STATE["jobs"]:
            t = time.ctime(j["time"])
            print(f"{t} → {j['cmd']}")
        return

    if cmd == "habits":
        top = sorted(STATE["habits"].items(), key=lambda x: -x[1])[:5]
        for k, v in top:
            print(f"{k}: {v}")
        return

    print("Unknown command.")

# =========================================================
# [13] MAIN INTERACTIVE LOOP
# =========================================================
log("ARGUS Behavioral Layer ready.", "SUCCESS")
while True:
    try:
        raw = input(f"{C.G}>> {C.X}").strip()
        if not raw: continue
        if raw in ("exit","quit"): break

        parsed = parse_natural_language(raw)
        if parsed:
            handle_command(parsed)
        else:
            handle_command(raw)
    except KeyboardInterrupt:
        print()
    except Exception as e:
        log(f"Loop error {e}", "ERROR")

# =========================================================
# [14] PATTERN RECOGNITION & INSIGHT ENGINE
# =========================================================
from collections import deque

# Initialize metrics buffer
if "metrics" not in STATE: STATE["metrics"] = deque(maxlen=500)
if "insights" not in STATE: STATE["insights"] = []

def sample_metrics():
    """Take a snapshot of system metrics and store them."""
    snap = snapshot()
    STATE["metrics"].append(snap)
    save_state()
    return snap

def compute_trends():
    """Compute trends and averages for CPU, RAM, and Disk usage."""
    if len(STATE["metrics"]) < 10: return None
    cpu = [m["cpu"] for m in STATE["metrics"]]
    ram = [m["ram"] for m in STATE["metrics"]]
    disk = [m["disk"] for m in STATE["metrics"]]

    def avg(vals): return round(statistics.mean(vals), 1)
    def slope(vals): return (vals[-1] - vals[0]) / max(len(vals)-1, 1)

    trends = {
        "cpu_avg": avg(cpu), "cpu_slope": slope(cpu),
        "ram_avg": avg(ram), "ram_slope": slope(ram),
        "disk_avg": avg(disk), "disk_slope": slope(disk)
    }
    return trends

def generate_insight():
    """Generate a system insight and store in STATE['insights']."""
    trends = compute_trends()
    if not trends: return None

    msg = []
    mood = "calm"

    if trends["cpu_slope"] > 1 or trends["ram_slope"] > 1:
        mood = "alert"
    if trends["cpu_avg"] > 80 or trends["ram_avg"] > 85:
        mood = "stressed"

    msg.append(f"CPU avg {trends['cpu_avg']}%, trend {'rising' if trends['cpu_slope']>0 else 'stable'}")
    msg.append(f"RAM avg {trends['ram_avg']}%, trend {'rising' if trends['ram_slope']>0 else 'stable'}")
    msg.append(f"Disk avg {trends['disk_avg']}%, trend {'rising' if trends['disk_slope']>0 else 'stable'}")

    insight = f"{ts()} — System is {mood}. " + "; ".join(msg)
    STATE["insights"].append(insight)
    STATE["insights"] = STATE["insights"][-200:]  # keep last 200 insights
    STATE["mood"] = mood
    save_state()
    log("New insight generated", "SUCCESS")
    return insight

def insight_loop():
    """Continuous loop to sample metrics and generate insights."""
    while True:
        sample_metrics()
        if len(STATE["metrics"]) >= 10:
            generate_insight()
        time.sleep(60)

threading.Thread(target=insight_loop, daemon=True).start()

# =========================================================
# [15] AUTONOMOUS DAILY REPORT
# =========================================================
REPORT_FILE = DATA / "daily_report.txt"

def ascii_graph(values, width=30):
    """Generate simple ASCII sparkline graph."""
    if not values: return ""
    mx, mn = max(values), min(values)
    scale = max(mx - mn, 1e-9)
    step = max(1, int(len(values)/width))
    bars = []
    for i in range(0, len(values), step):
        v = values[i]
        lvl = int(((v - mn)/scale) * 8)
        bars.append(" ▁▂▃▄▅▆▇█"[lvl])
    return "".join(bars)

def generate_daily_report():
    """Generate and save a daily report with trends and anomalies."""
    trends = compute_trends()
    if not trends: return "Not enough data."

    cpu = [m["cpu"] for m in STATE["metrics"]]
    ram = [m["ram"] for m in STATE["metrics"]]
    disk = [m["disk"] for m in STATE["metrics"]]
    anomalies = analyze_event_correlations()

    lines = [
        f"ARGUS Autonomous Daily Report — {ts()}",
        f"Mood: {STATE['mood']}",
        "",
        f"CPU avg {trends['cpu_avg']}% slope {trends['cpu_slope']}",
        ascii_graph(cpu),
        f"RAM avg {trends['ram_avg']}% slope {trends['ram_slope']}",
        ascii_graph(ram),
        f"Disk avg {trends['disk_avg']}% slope {trends['disk_slope']}",
        ascii_graph(disk),
        ""
    ]

    if anomalies:
        lines.append("⚠️  High-load correlated events:")
        for a in anomalies[-5:]:
            lines.append(f" - {a['time']} {a['event']} CPU:{a['cpu']} RAM:{a['ram']}")
    else:
        lines.append("No significant anomalies detected.")

    REPORT_FILE.write_text("\n".join(lines))
    STATE["last_report"] = time.time()
    save_state()
    log("Daily report generated", "SUCCESS")
    return "\n".join(lines)

def daily_report_loop():
    """Automatically generate daily reports every 24 hours."""
    while True:
        now = time.time()
        if now - STATE.get("last_report", 0) > 86400:  # 24h
            generate_daily_report()
        time.sleep(600)

threading.Thread(target=daily_report_loop, daemon=True).start()

# =========================================================
# [16] COMMANDS FOR INSIGHTS & REPORTS
# =========================================================
def cmd_status():
    s = sample_metrics()
    print(f"CPU {s['cpu']}% RAM {s['ram']}% DISK {s['disk']}% Mood {STATE['mood']}")

def cmd_insight():
    print(generate_daily_report())

def cmd_events():
    for e in STATE["events"][-10:]:
        print(f"{e['time']} — {e['event']} CPU {e['cpu']} RAM {e['ram']}")

def handle(cmd):
    log_event(cmd)
    if cmd == "status": cmd_status()
    elif cmd in ("insight", "report"): print(generate_daily_report())
    elif cmd == "events": cmd_events()
    else: print("Unknown command.")

# =========================================================
# [17] MAIN LOOP
# =========================================================
log("ARGUS Reporting & Correlation Layer active.", "SUCCESS")
while True:
    try:
        raw = input(f"{C.G}>> {C.X}").strip()
        if raw in ("exit","quit"): break
        if raw: handle(raw)
    except KeyboardInterrupt:
        print()
    except Exception as e:
        log(f"Loop error {e}", "ERROR")

# =========================================================
# [18] GAMES MODULE
# =========================================================
import random

GAMES = ["guess_number", "tic_tac_toe", "rock_paper_scissors"]

def game_guess_number():
    """Classic number guessing game."""
    n = random.randint(1, 100)
    attempts = 0
    print("Guess a number between 1-100")
    while True:
        try:
            guess = int(input("Your guess> "))
            attempts += 1
            if guess == n:
                print(f"Correct! You guessed in {attempts} attempts.")
                break
            elif guess < n:
                print("Too low")
            else:
                print("Too high")
        except ValueError:
            print("Enter a valid number.")

def game_rock_paper_scissors():
    choices = ["rock", "paper", "scissors"]
    while True:
        player = input("Rock/Paper/Scissors> ").lower()
        if player not in choices:
            print("Invalid choice.")
            continue
        computer = random.choice(choices)
        print(f"Computer chose {computer}")
        if player == computer:
            print("Tie!")
        elif (player == "rock" and computer=="scissors") or \
             (player=="paper" and computer=="rock") or \
             (player=="scissors" and computer=="paper"):
            print("You win!")
        else:
            print("You lose!")
        again = input("Play again? (y/n)> ").lower()
        if again != "y": break

def start_game(name):
    if name == "guess_number": game_guess_number()
    elif name == "rock_paper_scissors": game_rock_paper_scissors()
    else: print("Game not implemented.")

# =========================================================
# [19] AI MODULES
# =========================================================
try:
    import openai
except ImportError:
    openai = None
    log("OpenAI library not found. AI features disabled.", "WARN")

def ai_chat(prompt):
    if not openai:
        print("AI module not available.")
        return
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=150
        )
        print(response.choices[0].text.strip())
    except Exception as e:
        print(f"AI error: {e}")

# =========================================================
# [20] NETWORKING UTILITIES
# =========================================================
import socket
import urllib.request

def net_ping(host="8.8.8.8", port=53, timeout=2):
    """Ping a host to check connectivity"""
    try:
        s = socket.create_connection((host, port), timeout=timeout)
        s.close()
        print(f"Host {host} reachable.")
    except Exception:
        print(f"Host {host} not reachable.")

def net_speed_test(url="http://google.com"):
    try:
        start = time.time()
        urllib.request.urlopen(url, timeout=5)
        end = time.time()
        print(f"Ping time to {url}: {(end-start)*1000:.2f}ms")
    except Exception as e:
        print(f"Network test failed: {e}")

# =========================================================
# [21] AUTOMATION & SMART TOOLS
# =========================================================
def auto_clean_temp():
    """Autonomously clear temp folder to free resources"""
    tmp = DATA / "tmp"
    tmp.mkdir(exist_ok=True)
    for f in tmp.iterdir():
        try: f.unlink()
        except: pass
    log("Temp folder cleared.", "SUCCESS")

def auto_optimize_system():
    """Combine temp cleaning + memory optimization"""
    auto_clean_temp()
    log("System optimized.", "SUCCESS")

# =========================================================
# [22] GAMES/AI/NETWORK COMMAND HANDLER
# =========================================================
def cmd_games():
    print("Available games:", ", ".join(GAMES))
    choice = input("Choose game> ").strip()
    start_game(choice)

def cmd_ai():
    prompt = input("Enter AI prompt> ")
    ai_chat(prompt)

def cmd_network():
    print("Networking tools: ping, speed")
    choice = input("Choose tool> ").strip()
    if choice=="ping": net_ping()
    elif choice=="speed": net_speed_test()
    else: print("Unknown tool")

# =========================================================
# INTEGRATE INTO MAIN LOOP
# =========================================================
def handle_full_command(cmd):
    log_event(cmd)
    cmd = cmd.lower()
    if cmd in ("status",): cmd_status()
    elif cmd in ("insight","report"): print(generate_daily_report())
    elif cmd in ("events",): cmd_events()
    elif cmd=="games": cmd_games()
    elif cmd=="ai": cmd_ai()
    elif cmd=="network": cmd_network()
    elif cmd=="optimize": auto_optimize_system()
    else: print("Unknown command.")

log("ARGUS Full Utility Layer active.", "SUCCESS")
while True:
    try:
        raw = input(f"{C.G}>> {C.X}").strip()
        if raw in ("exit","quit"): break
        if raw: handle_full_command(raw)
    except KeyboardInterrupt: print()
    except Exception as e:
        log(f"Main loop error {e}", "ERROR")

# =========================================================
# [23] ADVANCED AI AGENTS
# =========================================================
class AIAgent:
    def __init__(self, name):
        self.name = name
        self.knowledge = []
    
    def learn(self, data):
        self.knowledge.append(data)
        log(f"Agent {self.name} learned: {data}", "INFO")
    
    def respond(self, prompt):
        response = f"{self.name} says: I have learned {len(self.knowledge)} things. About '{prompt}', I suggest thinking differently."
        log(f"Agent response: {response}", "INFO")
        return response

AGENTS = {"athena": AIAgent("Athena"), "zephyr": AIAgent("Zephyr")}

def cmd_ai_agent():
    print("Available agents:", ", ".join(AGENTS.keys()))
    agent_name = input("Choose agent> ").strip().lower()
    if agent_name not in AGENTS:
        print("Unknown agent.")
        return
    prompt = input("Prompt> ")
    print(AGENTS[agent_name].respond(prompt))
    AGENTS[agent_name].learn(prompt)

# =========================================================
# [24] ADVANCED GAMES SUITE
# =========================================================
def game_tic_tac_toe():
    board = [" "]*9
    def print_board():
        print(f"{board[0]}|{board[1]}|{board[2]}\n-+-+-\n{board[3]}|{board[4]}|{board[5]}\n-+-+-\n{board[6]}|{board[7]}|{board[8]}")
    def check_win(p):
        wins = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
        return any(all(board[i]==p for i in w) for w in wins)
    player = "X"; computer = "O"
    for turn in range(9):
        print_board()
        if turn%2==0:
            move=int(input("Your move (0-8)> "))
            if board[move]!=" ": continue
            board[move]=player
            if check_win(player): print_board(); print("You win!"); return
        else:
            move=random.choice([i for i,x in enumerate(board) if x==" "])
            board[move]=computer
            if check_win(computer): print_board(); print("Computer wins!"); return
    print_board(); print("Tie!")

GAMES.extend(["tic_tac_toe"])

# =========================================================
# [25] NETWORK SCANNING
# =========================================================
import subprocess

def scan_local_network():
    """Scan local network for active IPs"""
    print("Scanning local network (192.168.1.0/24)...")
    active_hosts = []
    for i in range(1,255):
        ip = f"192.168.1.{i}"
        try:
            output = subprocess.check_output(["ping", "-c", "1", "-W", "1", ip], stderr=subprocess.DEVNULL)
            active_hosts.append(ip)
        except Exception:
            continue
    print("Active hosts:", active_hosts)

def cmd_network_advanced():
    print("Network commands: scan, ping, speed")
    choice = input("Tool> ").strip().lower()
    if choice=="scan": scan_local_network()
    elif choice=="ping": net_ping()
    elif choice=="speed": net_speed_test()
    else: print("Unknown network tool")

# =========================================================
# [26] PREDICTIVE ANALYTICS
# =========================================================
def predictive_alerts():
    for metric, history in RESOURCE_HISTORY.items():
        if len(history) >= TREND_WINDOW:
            recent = list(history)[-TREND_WINDOW:]
            slope = compute_trend(recent)
            avg = statistics.mean(recent)
            if slope > 0.5 and avg > 70:
                log(f"Predictive Alert: {metric.upper()} trending upward (avg {avg:.1f}%)", "WARN")

threading.Thread(target=lambda: [predictive_alerts(); time.sleep(10)], daemon=True).start()

# =========================================================
# [27] AUTOMATED TASK SCRIPTS
# =========================================================
def auto_backup():
    """Dummy backup script"""
    log("Auto-backup started.", "INFO")
    time.sleep(2)
    log("Auto-backup completed.", "SUCCESS")

def auto_cleanup_logs():
    """Rotate ARGUS logs"""
    log("Auto-cleaning logs...", "INFO")
    try:
        for i in range(CONFIG.get("max_log_files",5)-1,0,-1):
            f1 = DATA / f"argus.log.{i}"
            f2 = DATA / f"argus.log.{i+1}"
            if f1.exists(): f1.rename(f2)
        log("Logs rotated.", "SUCCESS")
    except Exception as e:
        log(f"Log rotation failed: {e}", "ERROR")

# =========================================================
# [28] FULL COMMAND INTEGRATION
# =========================================================
def handle_supreme_command(cmd):
    log_event(cmd)
    cmd = cmd.lower()
    if cmd in ("status",): cmd_status()
    elif cmd in ("insight","report"): print(generate_daily_report())
    elif cmd in ("events",): cmd_events()
    elif cmd=="games": cmd_games()
    elif cmd=="tic_tac_toe": game_tic_tac_toe()
    elif cmd=="ai": cmd_ai()
    elif cmd=="ai_agent": cmd_ai_agent()
    elif cmd=="network": cmd_network()
    elif cmd=="network_advanced": cmd_network_advanced()
    elif cmd=="optimize": auto_optimize_system()
    elif cmd=="backup": auto_backup()
    elif cmd=="cleanup_logs": auto_cleanup_logs()
    else: print("Unknown command.")

# =========================================================
# [29] SUPREME MAIN LOOP
# =========================================================
log("ARGUS Supreme-Core Section 6 active: AI + Games + Networking + Automation", "SUCCESS")
while True:
    try:
        raw = input(f"{C.G}>> {C.X}").strip()
        if raw in ("exit","quit"): break
        if raw: handle_supreme_command(raw)
    except KeyboardInterrupt: print()
    except Exception as e:
        log(f"Supreme loop error {e}", "ERROR")

# =======================================================
# Section 7 — Advanced Games, Safe Network Lab, Predictive Intelligence
# =======================================================
import math
import copy
import random

# Helper to register commands whether using decorator or COMMANDS dict
def register_command(name, fn, help_text=""):
    try:
        # prefer decorator style if available
        if 'argus_command' in globals() and callable(argus_command):
            # create a wrapper to keep signature consistent
            @argus_command(name, help_text=help_text)
            def _wrapper(args):
                return fn(args)
        else:
            # fall back to COMMANDS dict
            COMMANDS.setdefault(name, (lambda args, f=fn: f(args)))
            COMMAND_HELP.setdefault(name, help_text)
    except Exception:
        # ensure fallback
        try:
            COMMANDS.setdefault(name, (lambda args, f=fn: f(args)))
            COMMAND_HELP.setdefault(name, help_text)
        except Exception:
            pass

# -------------------------
# 7.1 Advanced Chess (Minimax + Alpha-Beta, Legal move generator)
# -------------------------
class ChessGame:
    def __init__(self):
        self.reset()

    def reset(self):
        # board: 8x8 with uppercase white, lowercase black, '.' empty
        self.board = [
            list("rnbqkbnr"),
            list("pppppppp"),
            list("........"),
            list("........"),
            list("........"),
            list("........"),
            list("PPPPPPPP"),
            list("RNBQKBNR"),
        ]
        self.turn = "white"  # white starts
        self.history = []

    def clone(self):
        g = ChessGame()
        g.board = [row[:] for row in self.board]
        g.turn = self.turn
        g.history = self.history[:]
        return g

    def is_in_bounds(self, r, c):
        return 0 <= r < 8 and 0 <= c < 8

    def enemy(self, a, b):
        if a == '.' or b == '.': return False
        return a.isupper() != b.isupper()

    def piece_color(self, p):
        if p == '.': return None
        return 'white' if p.isupper() else 'black'

    def generate_moves_for(self, r, c):
        piece = self.board[r][c]
        if piece == '.': return []
        moves = []
        lower = piece.lower()
        sign = 1 if piece.islower() else -1  # black moves +1 in rows, white -1
        # pawn
        if lower == 'p':
            step = 1 if piece.islower() else -1
            # forward
            nr = r + step
            if self.is_in_bounds(nr, c) and self.board[nr][c] == '.':
                moves.append((r, c, nr, c))
                # double
                start_row = 1 if piece.islower() else 6
                if r == start_row:
                    nr2 = r + 2*step
                    if self.is_in_bounds(nr2, c) and self.board[nr2][c] == '.':
                        moves.append((r, c, nr2, c))
            # captures
            for dc in (-1, 1):
                nc = c + dc
                nr = r + step
                if self.is_in_bounds(nr, nc) and self.enemy(piece, self.board[nr][nc]):
                    moves.append((r, c, nr, nc))
            return moves
        # knight
        if lower == 'n':
            deltas = [(2,1),(1,2),(-1,2),(-2,1),(-2,-1),(-1,-2),(1,-2),(2,-1)]
            for dr,dc in deltas:
                nr, nc = r+dr, c+dc
                if not self.is_in_bounds(nr,nc): continue
                if self.board[nr][nc] == '.' or self.enemy(piece, self.board[nr][nc]):
                    moves.append((r,c,nr,nc))
            return moves
        # king
        if lower == 'k':
            for dr in (-1,0,1):
                for dc in (-1,0,1):
                    if dr==0 and dc==0: continue
                    nr, nc = r+dr, c+dc
                    if not self.is_in_bounds(nr,nc): continue
                    if self.board[nr][nc] == '.' or self.enemy(piece, self.board[nr][nc]):
                        moves.append((r,c,nr,nc))
            return moves
        # sliding pieces: rook, bishop, queen
        directions = []
        if lower == 'r':
            directions = [(1,0),(-1,0),(0,1),(0,-1)]
        elif lower == 'b':
            directions = [(1,1),(1,-1),(-1,1),(-1,-1)]
        elif lower == 'q':
            directions = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
        for dr, dc in directions:
            nr, nc = r+dr, c+dc
            while self.is_in_bounds(nr, nc):
                if self.board[nr][nc] == '.':
                    moves.append((r,c,nr,nc))
                elif self.enemy(piece, self.board[nr][nc]):
                    moves.append((r,c,nr,nc)); break
                else:
                    break
                nr += dr; nc += dc
        return moves

    def all_moves(self, for_color=None):
        moves = []
        for r in range(8):
            for c in range(8):
                p = self.board[r][c]
                if p == '.': continue
                if for_color and self.piece_color(p) != for_color: continue
                moves.extend(self.generate_moves_for(r,c))
        return moves

    def apply_move(self, mv):
        sr, sc, dr, dc = mv
        piece = self.board[sr][sc]
        captured = self.board[dr][dc]
        self.board[dr][dc] = piece
        self.board[sr][sc] = '.'
        self.history.append((mv, captured))
        self.turn = 'black' if self.turn == 'white' else 'white'

    def undo_last(self):
        if not self.history: return
        (sr, sc, dr, dc), captured = self.history.pop()
        piece = self.board[dr][dc]
        self.board[sr][sc] = piece
        self.board[dr][dc] = captured
        self.turn = 'black' if self.turn == 'white' else 'white'

    def evaluate(self):
        # rough evaluation: piece values + center control
        vals = {'p':1,'n':3,'b':3,'r':5,'q':9,'k':800}
        score = 0
        for r in range(8):
            for c in range(8):
                p = self.board[r][c]
                if p == '.': continue
                v = vals.get(p.lower(), 0)
                bonus = 0.1*v if 2 <= r <= 5 and 2 <= c <= 5 else 0
                if p.isupper():
                    score += v + bonus
                else:
                    score -= v + bonus
        return score

    def is_game_over(self):
        # simplified: no check detection; just no moves
        return len(self.all_moves(self.turn)) == 0

# Minimax with alpha-beta and depth limit
def chess_minimax(game, depth, alpha, beta, maximizing):
    if depth == 0 or game.is_game_over():
        return game.evaluate(), None
    best_move = None
    color = 'white' if maximizing else 'black'
    moves = game.all_moves(for_color=color)
    if not moves:
        return game.evaluate(), None
    if maximizing:
        max_eval = -1e9
        for mv in moves:
            game.apply_move(mv)
            val, _ = chess_minimax(game, depth-1, alpha, beta, False)
            game.undo_last()
            if val > max_eval:
                max_eval = val; best_move = mv
            alpha = max(alpha, val)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = 1e9
        for mv in moves:
            game.apply_move(mv)
            val, _ = chess_minimax(game, depth-1, alpha, beta, True)
            game.undo_last()
            if val < min_eval:
                min_eval = val; best_move = mv
            beta = min(beta, val)
            if beta <= alpha:
                break
        return min_eval, best_move

# CLI wrapper & command
_chess_table = {"active": None}
def chess_start(args):
    _chess_table["active"] = ChessGame()
    return "Chess game started. Use 'chess_move e2e4' to move, 'chess_show' to draw board, 'chess_ai <depth>' to let AI move."

def chess_show(args):
    g = _chess_table.get("active")
    if not g: return "No active chess game."
    letters = "  a b c d e f g h"
    out = [letters]
    for i, row in enumerate(g.board):
        rnum = 8 - i
        line = f"{rnum} " + " ".join(p if p != '.' else '.' for p in row)
        out.append(line)
    out.append(letters)
    return "\n".join(out)

def parse_sq(sq):
    cols = "abcdefgh"
    return 8 - int(sq[1]), cols.index(sq[0])

def chess_move(args):
    if not args: return "Usage: chess_move <e2e4>"
    g = _chess_table.get("active")
    if not g: return "No active chess game."
    mv = args[0].strip()
    if len(mv) != 4: return "Move format e2e4"
    try:
        sr, sc = parse_sq(mv[:2]); dr, dc = parse_sq(mv[2:])
    except Exception:
        return "Invalid squares."
    legal = g.generate_moves_for(sr, sc)
    if (sr, sc, dr, dc) not in legal:
        return "Illegal or invalid move."
    g.apply_move((sr, sc, dr, dc))
    return "Move applied."

def chess_ai(args):
    depth = int(args[0]) if args else 2
    g = _chess_table.get("active")
    if not g: return "No active chess game."
    maximizing = True if g.turn == 'white' else False
    val, mv = chess_minimax(g, depth, -1e9, 1e9, maximizing)
    if mv:
        g.apply_move(mv)
        return f"AI moved {mv}"
    return "AI found no move."

# register chess commands
register_command("chess_start", chess_start, "Start a chess game")
register_command("chess_show", chess_show, "Show chess board")
register_command("chess_move", chess_move, "Make a move: chess_move e2e4")
register_command("chess_ai", chess_ai, "Let AI play a move: chess_ai [depth]")

# -------------------------
# 7.2 Small Text RPG (local, safe)
# -------------------------
class TinyRPG:
    def __init__(self):
        self.player = {"hp": 30, "atk": 4, "xp":0, "lvl":1}
        self.enemy = None

    def spawn_enemy(self):
        lvl = max(1, self.player["lvl"] + random.choice([-1,0,1]))
        hp = 8 + lvl * 4
        atk = 1 + lvl
        self.enemy = {"hp": hp, "atk": atk, "lvl": lvl}

    def attack(self):
        if not self.enemy: return "No enemy. Use 'rpg_spawn' to provoke one."
        dmg = random.randint(1, self.player["atk"])
        self.enemy["hp"] -= dmg
        reply = f"You hit for {dmg}. Enemy HP {self.enemy['hp']}."
        if self.enemy["hp"] <= 0:
            self.player["xp"] += 5 * self.enemy["lvl"]
            self.enemy = None
            reply += " Enemy defeated! You gained XP."
            if self.player["xp"] >= 10 * self.player["lvl"]:
                self.player["lvl"] += 1
                self.player["hp"] += 5
                reply += f" Level up! Now lvl {self.player['lvl']}."
        else:
            edmg = random.randint(1, self.enemy["atk"])
            self.player["hp"] -= edmg
            reply += f" Enemy strikes back for {edmg}. Your HP {self.player['hp']}."
            if self.player["hp"] <= 0:
                reply += " You were defeated. Respawning..."
                self.player = {"hp": 30, "atk":4, "xp":0, "lvl":1}
        return reply

_rpg = TinyRPG()
def rpg_spawn(args):
    _rpg.spawn_enemy()
    return f"Enemy spawned: lvl {_rpg.enemy['lvl']} HP {_rpg.enemy['hp']} ATK {_rpg.enemy['atk']}"

def rpg_attack(args):
    return _rpg.attack()

def rpg_status(args):
    return f"Player: HP {_rpg.player['hp']} ATK {_rpg.player['atk']} LVL {_rpg.player['lvl']} XP {_rpg.player['xp']}"

register_command("rpg_spawn", rpg_spawn, "Spawn an enemy")
register_command("rpg_attack", rpg_attack, "Attack the enemy")
register_command("rpg_status", rpg_status, "Show player status")

# -------------------------
# 7.3 Safe Network Simulation Lab (NO real scans)
# -------------------------
class NetworkLab:
    """
    Local simulation environment for teaching/network exercises.
    This is a MODEL of a network — not a real scanner or attacker.
    Use it to test ideas, practice parsing, or simulate mapping.
    """
    def __init__(self):
        # simulated nodes: ip -> services
        self.nodes = {
            "192.168.1.10": {"host":"printer.local", "services":["ipp","http"]},
            "192.168.1.20": {"host":"nas.local", "services":["smb","ssh"]},
            "192.168.1.30": {"host":"camera.local", "services":["rtsp"]},
        }
    def list_nodes(self):
        return self.nodes.copy()
    def service_summary(self):
        summary = {}
        for ip,info in self.nodes.items():
            for s in info["services"]:
                summary.setdefault(s, []).append(ip)
        return summary
    def simulate_exploit(self, ip, service):
        # purely deterministic simulation with random chance influenced by service type
        if ip not in self.nodes: return {"ok":False, "reason":"unknown host"}
        if service not in self.nodes[ip]["services"]: return {"ok":False, "reason":"service not present"}
        base = {"ssh":0.05,"http":0.15,"smb":0.1,"rtsp":0.2,"ipp":0.02}
        chance = base.get(service, 0.05)
        # seed with ip so it's reproducible
        rnd = (sum(map(int, ip.split("."))) % 100) / 100.0
        success = rnd < chance
        return {"ok": success, "chance": chance, "seed": rnd}

_netlab = NetworkLab()
def netlab_nodes(args):
    nodes = _netlab.list_nodes()
    lines = [f"{ip} -> {info['host']} services={info['services']}" for ip,info in nodes.items()]
    return "\n".join(lines)

def netlab_services(args):
    s = _netlab.service_summary()
    return "\n".join(f"{svc}: {ips}" for svc, ips in s.items())

def netlab_simulate(args):
    if not args or len(args) < 2:
        return "Usage: netlab_simulate <ip> <service>"
    ip, svc = args[0], args[1]
    res = _netlab.simulate_exploit(ip, svc)
    return f"Simulated exploit -> {res}"

register_command("netlab_nodes", netlab_nodes, "Show simulated network nodes")
register_command("netlab_services", netlab_services, "Show simulated services")
register_command("netlab_simulate", netlab_simulate, "Simulate exploit in lab (safe)")

# -------------------------
# 7.4 Predictive Intelligence Extensions
# -------------------------
def moving_average(series, window=5):
    if not series: return 0.0
    window = min(window, len(series))
    return sum(series[-window:]) / window

def monte_carlo_forecast(metric_series, steps=10, trials=100):
    """
    Simple Monte Carlo forecast using random walk based on recent deltas.
    Returns mean projected value after 'steps'.
    """
    if len(metric_series) < 2:
        return None
    deltas = [metric_series[i+1]-metric_series[i] for i in range(len(metric_series)-1)]
    if not deltas:
        return None
    mu = sum(deltas) / len(deltas)
    sigma = (sum((d-mu)**2 for d in deltas) / max(1,len(deltas)-1))**0.5
    results = []
    last = metric_series[-1]
    for _ in range(trials):
        value = last
        for _ in range(steps):
            step = random.gauss(mu, sigma)
            value += step
            if value < 0: value = 0
            if value > 100: value = 100
        results.append(value)
    mean_proj = sum(results)/len(results)
    return {"mean": mean_proj, "trials": trials, "steps": steps, "samples": results[:5]}

def cmd_forecast_mc(args):
    metric = (args[0] if args else "cpu").lower()
    buffer = []
    # try multiple possible histories in global memory/state
    for candidate in ("RESOURCE_HISTORY", "memory", "state", "mem"):
        if candidate in globals():
            obj = globals()[candidate]
            try:
                if candidate == "RESOURCE_HISTORY":
                    buffer = list(RESOURCE_HISTORY.get(metric, []))
                else:
                    # try to extract time series stored as dicts
                    seq = []
                    if isinstance(obj, dict):
                        # look for system metrics list inside
                        for k in ["system_metrics", "metrics", "system_metrics_buffer"]:
                            if k in obj:
                                seq = obj[k]
                                break
                    if seq and isinstance(seq, list) and seq and isinstance(seq[0], dict):
                        buffer = [float(x.get(metric, 0.0)) for x in seq if metric in x]
                if buffer:
                    break
            except Exception:
                continue
    if not buffer:
        return "No historical series available for forecasting."
    res = monte_carlo_forecast(buffer, steps=int(args[1]) if len(args)>1 else 10, trials=int(args[2]) if len(args)>2 else 200)
    if not res:
        return "Forecast failed (not enough data)."
    return f"MC forecast mean {res['mean']:.2f}% after {res['steps']} steps (sample results: {res['samples']})"

register_command("forecast_mc", cmd_forecast_mc, "Monte Carlo forecast: forecast_mc [metric] [steps] [trials]")

# -------------------------
# 7.5 Integration: games & tools list
# -------------------------
def cmd_games(args):
    games = ["tic_tac_toe", "chess_start", "rpg_spawn"]
    return "Available games: " + ", ".join(games)

register_command("games", cmd_games, "List available games and game commands")

# -------------------------
# 7.6 Quick bindings for older ARGUS variants
# -------------------------
# Some older snippets used execute()/handle(). Add safe wrappers that call register_command
def _safe_call(name, args):
    if name in COMMANDS:
        try:
            return COMMANDS[name](args)
        except Exception as e:
            return f"Error running {name}: {e}"
    return f"Command {name} not found."

# -------------------------
# 7.7 Section 7 Ready
# -------------------------
try:
    log("Section 7 loaded: Advanced Games, Safe Network Lab & Predictive Intelligence", "success")
except Exception:
    pass

# =======================================================
# Section 8 — ARGUS UI/CLI Polishing, Plugins & Developer API
# =======================================================
import json
import readline
import inspect
import functools
import atexit
import os

# -------------------------
# 8.1 Persistent Command History & Autocomplete
# -------------------------
HISTORY_FILE = os.path.expanduser("~/.argus_history")
try:
    readline.read_history_file(HISTORY_FILE)
except FileNotFoundError:
    pass

atexit.register(lambda: readline.write_history_file(HISTORY_FILE))

def complete(text, state):
    options = [cmd for cmd in COMMANDS if cmd.startswith(text)]
    if state < len(options):
        return options[state]
    return None

readline.set_completer(complete)
readline.parse_and_bind("tab: complete")

# -------------------------
# 8.2 Colored CLI output
# -------------------------
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    RESET = '\033[0m'

def cprint(msg, color='RESET'):
    col = getattr(Colors, color.upper(), Colors.RESET)
    print(f"{col}{msg}{Colors.RESET}")

# -------------------------
# 8.3 Plugin System (load/unload scripts dynamically)
# -------------------------
PLUGINS = {}
PLUGIN_DIR = os.path.expanduser("~/.argus_plugins")
os.makedirs(PLUGIN_DIR, exist_ok=True)

def load_plugin(filename):
    """Load a Python plugin into ARGUS runtime."""
    path = os.path.join(PLUGIN_DIR, filename)
    if not os.path.isfile(path):
        return f"Plugin {filename} not found."
    try:
        code = open(path, "r").read()
        exec(code, globals())
        PLUGINS[filename] = code
        return f"Plugin {filename} loaded successfully."
    except Exception as e:
        return f"Error loading plugin: {e}"

def unload_plugin(filename):
    if filename in PLUGINS:
        del PLUGINS[filename]
        return f"Plugin {filename} unloaded."
    return f"Plugin {filename} not loaded."

def list_plugins(args=None):
    return "Loaded plugins: " + ", ".join(PLUGINS.keys()) if PLUGINS else "No plugins loaded."

register_command("plugin_load", lambda args: load_plugin(args[0]) if args else "Usage: plugin_load <file.py>", "Load a plugin")
register_command("plugin_unload", lambda args: unload_plugin(args[0]) if args else "Usage: plugin_unload <file.py>", "Unload a plugin")
register_command("plugin_list", list_plugins, "List loaded plugins")

# -------------------------
# 8.4 Developer API helpers
# -------------------------
def argus_export(name=None):
    """Decorator to export a function as ARGUS command"""
    def decorator(fn):
        cmd_name = name or fn.__name__
        register_command(cmd_name, fn, help_text=fn.__doc__ or "")
        return fn
    return decorator

# Example dev API usage
@argus_export("hello")
def hello_cmd(args):
    """Say hello to ARGUS"""
    return f"Hello ARGUS! Args: {args}"

# -------------------------
# 8.5 Extended Game AI hooks
# -------------------------
# Example: allow chess AI depth and rpg difficulty settings
ARGUS_GAME_SETTINGS = {
    "chess_depth": 3,
    "rpg_enemy_lvl_offset": 1,
}

@argus_export("game_config")
def game_config(args):
    """
    View or set game configuration.
    Usage:
        game_config                # show current settings
        game_config chess_depth 4  # set chess AI depth to 4
    """
    if not args:
        return f"Current settings: {ARGUS_GAME_SETTINGS}"
    if len(args) < 2:
        return "Usage: game_config <key> <value>"
    key, val = args[0], args[1]
    if key not in ARGUS_GAME_SETTINGS:
        return f"Unknown setting {key}"
    try:
        ARGUS_GAME_SETTINGS[key] = int(val)
        return f"{key} set to {val}"
    except Exception as e:
        return f"Failed to set {key}: {e}"

# -------------------------
# 8.6 Enhanced CLI loop
# -------------------------
def argus_cli_loop(prompt="ARGUS> "):
    cprint("Welcome to ARGUS CLI. Type 'help' for commands.", "CYAN")
    while True:
        try:
            line = input(prompt)
            if not line.strip(): continue
            parts = line.strip().split()
            cmd, args = parts[0], parts[1:]
            if cmd.lower() in ("exit","quit"):
                cprint("Exiting ARGUS...", "YELLOW")
                break
            if cmd.lower() == "help":
                out = ["Available commands:"]
                for k in sorted(COMMANDS.keys()):
                    help_text = COMMAND_HELP.get(k, "")
                    out.append(f" {k} - {help_text}")
                cprint("\n".join(out), "CYAN")
                continue
            if cmd not in COMMANDS:
                cprint(f"Unknown command: {cmd}", "RED")
                continue
            res = _safe_call(cmd, args)
            if res is not None:
                cprint(str(res), "GREEN")
        except KeyboardInterrupt:
            cprint("\nKeyboardInterrupt detected. Type 'exit' to quit.", "YELLOW")
        except Exception as e:
            cprint(f"Error: {e}", "RED")

# -------------------------
# 8.7 CLI bootstrap if run directly
# -------------------------
if __name__ == "__main__":
    argus_cli_loop()

