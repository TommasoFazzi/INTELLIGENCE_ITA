#!/usr/bin/env python3
"""
Daily pipeline status check — runs at 9:00 AM via launchd.
Checks if the 8:00 AM pipeline ran successfully and sends a macOS notification.
"""
import os
import sys
import subprocess
from datetime import datetime, date
from pathlib import Path

BASE_DIR = Path("/Users/tommasofazzi/INTELLIGENCE_ITA/INTELLIGENCE_ITA")
LOGS_DIR = BASE_DIR / "logs"


def notify(title: str, message: str, sound: bool = False):
    """Send a macOS notification."""
    script = f'display notification "{message}" with title "{title}"'
    if sound:
        script += ' sound name "Ping"'
    subprocess.run(["osascript", "-e", script], capture_output=True)


def check_running_processes() -> list[str]:
    result = subprocess.run(
        ["ps", "aux"], capture_output=True, text=True
    )
    keywords = ["daily_pipeline", "process_nlp", "load_to_database",
                 "process_narratives", "generate_report"]
    running = []
    for line in result.stdout.splitlines():
        for kw in keywords:
            if kw in line and "grep" not in line:
                running.append(kw)
                break
    return running


def find_todays_log() -> tuple[Path | None, str]:
    """Return the most recent log file and its last modification info."""
    today = date.today().strftime("%Y-%m-%d")
    log_files = sorted(LOGS_DIR.glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True)

    for log_file in log_files[:5]:
        mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
        if mtime.date() == date.today():
            return log_file, mtime.strftime("%H:%M:%S")

    return None, ""


def check_last_log_content(log_file: Path) -> dict:
    """Read last lines of log to determine success/failure."""
    try:
        lines = log_file.read_text(errors="replace").splitlines()
        tail = "\n".join(lines[-30:])
        success = any(x in tail for x in ["Pipeline completed", "SUCCESS", "completato", "✓"])
        error = any(x in tail for x in ["ERROR", "CRITICAL", "Traceback", "Exception", "FAILED"])
        return {"success": success, "error": error, "tail": tail[-500:]}
    except Exception as e:
        return {"success": False, "error": False, "tail": str(e)}


def check_db_last_report() -> str:
    """Query DB for the most recent report timestamp."""
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        return "DATABASE_URL not set"
    try:
        import psycopg2
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        cur.execute("SELECT created_at FROM reports ORDER BY created_at DESC LIMIT 1")
        row = cur.fetchone()
        conn.close()
        if row:
            return row[0].strftime("%Y-%m-%d %H:%M")
        return "no reports found"
    except Exception as e:
        return f"DB error: {e}"


def main():
    now = datetime.now()
    print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Pipeline status check starting")

    # 1. Check running processes
    running = check_running_processes()
    if running:
        msg = f"Still running: {', '.join(running)}"
        print(f"  ⏳ {msg}")
        notify("Intelligence Pipeline", f"⏳ {msg}", sound=False)
        return

    # 2. Find today's log
    log_file, log_mtime = find_todays_log()
    if not log_file:
        msg = "No log file found for today — pipeline may not have started"
        print(f"  ⚠️  {msg}")
        notify("Intelligence Pipeline", f"⚠️ {msg}", sound=True)
        return

    # 3. Check log content
    result = check_last_log_content(log_file)

    # 4. Check DB for last report
    last_report = check_db_last_report()

    # 5. Build status message
    if result["success"]:
        status = "✅ Completed successfully"
        sound = False
    elif result["error"]:
        status = "❌ Errors detected"
        sound = True
    else:
        status = "⚠️ Unknown state"
        sound = True

    summary = f"{status} | Log: {log_mtime} | Last report: {last_report}"
    print(f"  {summary}")
    if result["error"]:
        print(f"  Log tail: {result['tail']}")

    notify("Intelligence Pipeline", summary, sound=sound)


if __name__ == "__main__":
    # Load .env if present
    env_file = BASE_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

    main()
