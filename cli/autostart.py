"""
cli/autostart.py -- run Cascade from login, so people only type their usual commands
(`cascade install` / `cascade uninstall`).

The login entry starts the model server with `keep_listening` and `lazy`: it listens
at once but loads no model until the first request, and unloads them again after
`server.idle_minutes`, so an idle machine holds no model in RAM.

    Windows   HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run  (no admin)
    macOS     ~/Library/LaunchAgents/dev.cascade.server.plist
    Linux     ~/.config/systemd/user/cascade.service (or ~/.config/autostart without systemd)

`--as-ollama` additionally gives Ollama's own port 11434 to Cascade, so `ollama run`
and every Ollama-based app use Cascade with no setting at all. The real Ollama then
runs on 11435, started by Cascade (`ollama serve` with OLLAMA_HOST=127.0.0.1:11435),
and gets every request Cascade does not answer itself (other models, pull, list,
embeddings). On Windows Ollama's own login entry (Startup\\Ollama.lnk) is set aside
and put back by `cascade uninstall`; on macOS/Linux the user is told how to stop
Ollama's own service, which needs their consent (and on Linux root).

Standard library + psutil (already a dependency).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

APP_NAME = "Cascade"
OLLAMA_PORT = 11434
MOVED_OLLAMA_URL = "http://127.0.0.1:11435"
REPO = Path(__file__).resolve().parents[1]
STATE_DIR = Path.home() / ".cascade"
OLLAMA_PID_FILE = STATE_DIR / "ollama_upstream.pid"
SET_ASIDE_DIR = STATE_DIR / "set-aside-autostart"
LAUNCH_AGENT = Path.home() / "Library" / "LaunchAgents" / "dev.cascade.server.plist"
SYSTEMD_UNIT = Path.home() / ".config" / "systemd" / "user" / "cascade.service"
XDG_AUTOSTART = Path.home() / ".config" / "autostart" / "cascade.desktop"
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"


def launch_command() -> list[str]:
    """What the login entry runs: this file's `run`, with a console-less Python on Windows."""
    exe = Path(sys.executable)
    if os.name == "nt" and exe.with_name("pythonw.exe").exists():
        exe = exe.with_name("pythonw.exe")
    return [str(exe), str(REPO / "cli" / "autostart.py"), "run"]


# ---- the login entry ----------------------------------------------------------------


def install_entry(command: Optional[list[str]] = None) -> str:
    """Register `command` to run at login; returns where it was registered."""
    command = command or launch_command()
    if os.name == "nt":
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
            winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, subprocess.list2cmdline(command))
        return f"HKCU\\{RUN_KEY}\\{APP_NAME}"
    if sys.platform == "darwin":
        LAUNCH_AGENT.parent.mkdir(parents=True, exist_ok=True)
        LAUNCH_AGENT.write_text(_launch_agent_plist(command))
        subprocess.run(["launchctl", "load", "-w", str(LAUNCH_AGENT)], capture_output=True)
        return str(LAUNCH_AGENT)
    if shutil.which("systemctl"):
        SYSTEMD_UNIT.parent.mkdir(parents=True, exist_ok=True)
        SYSTEMD_UNIT.write_text(_systemd_unit(command))
        subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
        subprocess.run(["systemctl", "--user", "enable", "cascade.service"], capture_output=True)
        return str(SYSTEMD_UNIT)
    XDG_AUTOSTART.parent.mkdir(parents=True, exist_ok=True)
    XDG_AUTOSTART.write_text(f"[Desktop Entry]\nType=Application\nName={APP_NAME}\n"
                             f"Exec={' '.join(_quote(c) for c in command)}\nX-GNOME-Autostart-enabled=true\n")
    return str(XDG_AUTOSTART)


def remove_entry() -> bool:
    """Remove the login entry; True if there was one."""
    if os.name == "nt":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
                winreg.DeleteValue(key, APP_NAME)
            return True
        except FileNotFoundError:
            return False
    removed = False
    if LAUNCH_AGENT.exists():
        subprocess.run(["launchctl", "unload", "-w", str(LAUNCH_AGENT)], capture_output=True)
        LAUNCH_AGENT.unlink()
        removed = True
    if SYSTEMD_UNIT.exists():
        subprocess.run(["systemctl", "--user", "disable", "cascade.service"], capture_output=True)
        SYSTEMD_UNIT.unlink()
        removed = True
    if XDG_AUTOSTART.exists():
        XDG_AUTOSTART.unlink()
        removed = True
    return removed


def entry_installed() -> bool:
    if os.name == "nt":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
                winreg.QueryValueEx(key, APP_NAME)
            return True
        except FileNotFoundError:
            return False
    return LAUNCH_AGENT.exists() or SYSTEMD_UNIT.exists() or XDG_AUTOSTART.exists()


def _quote(arg: str) -> str:
    return f'"{arg}"' if " " in arg else arg


def _launch_agent_plist(command: list[str]) -> str:
    args = "".join(f"\n        <string>{a}</string>" for a in command)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>dev.cascade.server</string>
    <key>ProgramArguments</key>
    <array>{args}
    </array>
    <key>RunAtLoad</key>
    <true/>
</dict>
</plist>
"""


def _systemd_unit(command: list[str]) -> str:
    return (f"[Unit]\nDescription={APP_NAME} model server\n\n[Service]\n"
            f"ExecStart={' '.join(_quote(c) for c in command)}\nRestart=on-failure\n\n"
            f"[Install]\nWantedBy=default.target\n")


# ---- Ollama's own login entry ----------------------------------------------------------


def ollama_startup_shortcut() -> Optional[Path]:
    """Windows: the Startup-folder shortcut the Ollama installer creates."""
    appdata = os.environ.get("APPDATA")
    if os.name != "nt" or not appdata:
        return None
    return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" / "Ollama.lnk"


def set_aside_ollama_autostart() -> Optional[str]:
    """Stop Ollama's app from taking port 11434 at login. Returns what was done, or the manual
    step for this platform (None when there is nothing to do)."""
    link = ollama_startup_shortcut()
    if link is not None:
        if link.exists():
            SET_ASIDE_DIR.mkdir(parents=True, exist_ok=True)
            shutil.move(str(link), str(SET_ASIDE_DIR / link.name))
            return f"moved Ollama's login shortcut to {SET_ASIDE_DIR} (put back by `cascade uninstall`)"
        return None
    if sys.platform == "darwin":
        return ("remove Ollama from System Settings > General > Login Items, so Cascade can take port "
                f"{OLLAMA_PORT}; Cascade starts Ollama itself on {MOVED_OLLAMA_URL}")
    return (f"run `sudo systemctl disable --now ollama` so Cascade can take port {OLLAMA_PORT}; "
            f"Cascade starts Ollama itself on {MOVED_OLLAMA_URL}")


def restore_ollama_autostart() -> Optional[Path]:
    """Put Ollama's login shortcut back; returns it when it was restored."""
    link = ollama_startup_shortcut()
    saved = SET_ASIDE_DIR / "Ollama.lnk"
    if link is None or not saved.exists():
        return None
    link.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(saved), str(link))
    return link


# ---- the real Ollama on its moved port ---------------------------------------------------


def find_ollama() -> Optional[str]:
    found = shutil.which("ollama")
    if found:
        return found
    candidates = []
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA", "")
        candidates += [Path(local) / "Programs" / "Ollama" / "ollama.exe"]
    else:
        candidates += [Path("/usr/local/bin/ollama"), Path("/Applications/Ollama.app/Contents/Resources/ollama")]
    return next((str(c) for c in candidates if c.exists()), None)


def server_kind(url: str) -> Optional[str]:
    """'cascade', 'ollama' or None (nothing listening) for the server at `url`."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/api/version", timeout=2) as resp:
            return "cascade" if resp.headers.get("X-Cascade") else "ollama"
    except urllib.error.HTTPError as exc:
        return "cascade" if exc.headers.get("X-Cascade") else "ollama"
    except OSError:
        return None


def start_ollama(url: str = MOVED_OLLAMA_URL, ollama: Optional[str] = None) -> Optional[int]:
    """Run `ollama serve` on `url` unless something already answers there. Returns the new pid."""
    if server_kind(url) is not None:
        return None
    ollama = ollama or find_ollama()
    if ollama is None:
        print(f"Ollama not found; install it from https://ollama.com (Cascade passes other models to {url})",
              file=sys.stderr)
        return None
    parts = urllib.parse.urlsplit(url)
    env = dict(os.environ, OLLAMA_HOST=f"{parts.hostname}:{parts.port}")
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    log = open(STATE_DIR / "ollama.log", "ab")
    kwargs: dict = {"stdin": subprocess.DEVNULL, "stdout": log, "stderr": log, "env": env, "close_fds": True}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen([ollama, "serve"], **kwargs)
    OLLAMA_PID_FILE.write_text(str(proc.pid))
    return proc.pid


def stop_started_ollama() -> bool:
    """Stop the `ollama serve` that start_ollama started, if it still runs."""
    import psutil

    try:
        pid = int(OLLAMA_PID_FILE.read_text())
    except (OSError, ValueError):
        return False
    OLLAMA_PID_FILE.unlink(missing_ok=True)
    try:
        proc = psutil.Process(pid)
        if "ollama" in proc.name().lower():
            proc.terminate()
            return True
    except psutil.Error:
        pass
    return False


def stop_ollama_app() -> int:
    """Windows: end the Ollama tray app and its server, so Cascade can take port 11434.
    Only called by `cascade install --as-ollama`. Returns how many processes were ended."""
    import psutil

    keep = None
    try:
        keep = int(OLLAMA_PID_FILE.read_text())
    except (OSError, ValueError):
        pass
    ended = 0
    for proc in psutil.process_iter(["name", "pid"]):
        name = (proc.info.get("name") or "").lower()
        if name in ("ollama app.exe", "ollama.exe", "ollama") and proc.info["pid"] != keep:
            try:
                proc.terminate()
                ended += 1
            except psutil.Error:
                pass
    return ended


def start_ollama_app(link: Path) -> None:
    if os.name == "nt":
        os.startfile(str(link))  # noqa: S606 -- the user's own Ollama shortcut, put back


# ---- what the login entry runs -------------------------------------------------------------


def run() -> None:
    """Serve Cascade for this login session (lazy, keep_listening), logging to ~/.cascade/server.log."""
    os.chdir(REPO)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    log = open(STATE_DIR / "server.log", "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = log
    from cli import main

    main.serve(port=None, idle_minutes=None, keep_listening=True, lazy=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["run"]:
        run()
    else:
        print(json.dumps({"entry_installed": entry_installed(), "command": launch_command()}))
