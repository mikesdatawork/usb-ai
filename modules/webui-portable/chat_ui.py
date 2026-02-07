#!/usr/bin/env python3
"""
USB-AI Chat Interface
Minimal Flask + HTMX chat UI for Ollama.

Modular design:
- chat_ui.py: Main Flask app
- llm_monitor.py: LLM state monitoring
- templates embedded for portability
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Generator, Optional

# Configure logging
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "chat_ui.log")),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("chat_ui")

# Portable dependencies
APP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app")
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import requests
from flask import Flask, Response, render_template_string, request, session, jsonify

# Optional monitor import
try:
    from llm_monitor import LLMMonitor
    MONITOR_ENABLED = True
except ImportError:
    MONITOR_ENABLED = False

# Configuration
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
DEFAULT_PORT = 3000
DEFAULT_HOST = "127.0.0.1"

# Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))

# Global monitor
monitor: Optional[LLMMonitor] = None


# =============================================================================
# CSS - Minimal flat design, no curves
# =============================================================================
CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: Arial, Helvetica, sans-serif;
    font-weight: 400;
    background: #1a1a1a;
    color: #e0e0e0;
    height: 100vh;
    display: flex;
    flex-direction: column;
}

.header {
    background: #1a1a1a;
    padding: 10px 16px;
    border-bottom: 1px solid #333;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    flex-wrap: wrap;
}

.header h1 { color: #ffa222; font-size: 1rem; font-weight: 400; }

.controls { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }

select, input, textarea, button {
    font-family: Arial, Helvetica, sans-serif;
    font-size: 13px;
    border-radius: 0;
}

select {
    background: #1a1a1a;
    color: #e0e0e0;
    border: 1px solid #333;
    padding: 5px 8px;
    min-width: 160px;
}

select:focus { outline: none; border-color: #ffa222; }

button {
    background: #1a1a1a;
    color: #888;
    border: 1px solid #333;
    padding: 5px 10px;
    cursor: pointer;
}

button:hover { border-color: #ffa222; color: #ffa222; }
button:disabled { opacity: 0.4; cursor: not-allowed; }

button.stop { border-color: #f44336; color: #f44336; }
button.stop:hover:not(:disabled) { background: #f44336; color: #1a1a1a; }
button.stop:disabled { opacity: 0.3; cursor: not-allowed; }
button.stop.active { animation: pulse 1s infinite; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }

.status { display: flex; align-items: center; gap: 5px; font-size: 12px; color: #666; }
.dot { width: 6px; height: 6px; background: #444; }
.dot.on { background: #4caf50; }
.dot.off { background: #f44336; }

.chat {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 16px;
}

.msg {
    max-width: 80%;
    padding: 10px 12px;
    line-height: 1.5;
    white-space: pre-wrap;
    word-wrap: break-word;
}

.msg.user {
    background: #1a1a1a;
    border: 1px solid #ffa222;
    color: #e0e0e0;
    align-self: flex-end;
}

.msg.assistant {
    background: #1a1a1a;
    border: none;
    align-self: flex-start;
}

.msg.error {
    background: #1a1a1a;
    border: 1px solid #f44336;
    color: #f44336;
    align-self: center;
    font-size: 13px;
}

.msg.streaming { border-color: #ffa222; }

.empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: #444;
}

.empty h2 { color: #ffa222; font-weight: 400; font-size: 1rem; margin-bottom: 4px; }

.input-area {
    background: #1a1a1a;
    padding: 10px 16px;
    border-top: 1px solid #333;
    display: flex;
    gap: 8px;
}

.input-area textarea {
    flex: 1;
    background: #1a1a1a;
    color: #e0e0e0;
    border: 1px solid #333;
    padding: 8px 10px;
    resize: none;
    min-height: 38px;
    max-height: 100px;
}

.input-area textarea:focus { outline: none; border-color: #ffa222; }
.input-area textarea::placeholder { color: #555; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #1a1a1a; }
::-webkit-scrollbar-thumb { background: #333; }

.typing { display: inline; }
.typing::after {
    content: '...';
    animation: dots 1s infinite;
}
@keyframes dots {
    0%, 20% { content: '.'; }
    40% { content: '..'; }
    60%, 100% { content: '...'; }
}
"""


# =============================================================================
# HTML Template
# =============================================================================
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>USB-AI</title>
    <script src="https://unpkg.com/htmx.org@1.9.10"></script>
    <style>""" + CSS + """</style>
</head>
<body>
    <div class="header">
        <h1>USB-AI</h1>
        <div class="controls">
            <select id="model" hx-get="/api/models" hx-trigger="load" hx-swap="innerHTML">
                <option value="">Loading...</option>
            </select>
            <button hx-get="/api/models" hx-target="#model" hx-swap="innerHTML">Refresh</button>
            <div class="status" hx-get="/api/status" hx-trigger="load, every 10s" hx-swap="innerHTML">
                <span class="dot"></span><span>...</span>
            </div>
            <button hx-post="/api/clear" hx-target="#messages" hx-swap="innerHTML">Clear</button>
            <a href="/monitor" style="color:#888;font-size:13px;">Monitor</a>
        </div>
    </div>

    <div class="chat" id="messages">
        {% if messages %}
            {% for m in messages %}
                <div class="msg {{ m.role }}">{{ m.content }}</div>
            {% endfor %}
        {% else %}
            <div class="empty"><h2>USB-AI</h2><p>Select a model to start</p></div>
        {% endif %}
    </div>

    <div class="input-area">
        <textarea id="input" placeholder="Message (Enter to send)" rows="1"></textarea>
        <button id="send" onclick="send()">Send</button>
        <button id="stop" class="stop" onclick="stop()" disabled>Stop</button>
    </div>

    <script>
    let controller = null;

    function send() {
        const input = document.getElementById('input');
        const model = document.getElementById('model').value;
        const msg = input.value.trim();
        if (!msg || !model) return;

        input.disabled = true;
        document.getElementById('send').disabled = true;
        const stopBtn = document.getElementById('stop');
        stopBtn.disabled = false;
        stopBtn.classList.add('active');

        const chat = document.getElementById('messages');
        const empty = chat.querySelector('.empty');
        if (empty) empty.remove();

        const userDiv = document.createElement('div');
        userDiv.className = 'msg user';
        userDiv.textContent = msg;
        chat.appendChild(userDiv);

        const assistDiv = document.createElement('div');
        assistDiv.className = 'msg assistant streaming';
        assistDiv.innerHTML = '<span class="typing"></span>';
        chat.appendChild(assistDiv);
        chat.scrollTop = chat.scrollHeight;

        input.value = '';

        controller = new AbortController();

        fetch('/api/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({message: msg, model: model}),
            signal: controller.signal
        }).then(r => {
            const reader = r.body.getReader();
            const decoder = new TextDecoder();
            let content = '';

            function read() {
                reader.read().then(({done, value}) => {
                    if (done) {
                        finish();
                        return;
                    }
                    const text = decoder.decode(value, {stream: true});
                    for (const line of text.split('\\n')) {
                        if (line.startsWith('data: ')) {
                            try {
                                const d = JSON.parse(line.slice(6));
                                if (d.content) {
                                    content += d.content;
                                    assistDiv.textContent = content;
                                    chat.scrollTop = chat.scrollHeight;
                                }
                                if (d.done && content) {
                                    // Save assistant response to session
                                    fetch('/api/save_response', {
                                        method: 'POST',
                                        headers: {'Content-Type': 'application/json'},
                                        body: JSON.stringify({response: content})
                                    });
                                }
                                if (d.error) {
                                    assistDiv.className = 'msg error';
                                    assistDiv.textContent = d.error;
                                }
                            } catch(e) {}
                        }
                    }
                    read();
                }).catch(e => {
                    if (e.name !== 'AbortError') {
                        assistDiv.className = 'msg error';
                        assistDiv.textContent = e.message;
                    }
                    finish();
                });
            }
            read();
        }).catch(e => {
            if (e.name !== 'AbortError') {
                assistDiv.className = 'msg error';
                assistDiv.textContent = e.message;
            }
            finish();
        });
    }

    function stop() {
        if (controller) {
            controller.abort();
            controller = null;
        }
        finish();
    }

    function finish() {
        const input = document.getElementById('input');
        input.disabled = false;
        document.getElementById('send').disabled = false;
        const stopBtn = document.getElementById('stop');
        stopBtn.disabled = true;
        stopBtn.classList.remove('active');
        document.querySelectorAll('.msg.streaming').forEach(el => el.classList.remove('streaming'));
        input.focus();
    }

    document.getElementById('input').addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
    });

    document.getElementById('input').addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 100) + 'px';
    });
    </script>
</body>
</html>"""


# =============================================================================
# Ollama API Functions
# =============================================================================

def ollama_models() -> list[str]:
    """Get available models from Ollama."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if r.ok:
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass
    return []


def ollama_status() -> tuple[bool, str]:
    """Check Ollama connection."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        if r.ok:
            n = len(r.json().get("models", []))
            return True, f"{n} models"
    except Exception:
        pass
    return False, "Offline"


def ollama_chat(message: str, history: list, model: str) -> Generator[str, None, None]:
    """Stream chat from Ollama with monitoring."""
    messages = [{"role": m["role"], "content": m["content"]} for m in history]
    messages.append({"role": "user", "content": message})

    start = time.time()
    tokens = 0
    first_token = None

    # Start monitoring
    req_id = None
    if monitor:
        req_id = monitor._start_request(model, message[:100], timeout=300)

    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": model, "messages": messages, "stream": True},
            stream=True,
            timeout=300
        )

        # Check HTTP status
        if not r.ok:
            try:
                err_data = r.json()
                error_msg = err_data.get("error", f"HTTP {r.status_code}")
            except:
                error_msg = f"HTTP {r.status_code}"
            log.error(f"OLLAMA HTTP ERROR | Model: {model} | {error_msg}")
            if monitor and req_id:
                monitor._end_request(req_id, success=False, error=error_msg)
            yield f"[ERROR]{error_msg}"
            return

        for line in r.iter_lines():
            if line:
                data = json.loads(line)
                # Check for error response
                if "error" in data:
                    error_msg = data["error"]
                    log.error(f"OLLAMA ERROR | Model: {model} | {error_msg}")
                    if monitor and req_id:
                        monitor._end_request(req_id, success=False, error=error_msg)
                    yield f"[ERROR]{error_msg}"
                    return
                if "message" in data and "content" in data["message"]:
                    tokens += 1
                    if first_token is None:
                        first_token = time.time()
                        ttft = first_token - start
                        log.info(f"TTFT: {ttft:.2f}s | Model: {model}")
                    yield data["message"]["content"]
                if data.get("done"):
                    break

        elapsed = time.time() - start
        tps = tokens / elapsed if elapsed > 0 else 0
        ttft_val = (first_token - start) if first_token else 0
        log.info(f"DONE | Model: {model} | Tokens: {tokens} | Time: {elapsed:.1f}s | TTFT: {ttft_val:.2f}s | Speed: {tps:.1f} tok/s")

        if monitor and req_id:
            monitor._end_request(req_id, success=True)

    except requests.exceptions.Timeout:
        elapsed = time.time() - start
        log.error(f"TIMEOUT | Model: {model} | After: {elapsed:.0f}s")
        if monitor and req_id:
            monitor._end_request(req_id, success=False, error="Timeout - model hanging")
        yield "[ERROR]Timeout - model may be hanging"
    except requests.exceptions.ConnectionError:
        log.error(f"CONNECTION ERROR | Model: {model}")
        if monitor and req_id:
            monitor._end_request(req_id, success=False, error="Connection failed")
        yield "[ERROR]Cannot connect to Ollama"
    except Exception as e:
        elapsed = time.time() - start
        log.error(f"ERROR | Model: {model} | Time: {elapsed:.1f}s | {e}")
        if monitor and req_id:
            monitor._end_request(req_id, success=False, error=str(e))
        yield f"[ERROR]{e}"


# =============================================================================
# Routes
# =============================================================================

@app.route("/")
def index():
    return render_template_string(HTML, messages=session.get("messages", []))


@app.route("/api/models")
def api_models():
    models = ollama_models()
    if not models:
        return '<option value="">No models</option>'
    return "\n".join(f'<option value="{m}">{m}</option>' for m in models)


@app.route("/api/status")
def api_status():
    ok, text = ollama_status()
    cls = "on" if ok else "off"
    return f'<span class="dot {cls}"></span><span>{text}</span>'


@app.route("/api/clear", methods=["POST"])
def api_clear():
    session["messages"] = []
    return '<div class="empty"><h2>USB-AI</h2><p>Select a model to start</p></div>'


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json()
    msg = data.get("message", "").strip()
    model = data.get("model", "")

    if not msg or not model:
        return Response("data: " + json.dumps({"error": "Missing message or model"}) + "\n\n",
                        mimetype="text/event-stream")

    if "messages" not in session:
        session["messages"] = []

    # Capture history before generator (session not available in generator)
    history = list(session.get("messages", []))
    session["messages"].append({"role": "user", "content": msg})
    session.modified = True

    def generate():
        full = ""
        error = False
        for chunk in ollama_chat(msg, history, model):
            if chunk.startswith("[ERROR]"):
                error = True
                yield f"data: {json.dumps({'error': chunk[7:]})}\n\n"
                break
            full += chunk
            yield f"data: {json.dumps({'content': chunk})}\n\n"

        # Note: Can't update session from generator (outside request context)
        # Client should handle storing assistant response if needed
        yield f'data: {json.dumps({"done": True, "full_response": full if not error else ""})}\n\n'

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/save_response", methods=["POST"])
def api_save_response():
    """Save assistant response to session history."""
    data = request.get_json()
    response = data.get("response", "").strip()

    if response and "messages" in session:
        session["messages"].append({"role": "assistant", "content": response})
        session.modified = True
        return jsonify({"status": "ok"})

    return jsonify({"status": "no response to save"})


@app.route("/monitor")
def monitor_page():
    if not monitor:
        return jsonify({"status": "Monitor not enabled"})

    summary = monitor.get_summary()

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>LLM Monitor</title>
    <meta http-equiv="refresh" content="5">
    <style>
        body {{ font-family: Arial; background: #1a1a1a; color: #e0e0e0; padding: 20px; }}
        h1, h2 {{ color: #ffa222; font-weight: 400; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #333; padding: 8px; text-align: left; }}
        th {{ background: #242424; }}
        .ok {{ color: #4caf50; }}
        .warn {{ color: #ff9800; }}
        .err {{ color: #f44336; }}
        a {{ color: #ffa222; }}
    </style>
</head>
<body>
    <h1>LLM Monitor</h1>
    <p style="color:#666">Refreshes every 5s | <a href="/">Back to Chat</a></p>
    <p>Time: {summary['timestamp']}</p>
    <h2>Status</h2>
    <ul>
        <li class="ok">Ready: {summary['ready']}</li>
        <li>Processing: {summary['processing']}</li>
        <li class="warn">Hanging: {summary['hanging']}</li>
        <li class="err">Errors: {summary['errors']}</li>
    </ul>
    <h2>Models</h2>
    <table>
        <tr><th>Model</th><th>State</th><th>Last Time</th><th>Avg Time</th><th>Requests</th><th>Errors</th></tr>"""

    for name, d in summary.get("models", {}).items():
        state = d["state"].upper()
        cls = "ok" if state in ("READY", "AVAILABLE") else ("warn" if state == "HANGING" else ("err" if state == "ERROR" else ""))
        html += f"""<tr>
            <td>{name}</td>
            <td class="{cls}">{state}</td>
            <td>{d['last_response_time_sec'] or '-'}s</td>
            <td>{d['avg_response_time_sec']}s</td>
            <td>{d['request_count']}</td>
            <td>{d['error_count']}</td>
        </tr>"""

    html += "</table></body></html>"
    return html


# =============================================================================
# Main
# =============================================================================

def main():
    global monitor

    parser = argparse.ArgumentParser(description="USB-AI Chat")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", type=str, default=DEFAULT_HOST)
    args = parser.parse_args()

    # Start monitor
    if MONITOR_ENABLED:
        monitor = LLMMonitor(OLLAMA_URL)
        monitor.start()
        print(f"[MONITOR] Started - http://{args.host}:{args.port}/monitor")

    print(f"\n{'='*50}")
    print(f"  USB-AI Chat")
    print(f"  http://{args.host}:{args.port}")
    print(f"{'='*50}\n")

    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    finally:
        if monitor:
            monitor.stop()


if __name__ == "__main__":
    main()
