"""Small standard-library web UI for the engineering intelligence demonstrator."""

from __future__ import annotations

import html
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from engineering_intelligence.demo.pipeline import run_demo


def build_dashboard_html(result: dict[str, Any]) -> str:
    """Build a single-page dashboard for demo results."""
    initial_json = html.escape(json.dumps(result, indent=2))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Engineering Intelligence Dashboard</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 0;
      background: #0b1020;
      color: #dbe7ff;
    }}
    header {{
      padding: 1rem 1.5rem;
      border-bottom: 1px solid #334;
      background: #101934;
    }}
    .content {{
      padding: 1rem 1.5rem 2rem;
      display: grid;
      gap: 1rem;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    }}
    .card {{
      background: #121d3b;
      border: 1px solid #2f3a5f;
      border-radius: 10px;
      padding: 1rem;
    }}
    button {{
      background: #4f7cff;
      border: none;
      color: white;
      border-radius: 8px;
      padding: 0.6rem 0.8rem;
      cursor: pointer;
    }}
    pre {{
      white-space: pre-wrap;
      word-wrap: break-word;
      max-height: 420px;
      overflow: auto;
      background: #0d1630;
      border: 1px solid #263455;
      border-radius: 8px;
      padding: 0.8rem;
    }}
    .label {{
      color: #8dacff;
      font-size: 0.9rem;
      margin-bottom: 0.4rem;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Engineering Intelligence Dashboard</h1>
    <p>Evidence-bound design closure demonstrator with explicit provenance and audit output.</p>
    <button id="rerun-btn">Re-run pipeline</button>
  </header>
  <main class="content">
    <section class="card">
      <div class="label">Selected design</div>
      <pre id="selected"></pre>
    </section>
    <section class="card">
      <div class="label">Closure classification</div>
      <pre id="closure"></pre>
    </section>
    <section class="card">
      <div class="label">Physics result</div>
      <pre id="physics"></pre>
    </section>
    <section class="card">
      <div class="label">Audit report</div>
      <pre id="audit"></pre>
    </section>
  </main>
  <script>
    let data = JSON.parse("{initial_json}");

    function render() {{
      document.getElementById("selected").textContent = JSON.stringify(data.selected_design, null, 2);
      document.getElementById("closure").textContent = JSON.stringify(data.closure, null, 2);
      document.getElementById("physics").textContent = JSON.stringify(data.physics_result, null, 2);
      document.getElementById("audit").textContent = JSON.stringify(data.audit, null, 2);
    }}

    async function rerun() {{
      const response = await fetch("/api/demo");
      if (!response.ok) {{
        throw new Error("API call failed");
      }}
      data = await response.json();
      render();
    }}

    document.getElementById("rerun-btn").addEventListener("click", () => {{
      rerun().catch((err) => alert(err.message));
    }});

    render();
  </script>
</body>
</html>
"""


class DemoRequestHandler(BaseHTTPRequestHandler):
    """Serve dashboard UI and a JSON API for rerunning the demo."""

    def do_GET(self) -> None:  # noqa: N802 (stdlib interface)
        if self.path == "/api/demo":
            payload = json.dumps(run_demo()).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if self.path == "/":
            page = build_dashboard_html(run_demo()).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        """Suppress default access log noise."""
        return


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the dashboard web server."""
    server = ThreadingHTTPServer((host, port), DemoRequestHandler)
    print(f"Serving dashboard on http://{host}:{port}")
    server.serve_forever()
