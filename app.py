"""Local web chatbot for the Day 8 RAG pipeline.

Run:
    python app.py --host 127.0.0.1 --port 8501
"""

from __future__ import annotations

import argparse
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.task10_generation import generate_with_citation


INDEX_HTML = r"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>DrugLaw RAG Chat</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {
      color-scheme: dark;
      --bg-0: #0a0e1a;
      --bg-1: #111827;
      --bg-2: #1a2235;
      --bg-3: #232d44;
      --ink: #f1f5f9;
      --ink-dim: #cbd5e1;
      --muted: #94a3b8;
      --line: rgba(148, 163, 184, 0.15);
      --line-strong: rgba(148, 163, 184, 0.28);
      --accent: #8b5cf6;
      --accent-2: #06b6d4;
      --accent-glow: rgba(139, 92, 246, 0.35);
      --success: #10b981;
      --warn: #f59e0b;
      --danger: #f43f5e;
      --gradient: linear-gradient(135deg, #8b5cf6 0%, #06b6d4 100%);
      --gradient-soft: linear-gradient(135deg, rgba(139, 92, 246, 0.12) 0%, rgba(6, 182, 212, 0.08) 100%);
      --shadow-lg: 0 20px 60px rgba(0, 0, 0, 0.4);
      --shadow-glow: 0 0 30px var(--accent-glow);
      font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, sans-serif;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg-0);
      background-image:
        radial-gradient(circle at 15% 10%, rgba(139, 92, 246, 0.12) 0%, transparent 40%),
        radial-gradient(circle at 85% 80%, rgba(6, 182, 212, 0.1) 0%, transparent 40%);
      background-attachment: fixed;
      color: var(--ink);
      overflow: hidden;
    }

    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--line-strong); border-radius: 999px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--muted); }

    .app {
      height: 100vh;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 380px;
    }

    .chat {
      height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr auto;
      min-width: 0;
    }

    /* HEADER */
    header {
      border-bottom: 1px solid var(--line);
      background: rgba(10, 14, 26, 0.6);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      padding: 14px clamp(16px, 4vw, 32px);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      z-index: 10;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 14px;
      min-width: 0;
    }

    .mark {
      width: 44px;
      height: 44px;
      border-radius: 12px;
      background: var(--gradient);
      color: white;
      display: grid;
      place-items: center;
      font-weight: 800;
      font-size: 13px;
      letter-spacing: 0.5px;
      box-shadow: var(--shadow-glow);
      position: relative;
    }

    .mark::after {
      content: '';
      position: absolute;
      inset: -2px;
      border-radius: 14px;
      background: var(--gradient);
      opacity: 0.4;
      filter: blur(10px);
      z-index: -1;
    }

    h1 {
      margin: 0;
      font-size: 17px;
      font-weight: 700;
      letter-spacing: -0.2px;
      background: linear-gradient(135deg, #f1f5f9 0%, #cbd5e1 100%);
      -webkit-background-clip: text;
      background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .subtitle {
      margin: 2px 0 0;
      color: var(--muted);
      font-size: 12.5px;
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .status-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--success);
      box-shadow: 0 0 8px var(--success);
      animation: pulse 2s ease-in-out infinite;
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.5; }
    }

    .actions { display: flex; gap: 8px; }

    button {
      appearance: none;
      border: 1px solid var(--line-strong);
      background: var(--bg-2);
      color: var(--ink-dim);
      border-radius: 10px;
      min-height: 40px;
      padding: 0 14px;
      font: inherit;
      font-size: 13.5px;
      font-weight: 500;
      cursor: pointer;
      transition: all 180ms ease;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }

    button:hover {
      border-color: var(--accent);
      color: var(--ink);
      background: var(--bg-3);
      transform: translateY(-1px);
    }

    button.primary {
      background: var(--gradient);
      border-color: transparent;
      color: white;
      font-weight: 600;
      min-width: 96px;
      justify-content: center;
      box-shadow: 0 4px 16px rgba(139, 92, 246, 0.4);
    }

    button.primary:hover {
      box-shadow: 0 6px 24px rgba(139, 92, 246, 0.6);
      transform: translateY(-2px);
    }

    button:disabled { opacity: 0.5; cursor: wait; transform: none; }

    /* MESSAGES */
    .messages {
      overflow-y: auto;
      padding: 28px clamp(16px, 5vw, 48px);
      display: flex;
      flex-direction: column;
      gap: 20px;
      scroll-behavior: smooth;
    }

    .message-row {
      display: flex;
      gap: 12px;
      align-items: flex-start;
      animation: slideUp 320ms cubic-bezier(0.16, 1, 0.3, 1);
    }

    @keyframes slideUp {
      from { opacity: 0; transform: translateY(12px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .avatar {
      width: 36px;
      height: 36px;
      border-radius: 10px;
      display: grid;
      place-items: center;
      font-weight: 700;
      font-size: 13px;
      flex: 0 0 auto;
    }

    .avatar.bot {
      background: var(--gradient);
      color: white;
      box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3);
    }

    .avatar.you {
      background: var(--bg-3);
      color: var(--ink);
      border: 1px solid var(--line-strong);
    }

    .message-row.user {
      flex-direction: row-reverse;
    }

    .message {
      max-width: min(720px, 85%);
      border-radius: 14px;
      padding: 14px 18px;
      line-height: 1.6;
      font-size: 14.5px;
      white-space: pre-wrap;
      word-wrap: break-word;
    }

    .message.assistant {
      background: var(--bg-2);
      border: 1px solid var(--line);
      color: var(--ink-dim);
      border-top-left-radius: 4px;
    }

    .message.user {
      background: var(--gradient);
      color: white;
      border-top-right-radius: 4px;
      box-shadow: 0 4px 16px rgba(139, 92, 246, 0.25);
    }

    .meta {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 10px;
      padding-top: 10px;
      border-top: 1px solid var(--line);
    }

    .tag {
      border-radius: 999px;
      background: var(--gradient-soft);
      color: var(--accent-2);
      border: 1px solid rgba(6, 182, 212, 0.25);
      padding: 3px 10px;
      font-size: 11.5px;
      font-weight: 500;
    }

    /* COMPOSER */
    .composer {
      border-top: 1px solid var(--line);
      background: rgba(10, 14, 26, 0.7);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      padding: 14px clamp(16px, 5vw, 48px) 20px;
    }

    .quick {
      display: flex;
      gap: 8px;
      overflow-x: auto;
      padding-bottom: 12px;
      scrollbar-width: thin;
    }

    .quick button {
      white-space: nowrap;
      font-size: 12.5px;
      min-height: 34px;
      padding: 0 12px;
      background: var(--gradient-soft);
      color: var(--ink-dim);
      border-color: var(--line);
    }

    .quick button:hover {
      color: white;
      background: var(--bg-3);
    }

    form {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      align-items: end;
      background: var(--bg-1);
      border: 1px solid var(--line-strong);
      border-radius: 14px;
      padding: 8px;
      transition: border-color 180ms ease, box-shadow 180ms ease;
    }

    form:focus-within {
      border-color: var(--accent);
      box-shadow: 0 0 0 4px var(--accent-glow);
    }

    textarea {
      width: 100%;
      min-height: 46px;
      max-height: 180px;
      resize: none;
      border: none;
      background: transparent;
      color: var(--ink);
      padding: 10px 12px;
      font: inherit;
      font-size: 14.5px;
      line-height: 1.5;
      outline: none;
    }

    textarea::placeholder { color: var(--muted); }

    /* SIDEBAR */
    aside {
      border-left: 1px solid var(--line);
      background: rgba(17, 24, 39, 0.4);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      height: 100vh;
      padding: 20px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }

    .panel-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 4px;
    }

    .panel-title {
      margin: 0;
      font-size: 14px;
      font-weight: 600;
      color: var(--ink);
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .panel-title::before {
      content: '';
      width: 3px;
      height: 16px;
      background: var(--gradient);
      border-radius: 2px;
    }

    .badge {
      background: var(--gradient-soft);
      color: var(--accent-2);
      border: 1px solid rgba(6, 182, 212, 0.25);
      padding: 2px 9px;
      border-radius: 999px;
      font-size: 11.5px;
      font-weight: 600;
    }

    .source {
      border: 1px solid var(--line);
      border-radius: 12px;
      background: var(--bg-2);
      padding: 14px;
      transition: all 180ms ease;
      position: relative;
      overflow: hidden;
    }

    .source::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      width: 3px;
      height: 100%;
      background: var(--gradient);
      opacity: 0.6;
    }

    .source:hover {
      border-color: var(--accent);
      transform: translateX(-2px);
    }

    .source .top {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: flex-start;
      margin-bottom: 8px;
    }

    .source strong {
      font-size: 13px;
      font-weight: 600;
      color: var(--ink);
      line-height: 1.4;
      overflow-wrap: anywhere;
    }

    .score {
      color: white;
      background: var(--gradient);
      border-radius: 999px;
      padding: 3px 9px;
      font-size: 11.5px;
      font-weight: 600;
      font-family: 'JetBrains Mono', monospace;
      flex: 0 0 auto;
      box-shadow: 0 2px 8px rgba(139, 92, 246, 0.3);
    }

    .source .type {
      display: inline-block;
      font-size: 11px;
      color: var(--muted);
      background: var(--bg-3);
      padding: 2px 8px;
      border-radius: 6px;
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      font-weight: 500;
    }

    .source p {
      margin: 0;
      color: var(--ink-dim);
      font-size: 12.5px;
      line-height: 1.55;
      overflow-wrap: anywhere;
    }

    .empty {
      color: var(--muted);
      font-size: 13.5px;
      line-height: 1.5;
      border: 1px dashed var(--line-strong);
      border-radius: 12px;
      padding: 28px 18px;
      background: rgba(26, 34, 53, 0.4);
      text-align: center;
    }

    .empty::before {
      content: '📚';
      display: block;
      font-size: 28px;
      margin-bottom: 8px;
      opacity: 0.6;
    }

    .status {
      min-height: 20px;
      margin-top: 10px;
      font-size: 12.5px;
      color: var(--muted);
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .status.loading::before {
      content: '';
      width: 12px;
      height: 12px;
      border: 2px solid var(--line-strong);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin 700ms linear infinite;
    }

    @keyframes spin { to { transform: rotate(360deg); } }

    .status.error { color: var(--danger); }

    .typing {
      display: inline-flex;
      gap: 4px;
      padding: 6px 0;
    }

    .typing span {
      width: 7px;
      height: 7px;
      background: var(--accent);
      border-radius: 50%;
      animation: bounce 1.2s ease-in-out infinite;
    }

    .typing span:nth-child(2) { animation-delay: 0.15s; }
    .typing span:nth-child(3) { animation-delay: 0.3s; }

    @keyframes bounce {
      0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
      30% { transform: translateY(-6px); opacity: 1; }
    }

    @media (max-width: 960px) {
      body { overflow: auto; }
      .app { grid-template-columns: 1fr; height: auto; }
      .chat { height: auto; min-height: 100vh; }
      aside { height: auto; border-left: 0; border-top: 1px solid var(--line); }
    }

    @media (max-width: 560px) {
      form { grid-template-columns: 1fr; }
      button.primary { width: 100%; }
      .message { max-width: 100%; }
      header { padding: 12px 16px; }
      h1 { font-size: 15px; }
      .subtitle { font-size: 11.5px; }
    }
  </style>
</head>
<body>
  <main class="app">
    <section class="chat">
      <header>
        <div class="brand">
          <div class="mark">RAG</div>
          <div>
            <h1>DrugLaw RAG Chat</h1>
            <p class="subtitle"><span class="status-dot"></span> Trợ lý pháp luật ma túy · Online</p>
          </div>
        </div>
        <div class="actions">
          <button id="evalBtn" type="button">📊 Evaluation</button>
          <button id="clearBtn" type="button">＋ New Chat</button>
        </div>
      </header>

      <div id="messages" class="messages" aria-live="polite"></div>

      <div class="composer">
        <div class="quick">
          <button type="button" data-q="Hình phạt cho tội tàng trữ trái phép chất ma túy?">💊 Tàng trữ ma túy</button>
          <button type="button" data-q="Các biện pháp cai nghiện ma túy gồm những gì?">🏥 Cai nghiện</button>
          <button type="button" data-q="Diễn viên Hữu Tín bị truy tố về tội gì?">🎬 Hữu Tín</button>
          <button type="button" data-q="Châu Việt Cường bị khởi tố theo tội danh nào?">⚖️ Châu Việt Cường</button>
        </div>
        <form id="chatForm">
          <textarea id="question" rows="1" placeholder="Hỏi về pháp luật ma túy..." required></textarea>
          <button class="primary" id="sendBtn" type="submit">
            <span>Gửi</span>
            <span>↗</span>
          </button>
        </form>
        <div id="status" class="status"></div>
      </div>
    </section>

    <aside>
      <div class="panel-header">
        <h2 class="panel-title">Nguồn tham khảo</h2>
        <span id="sourceCount" class="badge">0</span>
      </div>
      <div id="sources" class="empty">Chưa có nguồn được truy xuất.</div>
    </aside>
  </main>

  <script>
    const messagesEl = document.getElementById("messages");
    const sourcesEl = document.getElementById("sources");
    const sourceCountEl = document.getElementById("sourceCount");
    const form = document.getElementById("chatForm");
    const questionEl = document.getElementById("question");
    const sendBtn = document.getElementById("sendBtn");
    const clearBtn = document.getElementById("clearBtn");
    const evalBtn = document.getElementById("evalBtn");
    const statusEl = document.getElementById("status");
    const history = [];

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function addMessage(role, content, meta = []) {
      const row = document.createElement("div");
      row.className = `message-row ${role}`;

      const avatar = document.createElement("div");
      avatar.className = `avatar ${role === "user" ? "you" : "bot"}`;
      avatar.textContent = role === "user" ? "Bạn" : "AI";

      const node = document.createElement("div");
      node.className = `message ${role}`;
      node.innerHTML = escapeHtml(content);

      if (meta.length) {
        const metaEl = document.createElement("div");
        metaEl.className = "meta";
        metaEl.innerHTML = meta.map(item => `<span class="tag">${escapeHtml(item)}</span>`).join("");
        node.appendChild(metaEl);
      }

      row.appendChild(avatar);
      row.appendChild(node);
      messagesEl.appendChild(row);
      messagesEl.scrollTop = messagesEl.scrollHeight;
      return row;
    }

    function addTypingIndicator() {
      const row = document.createElement("div");
      row.className = "message-row assistant";
      row.id = "typing-row";
      row.innerHTML = `
        <div class="avatar bot">AI</div>
        <div class="message assistant">
          <div class="typing"><span></span><span></span><span></span></div>
        </div>`;
      messagesEl.appendChild(row);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function removeTypingIndicator() {
      const t = document.getElementById("typing-row");
      if (t) t.remove();
    }

    function renderSources(sources) {
      const count = (sources || []).length;
      sourceCountEl.textContent = count;
      if (!count) {
        sourcesEl.className = "empty";
        sourcesEl.textContent = "Chưa có nguồn được truy xuất.";
        return;
      }
      sourcesEl.className = "";
      sourcesEl.innerHTML = sources.map((source, index) => {
        const metadata = source.metadata || {};
        const title = metadata.source || metadata.filename || metadata.path || `Source ${index + 1}`;
        const type = metadata.type || metadata.doc_type || source.source || "unknown";
        const score = Number(source.score || 0).toFixed(3);
        const text = String(source.content || "").replace(/\s+/g, " ").slice(0, 260);
        return `
          <article class="source">
            <div class="top">
              <strong>${escapeHtml(title)}</strong>
              <span class="score">${escapeHtml(score)}</span>
            </div>
            <span class="type">${escapeHtml(type)}</span>
            <p>${escapeHtml(text)}</p>
          </article>
        `;
      }).join("");
    }

    async function ask(question) {
      statusEl.className = "status loading";
      statusEl.textContent = "Đang truy xuất tài liệu...";
      sendBtn.disabled = true;
      addMessage("user", question);
      addTypingIndicator();

      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question, history })
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        removeTypingIndicator();
        addMessage("assistant", data.answer, [
          `🔍 ${data.retrieval_source || "none"}`,
          `📄 ${(data.sources || []).length} nguồn`
        ]);
        renderSources(data.sources || []);
        history.push({ role: "user", content: question });
        history.push({ role: "assistant", content: data.answer });
        while (history.length > 10) history.shift();
        statusEl.className = "status";
        statusEl.textContent = "";
      } catch (error) {
        removeTypingIndicator();
        statusEl.className = "status error";
        statusEl.textContent = "⚠️ Không thể xử lý câu hỏi.";
        addMessage("assistant", "Xin lỗi, tôi không thể xử lý yêu cầu này lúc này.");
      } finally {
        sendBtn.disabled = false;
        questionEl.focus();
      }
    }

    form.addEventListener("submit", event => {
      event.preventDefault();
      const question = questionEl.value.trim();
      if (!question) return;
      questionEl.value = "";
      questionEl.style.height = "auto";
      ask(question);
    });

    questionEl.addEventListener("input", () => {
      questionEl.style.height = "auto";
      questionEl.style.height = Math.min(questionEl.scrollHeight, 180) + "px";
    });

    questionEl.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        form.requestSubmit();
      }
    });

    document.querySelectorAll("[data-q]").forEach(button => {
      button.addEventListener("click", () => ask(button.dataset.q));
    });

    clearBtn.addEventListener("click", () => {
      history.splice(0, history.length);
      messagesEl.innerHTML = "";
      renderSources([]);
      statusEl.textContent = "";
      questionEl.focus();
      addMessage("assistant", "Xin chào! Tôi đã sẵn sàng trả lời câu hỏi mới của bạn.", ["hybrid retrieval", "citations"]);
    });

    evalBtn.addEventListener("click", async () => {
      statusEl.className = "status loading";
      statusEl.textContent = "Đang đọc kết quả evaluation...";
      try {
        const response = await fetch("/api/evaluation");
        const data = await response.json();
        addMessage("assistant", data.summary, ["📊 evaluation"]);
        statusEl.className = "status";
        statusEl.textContent = "";
      } catch {
        statusEl.className = "status error";
        statusEl.textContent = "⚠️ Không đọc được evaluation.";
      }
    });

    addMessage(
      "assistant",
      "Xin chào! 👋 Tôi là trợ lý RAG về pháp luật ma túy. Hãy đặt câu hỏi hoặc chọn gợi ý phía dưới — tôi sẽ trả lời kèm citation từ corpus hiện có.",
      ["hybrid retrieval", "citations"]
    );
    questionEl.focus();
  </script>
</body>
</html>
"""


def summarize_history(history: list[dict], max_turns: int = 4) -> str:
    """Compact recent turns so follow-up questions carry local context."""
    recent = history[-max_turns * 2 :]
    lines = []
    for item in recent:
        role = item.get("role", "")
        content = " ".join(str(item.get("content", "")).split())
        if role in {"user", "assistant"} and content:
            lines.append(f"{role}: {content[:220]}")
    return "\n".join(lines)


def build_contextual_question(question: str, history: list[dict]) -> str:
    summary = summarize_history(history)
    if not summary:
        return question
    return f"Ngữ cảnh hội thoại gần đây:\n{summary}\n\nCâu hỏi hiện tại: {question}"


def evaluation_summary() -> str:
    results_path = PROJECT_ROOT / "group_project" / "evaluation" / "results.md"
    if not results_path.exists():
        return "Chưa có báo cáo evaluation."
    text = results_path.read_text(encoding="utf-8")
    lines = []
    capture = False
    for line in text.splitlines():
        if line.startswith("## Overall Scores"):
            capture = True
        if capture:
            lines.append(line)
        if capture and line.startswith("## A/B Comparison"):
            break
    return "\n".join(lines).strip() or text[:1200]


class RAGChatHandler(BaseHTTPRequestHandler):
    server_version = "DrugLawRAG/1.0"

    def log_message(self, format: str, *args):  # noqa: A002
        return

    def send_json(self, payload: dict, status: int = HTTPStatus.OK):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_html(self):
        body = INDEX_HTML.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self.send_html()
            return
        if path == "/api/health":
            self.send_json({"status": "ok"})
            return
        if path == "/api/evaluation":
            self.send_json({"summary": evaluation_summary()})
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):  # noqa: N802
        path = urlparse(self.path).path
        if path != "/api/chat":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            question = str(payload.get("question", "")).strip()
            history = payload.get("history", [])
            if not question:
                self.send_json({"error": "question is required"}, HTTPStatus.BAD_REQUEST)
                return

            contextual_question = build_contextual_question(question, history if isinstance(history, list) else [])
            result = generate_with_citation(contextual_question, top_k=5)
            self.send_json(
                {
                    "answer": result.get("answer", ""),
                    "sources": result.get("sources", []),
                    "retrieval_source": result.get("retrieval_source", "none"),
                }
            )
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main():
    parser = argparse.ArgumentParser(description="Run the Day 8 RAG web chatbot.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8501, type=int)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), RAGChatHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"DrugLaw RAG Chat running at {url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()