import { FormEvent, useMemo, useRef, useState } from "react";
import {
  ExternalLink,
  Gauge,
  Github,
  Loader2,
  RefreshCw,
  SendHorizontal,
  Square,
  Trash2,
  Zap
} from "lucide-react";
import {
  ChatMessage,
  GenerationSettings,
  LoadProgress,
  generateTalkieReply,
  hasWebGPU,
  loadTalkieRuntime,
  modelDefaults,
  resetTalkieRuntime
} from "./talkieClient";

const initialMessages: ChatMessage[] = [
  {
    role: "assistant",
    content: "Good day. I am ready."
  }
];

const defaultSettings: GenerationSettings = {
  maxNewTokens: 96,
  temperature: 0.7,
  topP: 0.9,
  topK: 50
};

function App() {
  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages);
  const [draft, setDraft] = useState("");
  const [settings, setSettings] = useState(defaultSettings);
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [status, setStatus] = useState("Idle");
  const [progress, setProgress] = useState<LoadProgress | null>(null);
  const [dtype, setDtype] = useState("unloaded");
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const defaults = useMemo(() => modelDefaults(), []);
  const gpuReady = hasWebGPU();

  const handleProgress = (event: LoadProgress) => {
    setProgress(event);
    if (event.status) setStatus(event.status);
  };

  const loadModel = async () => {
    setError(null);
    setLoading(true);
    setStatus("Loading");
    try {
      const runtime = await loadTalkieRuntime(handleProgress);
      setDtype(runtime.session.dtype);
      setStatus("Ready");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setStatus("Error");
    } finally {
      setLoading(false);
    }
  };

  const resetModel = async () => {
    setError(null);
    setLoading(true);
    setStatus("Resetting");
    try {
      await resetTalkieRuntime();
      setDtype("unloaded");
      setProgress(null);
      setStatus("Idle");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setStatus("Error");
    } finally {
      setLoading(false);
    }
  };

  const stopGeneration = () => {
    abortRef.current?.abort();
    abortRef.current = null;
    setGenerating(false);
    setStatus("Stopped");
  };

  const clearChat = () => {
    stopGeneration();
    setMessages(initialMessages);
  };

  const send = async (event: FormEvent) => {
    event.preventDefault();
    const prompt = draft.trim();
    if (!prompt || generating) return;

    setDraft("");
    setError(null);
    const controller = new AbortController();
    abortRef.current = controller;
    setGenerating(true);
    setStatus("Thinking");

    const nextMessages: ChatMessage[] = [...messages, { role: "user", content: prompt }, { role: "assistant", content: "" }];
    setMessages(nextMessages);

    try {
      await generateTalkieReply(
        nextMessages.slice(0, -1),
        settings,
        (text) => {
          setMessages((current) => {
            const copy = [...current];
            copy[copy.length - 1] = { role: "assistant", content: text };
            return copy;
          });
        },
        controller.signal,
        handleProgress
      );
      setStatus(controller.signal.aborted ? "Stopped" : "Ready");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setStatus("Error");
      setMessages((current) => current.slice(0, -1));
    } finally {
      abortRef.current = null;
      setGenerating(false);
    }
  };

  return (
    <main className="shell">
      <section className="workspace" aria-label="Talkie chat">
        <header className="topbar">
          <div>
            <h1>Talkie Quant WebGPU</h1>
            <p>{defaults.modelId}</p>
          </div>
          <div className="status-strip" aria-live="polite">
            <span className={gpuReady ? "status ok" : "status bad"}>
              <Zap size={16} />
              {gpuReady ? "WebGPU" : "No WebGPU"}
            </span>
            <span className="status">
              <Gauge size={16} />
              {dtype}
            </span>
            <span className="status">{status}</span>
          </div>
        </header>

        <div className="body">
          <aside className="controls" aria-label="Generation controls">
            <button type="button" className="primary" onClick={loadModel} disabled={loading || generating || !gpuReady} title="Load model">
              {loading ? <Loader2 className="spin" size={18} /> : <Zap size={18} />}
              Load
            </button>
            <button type="button" onClick={resetModel} disabled={loading || generating} title="Reload model">
              <RefreshCw size={18} />
              Reset
            </button>
            <button type="button" onClick={clearChat} title="Clear chat">
              <Trash2 size={18} />
              Clear
            </button>

            <label>
              <span>Temperature</span>
              <input
                type="range"
                min="0.1"
                max="1.5"
                step="0.05"
                value={settings.temperature}
                onChange={(event) => setSettings({ ...settings, temperature: Number(event.currentTarget.value) })}
              />
              <output>{settings.temperature.toFixed(2)}</output>
            </label>
            <label>
              <span>Top-p</span>
              <input
                type="range"
                min="0.1"
                max="1"
                step="0.05"
                value={settings.topP}
                onChange={(event) => setSettings({ ...settings, topP: Number(event.currentTarget.value) })}
              />
              <output>{settings.topP.toFixed(2)}</output>
            </label>
            <label>
              <span>Max tokens</span>
              <input
                type="number"
                min="1"
                max="512"
                value={settings.maxNewTokens}
                onChange={(event) => setSettings({ ...settings, maxNewTokens: Number(event.currentTarget.value) })}
              />
            </label>
            <label>
              <span>Top-k</span>
              <input
                type="number"
                min="1"
                max="500"
                value={settings.topK}
                onChange={(event) => setSettings({ ...settings, topK: Number(event.currentTarget.value) })}
              />
            </label>

            <div className="meter">
              <span>{progress?.file ?? defaults.revision}</span>
              <progress value={progress?.progress ?? 0} max={100} />
            </div>

            <nav className="project-links" aria-label="Project links">
              <a href="https://huggingface.co/scasella91/talkie-1930-13b-it-ONNX" target="_blank" rel="noreferrer">
                <ExternalLink size={14} />
                Model
              </a>
              <a href="https://github.com/scasella/talkie-quant-webgpu" target="_blank" rel="noreferrer">
                <Github size={14} />
                GitHub
              </a>
            </nav>
          </aside>

          <section className="chat" aria-label="Conversation">
            <div className="messages">
              {messages.map((message, index) => (
                <article className={`message ${message.role}`} key={`${message.role}-${index}`}>
                  <span>{message.role}</span>
                  <p>{message.content || (message.role === "assistant" && generating ? " " : "")}</p>
                </article>
              ))}
            </div>

            {error ? <div className="error">{error}</div> : null}

            <form className="composer" onSubmit={send}>
              <textarea
                value={draft}
                onChange={(event) => setDraft(event.currentTarget.value)}
                placeholder="Ask Talkie..."
                rows={3}
                disabled={generating}
              />
              {generating ? (
                <button type="button" className="stop" onClick={stopGeneration} title="Stop">
                  <Square size={18} />
                  Stop
                </button>
              ) : (
                <button type="submit" className="primary" disabled={!draft.trim()} title="Send">
                  <SendHorizontal size={18} />
                  Send
                </button>
              )}
            </form>
          </section>
        </div>
      </section>
    </main>
  );
}

export default App;
