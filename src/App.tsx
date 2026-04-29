import { type FormEvent, type MutableRefObject, useMemo, useRef, useState } from "react";
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

type LoadTarget = "q4f16" | "q8";

interface FileLoadState {
  loaded: number;
  total: number;
}

interface LoadMeter {
  label: string;
  detail: string;
  percent: number | null;
}

const DEFAULT_LOAD_METER: LoadMeter = {
  label: "Not loaded",
  detail: "main",
  percent: null
};

const TARGET_TOTAL_BYTES: Record<LoadTarget, number> = {
  q4f16: 10_583_998_959,
  q8: 15_312_926_922
};

function App() {
  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages);
  const [draft, setDraft] = useState("");
  const [settings, setSettings] = useState(defaultSettings);
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [status, setStatus] = useState("Idle");
  const [loadMeter, setLoadMeter] = useState<LoadMeter>(DEFAULT_LOAD_METER);
  const [dtype, setDtype] = useState("unloaded");
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const loadFilesRef = useRef<Map<string, FileLoadState>>(new Map());
  const loadTargetRef = useRef<LoadTarget | null>(null);
  const bestLoadPercentRef = useRef(0);

  const defaults = useMemo(() => modelDefaults(), []);
  const gpuReady = hasWebGPU();

  const handleProgress = (event: LoadProgress) => {
    setLoadMeter(updateLoadMeter(event, loadFilesRef.current, loadTargetRef, bestLoadPercentRef));
    if (event.status) setStatus(event.status);
  };

  const loadModel = async () => {
    setError(null);
    setLoading(true);
    setStatus("Loading");
    loadFilesRef.current.clear();
    loadTargetRef.current = null;
    bestLoadPercentRef.current = 0;
    setLoadMeter({
      label: "Preparing model",
      detail: "Fetching tokenizer and config",
      percent: 0
    });
    try {
      const runtime = await loadTalkieRuntime(handleProgress);
      setDtype(runtime.session.dtype);
      setStatus("Ready");
      setLoadMeter({
        label: "Ready",
        detail: `Loaded ${runtime.session.dtype}`,
        percent: 100
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setStatus("Error");
      setLoadMeter({
        label: "Load failed",
        detail: "See error message",
        percent: null
      });
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
      loadFilesRef.current.clear();
      loadTargetRef.current = null;
      bestLoadPercentRef.current = 0;
      setLoadMeter(DEFAULT_LOAD_METER);
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
              <div className="meter-header">
                <span>{loadMeter.label}</span>
                <strong>{loadMeter.percent == null ? "--" : `${loadMeter.percent}%`}</strong>
              </div>
              <span>{loadMeter.detail || defaults.revision}</span>
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

function updateLoadMeter(
  event: LoadProgress,
  files: Map<string, FileLoadState>,
  targetRef: MutableRefObject<LoadTarget | null>,
  bestPercentRef: MutableRefObject<number>
): LoadMeter {
  const file = event.file ?? "";
  const target = detectLoadTarget(file);
  if (target) targetRef.current = target;

  const loaded = finiteNumber(event.loaded);
  const total = finiteNumber(event.total);
  if (file && loaded != null && total != null && total > 0) {
    files.set(file, {
      loaded: Math.min(Math.max(loaded, 0), total),
      total
    });
  }

  const activeTarget = targetRef.current;
  const rawPercent = activeTarget
    ? aggregateTargetPercent(files, activeTarget)
    : metadataPercent(event.progress);
  const nextPercent =
    rawPercent == null
      ? bestPercentRef.current
      : Math.max(bestPercentRef.current, Math.min(99, Math.floor(rawPercent)));
  bestPercentRef.current = nextPercent;

  return {
    label: activeTarget ? `Loading ${activeTarget}` : "Preparing model",
    detail: readableProgressDetail(file, event.status),
    percent: nextPercent
  };
}

function aggregateTargetPercent(files: Map<string, FileLoadState>, target: LoadTarget): number {
  const needle = target === "q8" ? "model_quantized.onnx" : "model_q4f16.onnx";
  let loaded = 0;
  for (const [file, state] of files) {
    if (file.includes(needle)) loaded += state.loaded;
  }
  return (loaded / TARGET_TOTAL_BYTES[target]) * 100;
}

function detectLoadTarget(file: string): LoadTarget | null {
  if (file.includes("model_quantized.onnx")) return "q8";
  if (file.includes("model_q4f16.onnx")) return "q4f16";
  return null;
}

function metadataPercent(progress: number | undefined): number | null {
  const percent = finiteNumber(progress);
  if (percent == null) return null;
  return Math.min(3, Math.max(0, percent / 40));
}

function readableProgressDetail(file: string, status: string | undefined): string {
  if (!file) return status ? sentenceCase(status) : "Fetching model metadata";
  const clean = file.split(/[?#]/)[0].split("/").filter(Boolean).at(-1);
  return clean ?? file;
}

function finiteNumber(value: unknown): number | null {
  const number = typeof value === "number" ? value : Number(value);
  return Number.isFinite(number) ? number : null;
}

function sentenceCase(value: string): string {
  if (!value) return value;
  return value.charAt(0).toUpperCase() + value.slice(1);
}

export default App;
