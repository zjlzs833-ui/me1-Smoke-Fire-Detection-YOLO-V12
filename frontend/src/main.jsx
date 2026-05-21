import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  AlertTriangle,
  Bell,
  Camera,
  CheckCircle2,
  Clock,
  Database,
  Flame,
  History,
  ImageUp,
  Loader2,
  PlayCircle,
  RadioTower,
  RefreshCw,
  Send,
  ShieldCheck,
  Square,
  Upload,
  Video,
} from "lucide-react";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

function apiUrl(path) {
  if (!path) return "";
  if (path.startsWith("http")) return path;
  return `${API_BASE}${path}`;
}

async function fetchJson(path, options) {
  let response;
  try {
    response = await fetch(apiUrl(path), options);
  } catch (error) {
    throw new Error(friendlyFetchError(error));
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || payload.error || "请求失败");
  }
  return payload;
}

function friendlyFetchError(error) {
  if (error instanceof TypeError) {
    return "无法连接后端服务，请确认 FastAPI 已在 127.0.0.1:8000 启动。";
  }
  return error?.message || "网络请求失败";
}

function App() {
  const [health, setHealth] = useState(null);
  const [stream, setStream] = useState(null);
  const [currentAlert, setCurrentAlert] = useState(null);
  const [historyItems, setHistoryItems] = useState([]);
  const [historyFilter, setHistoryFilter] = useState("");
  const [selectedEvent, setSelectedEvent] = useState(null);
  const [historyLoading, setHistoryLoading] = useState(false);

  async function refreshHealth() {
    try {
      const payload = await fetchJson("/api/health");
      setHealth(payload);
    } catch {
      setHealth({ status: "offline", model_loaded: false, device: "unknown" });
    }
  }

  async function loadStreams() {
    try {
      const payload = await fetchJson("/api/streams");
      const items = payload.items || [];
      const active =
        items.find((item) => !["stopped", "error", "offline"].includes(item.status)) || items[0];
      if (!active) return;
      setStream((current) => current || active);
      if (active.last_summary) {
        setCurrentAlert({
          ...active.last_summary,
          history_event: active.last_event,
          stream_status: active.status,
        });
      }
      if (active.last_event) {
        setSelectedEvent((current) => current || active.last_event);
      }
    } catch {
      // The stream list is optional; upload-only use still works without it.
    }
  }

  async function loadHistory(filter = historyFilter, options = {}) {
    setHistoryLoading(true);
    try {
      const query = new URLSearchParams({ limit: "20" });
      if (filter) query.set("alert_level", filter);
      const payload = await fetchJson(`/api/history?${query.toString()}`);
      const items = payload.items || [];
      setHistoryItems(items);
      if (items.length) {
        setSelectedEvent((current) => current || items[0]);
        if (options.promoteLatest) {
          setCurrentAlert((current) => {
            if (!current) return items[0];
            if (items[0].alert_level === "danger" && current.id !== items[0].id) {
              return items[0];
            }
            return current;
          });
        }
      }
    } catch {
      setHistoryItems([]);
    } finally {
      setHistoryLoading(false);
    }
  }

  useEffect(() => {
    refreshHealth();
    loadStreams();
    const timer = window.setInterval(refreshHealth, 10000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    loadHistory(historyFilter, { promoteLatest: true });
    const timer = window.setInterval(
      () => loadHistory(historyFilter, { promoteLatest: true }),
      5000,
    );
    return () => window.clearInterval(timer);
  }, [historyFilter]);

  useEffect(() => {
    if (!stream?.id) return undefined;
    const timer = window.setInterval(async () => {
      try {
        const payload = await fetchJson(`/api/streams/${stream.id}`);
        setStream(payload);
        if (payload.last_summary) {
          setCurrentAlert({
            ...payload.last_summary,
            history_event: payload.last_event,
            stream_status: payload.status,
          });
        }
        if (payload.last_event) {
          setSelectedEvent(payload.last_event);
          loadHistory(historyFilter);
        }
      } catch {
        setStream((current) =>
          current ? { ...current, status: "offline", last_error: "实时流状态获取失败" } : current,
        );
      }
    }, 1800);
    return () => window.clearInterval(timer);
  }, [stream?.id, historyFilter]);

  const focusAlert = currentAlert || selectedEvent;

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <Flame size={24} />
          <div>
            <strong>火灾智能检测平台</strong>
            <span>RTSP 实时检测 / 飞书报警 / 历史追踪</span>
          </div>
        </div>
        <nav className="nav-actions">
          <a href="#live">实时检测</a>
          <a href="#upload">上传检测</a>
          <a href="#history">历史记录</a>
          <StatusPill health={health} />
        </nav>
      </header>

      <section className="status-grid">
        <Metric icon={<ShieldCheck />} label="模型权重" value={health?.model_path || "Smoke Fire.pt"} />
        <Metric icon={<RadioTower />} label="运行设备" value={health?.device || "检测中"} />
        <Metric
          icon={<Bell />}
          label="飞书报警"
          value={health?.feishu_configured ? "已配置" : "未配置"}
          tone={health?.feishu_configured ? "ok" : "warn"}
        />
        <Metric icon={<Database />} label="历史记录" value="SQLite" />
      </section>

      <section className="command-grid">
        <LiveStreamPanel
          stream={stream}
          setStream={setStream}
          onAlert={setCurrentAlert}
          onHistoryRefresh={() => loadHistory(historyFilter)}
        />
        <AlertPanel data={focusAlert} health={health} onHealthRefresh={refreshHealth} />
      </section>

      <section className="lower-grid">
        <UploadPanel
          onResult={(payload) => {
            setCurrentAlert(payload);
            setSelectedEvent(payload.history_event || null);
            loadHistory(historyFilter);
          }}
        />
        <HistoryPanel
          filter={historyFilter}
          items={historyItems}
          loading={historyLoading}
          selected={selectedEvent}
          setFilter={setHistoryFilter}
          onRefresh={() => loadHistory(historyFilter)}
          onSelect={(item) => {
            setSelectedEvent(item);
            setCurrentAlert(item);
          }}
        />
        <SystemPanel health={health} />
      </section>
    </main>
  );
}

function StatusPill({ health }) {
  const ready = health?.model_loaded;
  const offline = health?.status === "offline";
  return (
    <span className={`status-pill ${ready ? "ready" : "warn"}`}>
      {ready ? <CheckCircle2 size={16} /> : <AlertTriangle size={16} />}
      {ready ? "模型已加载" : offline ? "后端离线" : "模型未加载"}
    </span>
  );
}

function Metric({ icon, label, value, tone = "neutral" }) {
  return (
    <div className={`metric ${tone}`}>
      <span className="metric-icon">{icon}</span>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function LiveStreamPanel({ stream, setStream, onAlert, onHistoryRefresh }) {
  const [form, setForm] = useState({
    name: "厂区一号摄像头",
    rtsp_url: "",
    conf: 0.25,
    imgsz: 640,
    alert_cooldown_seconds: 300,
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function startStream(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      const payload = await fetchJson("/api/streams", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      setStream(payload);
      onAlert(payload.last_summary || null);
      onHistoryRefresh();
    } catch (err) {
      setError(err.message || "实时流启动失败");
    } finally {
      setLoading(false);
    }
  }

  async function stopStream() {
    if (!stream?.id) return;
    setLoading(true);
    setError("");
    try {
      const payload = await fetchJson(`/api/streams/${stream.id}/stop`, { method: "POST" });
      setStream(payload);
    } catch (err) {
      setError(err.message || "停止实时流失败");
    } finally {
      setLoading(false);
    }
  }

  const running = stream && !["stopped", "error", "offline"].includes(stream.status);

  return (
    <section className="panel live-panel" id="live">
      <div className="panel-heading">
        <div>
          <span className="section-label">RTSP 实时检测</span>
          <h1>网络摄像头火灾监测</h1>
        </div>
        <Camera size={27} />
      </div>

      <div className="live-layout">
        <div className="video-stage">
          {stream?.id ? (
            <img src={apiUrl(stream.mjpeg_url)} alt="实时检测画面" />
          ) : (
            <img src="/banners/smart-command-center.png" alt="监控中心预览" />
          )}
          <span className={`stream-badge ${streamStatusClass(stream?.status)}`}>
            {stream ? streamStatusText(stream.status) : "等待接入"}
          </span>
        </div>

        <form className="control-stack" onSubmit={startStream}>
          <label>
            摄像头名称
            <input
              value={form.name}
              onChange={(event) => setForm({ ...form, name: event.target.value })}
            />
          </label>
          <label>
            RTSP 地址
            <input
              placeholder="rtsp://user:password@192.168.1.10:554/stream"
              value={form.rtsp_url}
              onChange={(event) => setForm({ ...form, rtsp_url: event.target.value })}
            />
          </label>
          <div className="split-controls">
            <label>
              置信度
              <input
                max="0.9"
                min="0.05"
                step="0.05"
                type="range"
                value={form.conf}
                onChange={(event) => setForm({ ...form, conf: Number(event.target.value) })}
              />
              <b>{form.conf.toFixed(2)}</b>
            </label>
            <label>
              尺寸
              <select
                value={form.imgsz}
                onChange={(event) => setForm({ ...form, imgsz: Number(event.target.value) })}
              >
                <option value="512">512</option>
                <option value="640">640</option>
                <option value="960">960</option>
                <option value="1280">1280</option>
              </select>
            </label>
          </div>
          <label>
            告警冷却秒数
            <input
              min="30"
              step="30"
              type="number"
              value={form.alert_cooldown_seconds}
              onChange={(event) =>
                setForm({ ...form, alert_cooldown_seconds: Number(event.target.value) })
              }
            />
          </label>

          <div className="button-row">
            <button className="primary-button" disabled={loading || running} type="submit">
              {loading && !running ? <Loader2 className="spin" size={18} /> : <PlayCircle size={18} />}
              启动检测
            </button>
            <button className="ghost-button" disabled={loading || !stream?.id} type="button" onClick={stopStream}>
              <Square size={16} />
              停止
            </button>
          </div>
          {error && <div className="message error">{error}</div>}
          {stream?.last_error && <div className="message warn">{stream.last_error}</div>}
        </form>
      </div>
    </section>
  );
}

function AlertPanel({ data, health, onHealthRefresh }) {
  const view = normalizeAlert(data);
  const eventId = getEventId(data);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState(null);
  const [syncing, setSyncing] = useState(false);
  const [syncResult, setSyncResult] = useState(null);

  async function testFeishu() {
    setTesting(true);
    setTestResult(null);
    try {
      const payload = await fetchJson("/api/feishu/test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: "火灾检测系统飞书报警通道测试。" }),
      });
      setTestResult(payload);
      onHealthRefresh();
    } catch (err) {
      setTestResult({ status: "failed", error: err.message });
    } finally {
      setTesting(false);
    }
  }

  async function syncAssessment() {
    if (!eventId) {
      setSyncResult({ status: "failed", error: "当前火情研判还没有历史记录，完成一次检测后再同步。" });
      return;
    }
    setSyncing(true);
    setSyncResult(null);
    try {
      const payload = await fetchJson(`/api/history/${eventId}/feishu`, { method: "POST" });
      setSyncResult(payload);
      onHealthRefresh();
    } catch (err) {
      setSyncResult({ status: "failed", error: err.message });
    } finally {
      setSyncing(false);
    }
  }

  return (
    <section className={`panel alert-panel ${view.alert_level}`}>
      <div className="panel-heading">
        <div>
          <span className="section-label">火情研判</span>
          <h2>{view.decision}</h2>
        </div>
        {view.alert_level === "danger" ? <AlertTriangle size={27} /> : <ShieldCheck size={27} />}
      </div>
      <p className="situation">{view.situation}</p>
      <p className="message-text">{view.message}</p>
      <div className="chips">
        <span>火焰 {view.fire_count}</span>
        <span>烟雾 {view.smoke_count}</span>
        <span>最高置信度 {formatPercent(view.max_confidence)}</span>
        <span>飞书 {feishuStatusText(view.feishu_status)}</span>
      </div>

      <div className="actions-box">
        <strong>应对措施</strong>
        <ol>
          {view.actions.map((action) => (
            <li key={action}>{action}</li>
          ))}
        </ol>
      </div>

      <div className="feishu-row">
        <div>
          <span>飞书机器人</span>
          <strong>{health?.feishu_configured ? "已配置" : "未配置 FEISHU_WEBHOOK_URL"}</strong>
        </div>
        <div className="feishu-buttons">
          <button className="primary-button" disabled={syncing || !eventId} onClick={syncAssessment} type="button">
            {syncing ? <Loader2 className="spin" size={16} /> : <Send size={16} />}
            同步研判
          </button>
          <button className="ghost-button" disabled={testing} onClick={testFeishu} type="button">
            {testing ? <Loader2 className="spin" size={16} /> : <Send size={16} />}
            测试
          </button>
        </div>
      </div>
      {syncResult && (
        <div className={`message ${syncResult.status === "sent" ? "ok" : "warn"}`}>
          {syncResult.status === "sent"
            ? "火情研判已同步到飞书。"
            : syncResult.error || "火情研判同步失败。"}
        </div>
      )}
      {testResult && (
        <div className={`message ${testResult.status === "sent" ? "ok" : "warn"}`}>
          {testResult.status === "sent"
            ? "飞书测试消息已发送。"
            : testResult.error || "飞书测试未发送。"}
        </div>
      )}
    </section>
  );
}

function UploadPanel({ onResult }) {
  const [mode, setMode] = useState("video");
  const [file, setFile] = useState(null);
  const [conf, setConf] = useState(0.25);
  const [imgsz, setImgsz] = useState(640);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const previewUrl = useMemo(() => (file ? URL.createObjectURL(file) : ""), [file]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  async function submitDetection(event) {
    event.preventDefault();
    if (!file) {
      setError("请先选择检测文件。");
      return;
    }

    const form = new FormData();
    form.append("file", file);
    form.append("conf", String(conf));
    form.append("imgsz", String(imgsz));

    setLoading(true);
    setError("");
    setResult(null);
    try {
      const endpoint = mode === "image" ? "/api/detect/image" : "/api/detect/video";
      let response;
      try {
        response = await fetch(apiUrl(endpoint), { method: "POST", body: form });
      } catch (err) {
        throw new Error(friendlyFetchError(err));
      }
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "检测失败");
      }
      setResult(payload);
      onResult(payload);
    } catch (err) {
      setError(err.message || "检测失败");
    } finally {
      setLoading(false);
    }
  }

  const resultUrl = result
    ? mode === "image"
      ? result.result_image_url
      : result.result_video_url
    : "";

  return (
    <section className="panel upload-panel" id="upload">
      <div className="panel-heading">
        <div>
          <span className="section-label">上传检测</span>
          <h2>视频文件与图片复核</h2>
        </div>
        <Upload size={25} />
      </div>

      <div className="segmented">
        <button className={mode === "video" ? "active" : ""} onClick={() => setMode("video")} type="button">
          <Video size={16} />
          视频
        </button>
        <button className={mode === "image" ? "active" : ""} onClick={() => setMode("image")} type="button">
          <ImageUp size={16} />
          图片
        </button>
      </div>

      <form className="control-stack" onSubmit={submitDetection}>
        <label className="dropzone">
          <input
            accept={mode === "image" ? "image/*" : "video/*"}
            type="file"
            onChange={(event) => setFile(event.target.files?.[0] || null)}
          />
          <span>{file ? file.name : "选择检测文件"}</span>
        </label>
        <div className="split-controls">
          <label>
            置信度
            <input
              max="0.9"
              min="0.05"
              step="0.05"
              type="range"
              value={conf}
              onChange={(event) => setConf(Number(event.target.value))}
            />
            <b>{conf.toFixed(2)}</b>
          </label>
          <label>
            尺寸
            <select value={imgsz} onChange={(event) => setImgsz(Number(event.target.value))}>
              <option value="512">512</option>
              <option value="640">640</option>
              <option value="960">960</option>
              <option value="1280">1280</option>
            </select>
          </label>
        </div>
        <button className="primary-button" disabled={loading} type="submit">
          {loading ? <Loader2 className="spin" size={18} /> : <PlayCircle size={18} />}
          {loading ? "检测中" : "开始检测"}
        </button>
      </form>
      {error && <div className="message error">{error}</div>}

      {(previewUrl || resultUrl) && (
        <div className="preview-grid">
          {previewUrl && <MediaPreview title="原始文件" type={mode} src={previewUrl} />}
          {resultUrl && <MediaPreview title="检测结果" type={mode} src={apiUrl(resultUrl)} />}
        </div>
      )}
      {result && <CompactSummary data={result} />}
    </section>
  );
}

function MediaPreview({ title, type, src }) {
  return (
    <div className="preview">
      <strong>{title}</strong>
      {type === "image" ? <img src={src} alt={title} /> : <video controls src={src} />}
    </div>
  );
}

function CompactSummary({ data }) {
  const view = normalizeAlert(data);
  return (
    <div className={`compact-summary ${view.alert_level}`}>
      <strong>{view.decision}</strong>
      <p>{view.message}</p>
      <div className="chips">
        <span>火焰 {view.fire_count}</span>
        <span>烟雾 {view.smoke_count}</span>
        <span>{formatPercent(view.max_confidence)}</span>
      </div>
    </div>
  );
}

function HistoryPanel({ items, selected, filter, setFilter, loading, onRefresh, onSelect }) {
  return (
    <section className="panel history-panel" id="history">
      <div className="panel-heading">
        <div>
          <span className="section-label">火灾历史记录</span>
          <h2>事件追踪</h2>
        </div>
        <History size={25} />
      </div>
      <div className="history-tools">
        <select value={filter} onChange={(event) => setFilter(event.target.value)}>
          <option value="">全部等级</option>
          <option value="danger">火灾报警</option>
          <option value="warning">烟雾预警</option>
          <option value="safe">安全记录</option>
        </select>
        <button className="icon-button" disabled={loading} onClick={onRefresh} type="button" title="刷新">
          <RefreshCw className={loading ? "spin" : ""} size={17} />
        </button>
      </div>

      <div className="history-list">
        {items.length === 0 ? (
          <p>暂无历史记录。</p>
        ) : (
          items.map((item) => (
            <button
              className={selected?.id === item.id ? "active" : ""}
              key={item.id}
              onClick={() => onSelect(item)}
              type="button"
            >
              <span className={`dot ${item.alert_level}`} />
              <strong>{item.decision}</strong>
              <small>{item.source_name}</small>
              <time>{formatTime(item.created_at)}</time>
            </button>
          ))
        )}
      </div>
    </section>
  );
}

function SystemPanel({ health }) {
  return (
    <section className="panel system-panel">
      <div className="panel-heading">
        <div>
          <span className="section-label">运行环境</span>
          <h2>Python 3.11 工作流</h2>
        </div>
        <Activity size={25} />
      </div>
      <div className="system-list">
        <InfoRow label="后端启动" value="conda run -n yolov12_fire uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000" />
        <InfoRow label="前端启动" value="cd frontend && npm.cmd run dev" />
        <InfoRow label="Python" value={health?.python_version || "3.11.x"} />
        <InfoRow label="CUDA" value={health?.cuda_available ? health.gpu_name || "可用" : "未启用"} />
        <InfoRow label="历史库" value={health?.history_db || "outputs/web/data/fire_events.sqlite3"} />
      </div>
      <div className="notice">
        <Clock size={16} />
        RTSP 地址和飞书 Webhook 通过运行时配置接入，检测输出保存在 outputs/web。
      </div>
    </section>
  );
}

function InfoRow({ label, value }) {
  return (
    <div className="info-row">
      <span>{label}</span>
      <code>{value}</code>
    </div>
  );
}

function normalizeAlert(data) {
  const fallback = {
    alert_level: "safe",
    decision: "等待检测",
    situation: "接入 RTSP 摄像头或上传视频后，系统会在这里显示火灾情况说明。",
    message: "当前没有新的火情判断。",
    actions: ["保持监控画面在线。", "完成飞书机器人配置后再进行报警联动测试。"],
    fire_count: 0,
    smoke_count: 0,
    max_confidence: 0,
    feishu_status: "not_sent",
  };
  if (!data) return fallback;
  const summary = data.summary || {};
  return {
    alert_level: data.alert_level || fallback.alert_level,
    decision: data.decision || fallback.decision,
    situation: summary.situation || data.situation || fallback.situation,
    message: summary.message || data.message || fallback.message,
    actions: summary.actions || data.actions || fallback.actions,
    fire_count: summary.fire_count ?? data.fire_count ?? 0,
    smoke_count: summary.smoke_count ?? data.smoke_count ?? 0,
    max_confidence: summary.max_confidence ?? data.max_confidence ?? 0,
    feishu_status:
      data.history_event?.feishu_status ||
      data.last_event?.feishu_status ||
      data.feishu_status ||
      fallback.feishu_status,
  };
}

function getEventId(data) {
  return data?.history_event?.id || data?.last_event?.id || data?.id || "";
}

function streamStatusText(status) {
  return (
    {
      created: "已创建",
      connecting: "连接中",
      running: "检测中",
      reconnecting: "重连中",
      stopping: "停止中",
      stopped: "已停止",
      error: "异常",
      offline: "离线",
    }[status] || "未知"
  );
}

function streamStatusClass(status) {
  if (status === "running") return "running";
  if (status === "error" || status === "offline") return "danger";
  if (status === "reconnecting" || status === "connecting") return "warning";
  return "idle";
}

function feishuStatusText(status) {
  return (
    {
      sent: "已发送",
      pending: "待发送",
      failed: "失败",
      skipped: "未触发",
      disabled: "未配置",
      not_sent: "未发送",
    }[status] || "未发送"
  );
}

function formatPercent(value) {
  return `${((Number(value) || 0) * 100).toFixed(1)}%`;
}

function formatTime(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("zh-CN", { hour12: false });
}

createRoot(document.getElementById("root")).render(<App />);
