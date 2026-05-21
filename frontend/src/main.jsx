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
  Gauge,
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
  WifiOff,
} from "lucide-react";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";
const MONITOR_SLIDES = [
  {
    image: "/banners/ai-fire-command-carousel.png",
    eyebrow: "AI 火情指挥",
    title: "实时识别烟雾与火焰",
    description: "接入摄像头后持续分析画面，发现火情风险时同步生成研判结果。",
  },
  {
    image: "/banners/yolo-industrial-detection.png",
    eyebrow: "YOLO 检测",
    title: "上传图片与视频复核",
    description: "支持本地文件检测，输出标注结果、目标数量、置信度和历史事件。",
  },
  {
    image: "/banners/smart-city-warning.png",
    eyebrow: "联动告警",
    title: "飞书通知与事件追踪",
    description: "火灾报警可同步到飞书，检测记录保存在本地 SQLite 便于复盘。",
  },
];

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
  const [streams, setStreams] = useState([]);
  const [selectedStreamId, setSelectedStreamId] = useState("");
  const [currentAlert, setCurrentAlert] = useState(null);
  const [realtimeFireEvents, setRealtimeFireEvents] = useState([]);
  const [historyItems, setHistoryItems] = useState([]);
  const [historyFilters, setHistoryFilters] = useState({ sourceType: "", alertLevel: "" });
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
      const viewable = items.find((item) => isActiveStream(item)) || items[0];
      const bestFireEvent = findBestFireEvent(items);
      setStreams(items);
      setSelectedStreamId((current) => {
        if (items.some((item) => item.id === current)) return current;
        return viewable?.id || "";
      });
      if (bestFireEvent) {
        setCurrentAlert(bestFireEvent);
        setSelectedEvent(bestFireEvent);
      } else if (viewable?.last_summary) {
        setCurrentAlert({
          ...viewable.last_summary,
          history_event: viewable.last_event,
          stream_status: viewable.status,
        });
      }
      if (bestFireEvent || viewable?.last_event) {
        loadHistory(historyFilters);
      }
    } catch {
      // The stream list is optional; upload-only use still works without it.
    }
  }

  async function loadHistory(filters = historyFilters, options = {}) {
    setHistoryLoading(true);
    try {
      const query = new URLSearchParams({ limit: filters.sourceType || filters.alertLevel ? "100" : "20" });
      if (filters.sourceType) query.set("source_type", filters.sourceType);
      if (filters.alertLevel) query.set("alert_level", filters.alertLevel);
      const payload = await fetchJson(`/api/history?${query.toString()}`);
      const items = applyHistoryFilters(payload.items || [], filters).slice(0, 20);
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

  async function loadRealtimeFireEvents() {
    try {
      const query = new URLSearchParams({
        limit: "100",
        source_type: "rtsp",
        alert_level: "danger",
      });
      const payload = await fetchJson(`/api/history?${query.toString()}`);
      const items = applyHistoryFilters(payload.items || [], {
        sourceType: "rtsp",
        alertLevel: "danger",
      });
      setRealtimeFireEvents(items);
    } catch {
      setRealtimeFireEvents([]);
    }
  }

  useEffect(() => {
    refreshHealth();
    loadStreams();
    loadRealtimeFireEvents();
    const timer = window.setInterval(refreshHealth, 10000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    loadHistory(historyFilters, { promoteLatest: true });
    const timer = window.setInterval(
      () => loadHistory(historyFilters, { promoteLatest: true }),
      5000,
    );
    return () => window.clearInterval(timer);
  }, [historyFilters]);

  useEffect(() => {
    const timer = window.setInterval(loadStreams, 1800);
    return () => window.clearInterval(timer);
  }, [historyFilters]);

  useEffect(() => {
    const timer = window.setInterval(loadRealtimeFireEvents, 5000);
    return () => window.clearInterval(timer);
  }, []);

  const focusAlert = currentAlert || selectedEvent;
  const activeStreams = health?.active_streams ?? streams.filter(isActiveStream).length;

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#live" aria-label="火灾智能检测平台首页">
          <span className="brand-mark">
            <Flame size={22} />
          </span>
          <span>
            <strong>火灾智能检测平台</strong>
            <small>烟雾与火焰 AI 监测指挥台</small>
          </span>
        </a>
        <nav className="nav-actions" aria-label="页面导航">
          <a href="#live">实时监控</a>
          <a href="#upload">上传检测</a>
          <a href="#history">事件追踪</a>
          <StatusPill health={health} />
        </nav>
      </header>

      <section className="dashboard-intro" aria-label="系统运行概览">
        <div>
          <span className="section-label">本地演示平台</span>
          <h1>消防监测指挥台</h1>
          <p>
            面向厂区、仓储和园区场景，集中接入 RTSP 实时画面、上传复核和飞书报警联动。
          </p>
        </div>
        <div className="status-grid">
          <Metric icon={<ShieldCheck />} label="模型权重" value={health?.model_path || "Smoke Fire.pt"} />
          <Metric icon={<RadioTower />} label="推理设备" value={health?.inference_device || health?.device || "检测中"} />
          <Metric
            icon={<Bell />}
            label="飞书报警"
            value={health?.feishu_configured ? "已配置" : "未配置"}
            tone={health?.feishu_configured ? "ok" : "warn"}
          />
          <Metric icon={<Camera />} label="实时任务" value={`${activeStreams} 路`} />
        </div>
      </section>

      <section className="command-grid">
        <LiveStreamPanel
          selectedStreamId={selectedStreamId}
          setSelectedStreamId={setSelectedStreamId}
          setStreams={setStreams}
          streams={streams}
          realtimeFireEvents={realtimeFireEvents}
          onAlert={setCurrentAlert}
          onHistoryRefresh={() => loadHistory(historyFilters)}
          onStreamsRefresh={loadStreams}
        />
        <AlertPanel data={focusAlert} health={health} onHealthRefresh={refreshHealth} />
      </section>

      <section className="workbench-grid">
        <UploadPanel
          onResult={(payload) => {
            setCurrentAlert(payload);
            setSelectedEvent(payload.history_event || null);
            loadHistory(historyFilters);
          }}
        />
        <HistoryPanel
          filters={historyFilters}
          items={historyItems}
          loading={historyLoading}
          selected={selectedEvent}
          setFilters={setHistoryFilters}
          onRefresh={() => loadHistory(historyFilters)}
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
      {ready ? <CheckCircle2 size={16} /> : offline ? <WifiOff size={16} /> : <AlertTriangle size={16} />}
      {ready ? "模型在线" : offline ? "后端离线" : "模型未加载"}
    </span>
  );
}

function Metric({ icon, label, value, tone = "neutral" }) {
  return (
    <div className={`metric ${tone}`}>
      <span className="metric-icon">{icon}</span>
      <span>{label}</span>
      <strong title={String(value)}>{value}</strong>
    </div>
  );
}

function CarouselPreview({ slide, slides, activeIndex, onSelect }) {
  return (
    <div className="carousel-preview">
      <img src={slide.image} alt={slide.title} />
      <div className="carousel-copy">
        <span>{slide.eyebrow}</span>
        <strong>{slide.title}</strong>
        <p>{slide.description}</p>
      </div>
      <div className="carousel-dots" aria-label="监控功能轮播">
        {slides.map((item, index) => (
          <button
            aria-label={`查看${item.title}`}
            className={index === activeIndex ? "active" : ""}
            key={item.title}
            onClick={() => onSelect(index)}
            type="button"
          />
        ))}
      </div>
    </div>
  );
}

const CAMERA_PRESETS = ["一号摄像头", "二号摄像头", "三号摄像头"];

function LiveStreamPanel({
  streams,
  selectedStreamId,
  setSelectedStreamId,
  setStreams,
  realtimeFireEvents,
  onAlert,
  onHistoryRefresh,
  onStreamsRefresh,
}) {
  const [slideIndex, setSlideIndex] = useState(0);
  const [forms, setForms] = useState(
    CAMERA_PRESETS.map((name) => ({
      name,
      rtsp_url: "",
      conf: 0.25,
      imgsz: 640,
      alert_cooldown_seconds: 300,
    })),
  );
  const [slotStreamIds, setSlotStreamIds] = useState(["", "", ""]);
  const [loadingSlot, setLoadingSlot] = useState(null);
  const [errors, setErrors] = useState({});

  useEffect(() => {
    if (streams.some(isActiveStream)) return undefined;
    const timer = window.setInterval(() => {
      setSlideIndex((current) => (current + 1) % MONITOR_SLIDES.length);
    }, 4200);
    return () => window.clearInterval(timer);
  }, [streams]);

  function updateForm(index, patch) {
    setForms((current) =>
      current.map((item, itemIndex) => (itemIndex === index ? { ...item, ...patch } : item)),
    );
  }

  function streamForSlot(index) {
    const id = slotStreamIds[index];
    const byId = streams.find((item) => item.id === id);
    if (byId) return byId;
    return streams.find((item) => item.name === forms[index].name && isActiveStream(item));
  }

  async function startStream(event, index) {
    event.preventDefault();
    setLoadingSlot(index);
    setErrors((current) => ({ ...current, [index]: "" }));
    try {
      const payload = await fetchJson("/api/streams", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(forms[index]),
      });
      setStreams((current) => [payload, ...current.filter((item) => item.id !== payload.id)]);
      setSlotStreamIds((current) =>
        current.map((item, itemIndex) => (itemIndex === index ? payload.id : item)),
      );
      setSelectedStreamId(payload.id);
      onAlert(payload.last_summary || null);
      onHistoryRefresh();
      onStreamsRefresh();
    } catch (err) {
      setErrors((current) => ({ ...current, [index]: err.message || "实时流启动失败" }));
    } finally {
      setLoadingSlot(null);
    }
  }

  async function stopStream(index) {
    const slotStream = streamForSlot(index);
    if (!slotStream?.id) return;
    setLoadingSlot(index);
    setErrors((current) => ({ ...current, [index]: "" }));
    try {
      const payload = await fetchJson(`/api/streams/${slotStream.id}/stop`, { method: "POST" });
      setStreams((current) => current.map((item) => (item.id === payload.id ? payload : item)));
      setSlotStreamIds((current) =>
        current.map((item, itemIndex) => (itemIndex === index ? "" : item)),
      );
      if (selectedStreamId === payload.id) {
        const next = streams.find((item) => item.id !== payload.id && isActiveStream(item));
        setSelectedStreamId(next?.id || "");
      }
      onStreamsRefresh();
    } catch (err) {
      setErrors((current) => ({ ...current, [index]: err.message || "停止实时流失败" }));
    } finally {
      setLoadingSlot(null);
    }
  }

  const selectedStream = streams.find((item) => item.id === selectedStreamId);
  const activeSlide = MONITOR_SLIDES[slideIndex];

  return (
    <section className="panel live-panel" id="live">
      <div className="panel-heading compact">
        <div>
          <span className="section-label">实时监控</span>
          <h2>网络摄像头火灾监测</h2>
        </div>
        <Camera size={26} />
      </div>

      <div className="live-layout">
        <div className="video-stage">
          {selectedStream?.id && isActiveStream(selectedStream) ? (
            <img src={apiUrl(selectedStream.mjpeg_url)} alt={`${selectedStream.name}实时检测画面`} />
          ) : (
            <CarouselPreview
              activeIndex={slideIndex}
              onSelect={setSlideIndex}
              slide={activeSlide}
              slides={MONITOR_SLIDES}
            />
          )}
          <div className="video-overlay">
            <span className={`stream-badge ${streamStatusClass(selectedStream?.status)}`}>
              {selectedStream ? streamStatusText(selectedStream.status) : "等待接入"}
            </span>
            <span className="stream-name">{selectedStream?.name || "请选择摄像头"}</span>
          </div>
          {streams.length > 0 && (
            <div className="camera-switcher">
              {forms.map((form, index) => {
                const slotStream = streamForSlot(index);
                const disabled = !slotStream?.id;
                return (
                  <button
                    className={selectedStreamId && selectedStreamId === slotStream?.id ? "active" : ""}
                    disabled={disabled}
                    key={form.name}
                    onClick={() => slotStream?.id && setSelectedStreamId(slotStream.id)}
                    type="button"
                  >
                    {form.name}
                  </button>
                );
              })}
            </div>
          )}
        </div>

        <div className="camera-config-list">
          <div className="control-card-title">
            <strong>多摄像头接入参数</strong>
            <span>最多同时接入 3 路，选择摄像头即可切换实时画面</span>
          </div>
          {forms.map((form, index) => {
            const slotStream = streamForSlot(index);
            const running = slotStream && isActiveStream(slotStream);
            const loading = loadingSlot === index;
            return (
              <form className="camera-config-card" key={form.name} onSubmit={(event) => startStream(event, index)}>
                <div className="camera-config-heading">
                  <strong>{form.name}</strong>
                  <span className={`stream-badge ${streamStatusClass(slotStream?.status)}`}>
                    {slotStream ? streamStatusText(slotStream.status) : "未接入"}
                  </span>
                </div>
                <label>
                  摄像头名称
                  <input
                    value={form.name}
                    onChange={(event) => updateForm(index, { name: event.target.value })}
                  />
                </label>
                <label>
                  RTSP 地址
                  <input
                    placeholder="rtsp://user:password@192.168.1.10:554/stream"
                    value={form.rtsp_url}
                    onChange={(event) => updateForm(index, { rtsp_url: event.target.value })}
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
                      onChange={(event) => updateForm(index, { conf: Number(event.target.value) })}
                    />
                    <b>{form.conf.toFixed(2)}</b>
                  </label>
                  <label>
                    尺寸
                    <select
                      value={form.imgsz}
                      onChange={(event) => updateForm(index, { imgsz: Number(event.target.value) })}
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
                      updateForm(index, { alert_cooldown_seconds: Number(event.target.value) })
                    }
                  />
                </label>

                <div className="button-row">
                  <button className="primary-button" disabled={loading || running} type="submit">
                    {loading && !running ? <Loader2 className="spin" size={18} /> : <PlayCircle size={18} />}
                    启动检测
                  </button>
                  <button
                    className="ghost-button"
                    disabled={loading || !slotStream?.id}
                    type="button"
                    onClick={() => stopStream(index)}
                  >
                    <Square size={16} />
                    停止
                  </button>
                </div>
                {errors[index] && <div className="message error">{errors[index]}</div>}
                {slotStream?.last_error && <div className="message warn">{slotStream.last_error}</div>}
              </form>
            );
          })}
        </div>
      </div>
      <RealtimeFireResultsPanel events={realtimeFireEvents} />
    </section>
  );
}

function RealtimeFireResultsPanel({ events }) {
  if (events.length === 0) {
    return (
      <div className="fire-results-panel empty">
        <div>
          <span className="section-label">火灾实时监控结果</span>
          <strong>等待火灾标注结果</strong>
        </div>
        <p>三路摄像头检测到火灾后，所有标注截图或视频会在这里持续显示，并由后端自动同步飞书警报。</p>
      </div>
    );
  }

  return (
    <div className="fire-results-panel danger">
      <div className="fire-results-heading">
        <div>
          <span className="section-label">火灾实时监控结果</span>
          <strong>已记录 {events.length} 条火灾标注结果</strong>
        </div>
        <span>自动同步飞书警报</span>
      </div>
      <div className="fire-result-grid">
        {events.map((event) => (
          <article className="fire-result-card" key={event.id}>
            <FireResultMedia event={event} />
            <div className="fire-result-copy">
              <strong>{event.source_name || "摄像头火灾事件"}</strong>
              <p>{event.situation || event.message}</p>
              <div className="chips">
                <span>最高置信度 {formatPercent(event.summary?.max_confidence)}</span>
                <span>{formatTime(event.created_at)}</span>
                <span>飞书 {feishuStatusText(event.feishu_status)}</span>
              </div>
            </div>
          </article>
        ))}
      </div>
    </div>
  );
}

function FireResultMedia({ event }) {
  if (!event.media_url) {
    return <div className="snapshot-placeholder">暂无标注结果</div>;
  }
  const src = apiUrl(event.media_url);
  if (isVideoUrl(event.media_url)) {
    return <video controls src={src} />;
  }
  return <img src={src} alt={`${event.source_name || "摄像头"}火灾标注结果`} />;
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
    <section className={`panel alert-panel ${view.alert_level}`} aria-live="polite">
      <div className="panel-heading compact">
        <div>
          <span className="section-label">火情研判</span>
          <h2>{view.decision}</h2>
        </div>
        {view.alert_level === "danger" ? <AlertTriangle size={27} /> : <ShieldCheck size={27} />}
      </div>

      <div className={`alert-level-card ${view.alert_level}`}>
        <span>{alertLevelText(view.alert_level)}</span>
        <strong>{view.message}</strong>
      </div>

      <p className="situation">{view.situation}</p>

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

      <div className="segmented" aria-label="上传检测类型">
        <button className={mode === "video" ? "active" : ""} onClick={() => setMode("video")} type="button">
          <Video size={16} />
          视频
        </button>
        <button className={mode === "image" ? "active" : ""} onClick={() => setMode("image")} type="button">
          <ImageUp size={16} />
          图片
        </button>
      </div>

      <form className="control-card upload-controls" onSubmit={submitDetection}>
        <label className="dropzone">
          <input
            accept={mode === "image" ? "image/*" : "video/*"}
            type="file"
            onChange={(event) => setFile(event.target.files?.[0] || null)}
          />
          <Upload size={20} />
          <span title={file?.name || ""}>{file ? file.name : "选择检测文件"}</span>
          <small>{mode === "image" ? "支持 JPG、PNG、WEBP 等图片" : "支持 MP4、AVI、MOV、MKV 等视频"}</small>
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
        <button className="primary-button full-width" disabled={loading} type="submit">
          {loading ? <Loader2 className="spin" size={18} /> : <PlayCircle size={18} />}
          {loading ? "检测中" : "开始检测"}
        </button>
      </form>
      {error && <div className="message error">{error}</div>}

      <div className={`result-area ${previewUrl || resultUrl ? "has-media" : ""}`}>
        {previewUrl || resultUrl ? (
          <div className="preview-grid">
            {previewUrl && <MediaPreview title="原始文件" type={mode} src={previewUrl} />}
            {resultUrl && <MediaPreview title="检测结果" type={mode} src={apiUrl(resultUrl)} />}
          </div>
        ) : (
          <div className="empty-result">
            <Gauge size={22} />
            <span>完成检测后将在这里显示原始文件与标注结果。</span>
          </div>
        )}
      </div>
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

function HistoryPanel({ items, selected, filters, setFilters, loading, onRefresh, onSelect }) {
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
        <select
          aria-label="按来源筛选历史记录"
          value={filters.sourceType}
          onChange={(event) => setFilters((current) => ({ ...current, sourceType: event.target.value }))}
        >
          <option value="">全部来源</option>
          <option value="image">图片</option>
          <option value="video">视频</option>
          <option value="rtsp">摄像头</option>
        </select>
        <select
          aria-label="按状态筛选历史记录"
          value={filters.alertLevel}
          onChange={(event) => setFilters((current) => ({ ...current, alertLevel: event.target.value }))}
        >
          <option value="">全部状态</option>
          <option value="danger">火灾报警</option>
          <option value="safe">安全</option>
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
              <span className={`event-tag ${item.alert_level}`}>{alertLevelText(item.alert_level)}</span>
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
          <h2>本地工作流</h2>
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

function isActiveStream(stream) {
  return Boolean(stream && !["stopped", "error", "offline"].includes(stream.status));
}

function isVideoUrl(url) {
  return /\.(mp4|avi|mov|mkv|webm)$/i.test(url || "");
}

function findBestFireEvent(streams) {
  return streams
    .map((stream) => stream.last_event)
    .filter((event) => event?.alert_level === "danger")
    .sort((left, right) => {
      const leftConfidence = Number(left.summary?.max_confidence) || 0;
      const rightConfidence = Number(right.summary?.max_confidence) || 0;
      return rightConfidence - leftConfidence;
    })[0] || null;
}

function applyHistoryFilters(items, filters) {
  return items.filter((item) => {
    const sourceMatched = !filters.sourceType || item.source_type === filters.sourceType;
    const alertMatched = !filters.alertLevel || item.alert_level === filters.alertLevel;
    return sourceMatched && alertMatched;
  });
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

function alertLevelText(level) {
  return (
    {
      danger: "火灾报警",
      warning: "烟雾预警",
      safe: "安全",
    }[level] || "未分级"
  );
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
