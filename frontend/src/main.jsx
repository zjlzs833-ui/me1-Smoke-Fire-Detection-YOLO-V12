import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  AlertTriangle,
  Camera,
  CheckCircle2,
  Flame,
  ImageUp,
  Loader2,
  PlayCircle,
  RadioTower,
  ShieldCheck,
  Video,
} from "lucide-react";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

const slides = [
  {
    image: "/banners/smart-command-center.png",
    kicker: "智慧消防监控",
    title: "烟雾与火焰智能识别",
    text: "接入训练好的 YOLO 模型，对图片与视频素材进行快速检测。",
  },
  {
    image: "/banners/yolo-industrial-detection.png",
    kicker: "YOLO 实时识别",
    title: "面向监控场景的目标检测",
    text: "输出检测框、类别、置信度和告警等级，便于演示与二次开发。",
  },
  {
    image: "/banners/smart-city-warning.png",
    kicker: "风险预警联动",
    title: "从发现异常到辅助研判",
    text: "统一展示检测结果，让烟火风险更早被看见。",
  },
];

function mediaUrl(path) {
  if (!path) return "";
  if (path.startsWith("http")) return path;
  return `${API_BASE}${path}`;
}

function App() {
  const [activeSlide, setActiveSlide] = useState(0);
  const [health, setHealth] = useState(null);
  const [history, setHistory] = useState([]);

  useEffect(() => {
    const timer = window.setInterval(
      () => setActiveSlide((current) => (current + 1) % slides.length),
      5200,
    );
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    fetch(`${API_BASE}/api/health`)
      .then((response) => response.json())
      .then(setHealth)
      .catch(() => setHealth({ status: "offline", model_loaded: false, device: "unknown" }));
  }, []);

  const currentSlide = slides[activeSlide];

  return (
    <main>
      <nav className="topbar">
        <div className="brand">
          <Flame size={24} />
          <span>火灾智能检测平台</span>
        </div>
        <div className="nav-actions">
          <a href="#image-detect">图片检测</a>
          <a href="#video-detect">视频检测</a>
          <a href="#camera">摄像头</a>
          <StatusPill health={health} />
        </div>
      </nav>

      <section className="hero">
        <img src={currentSlide.image} alt="" />
        <div className="hero-scrim" />
        <div className="hero-content">
          <span className="kicker">{currentSlide.kicker}</span>
          <h1>{currentSlide.title}</h1>
          <p>{currentSlide.text}</p>
          <div className="hero-actions">
            <a href="#image-detect" className="primary-action">
              <ImageUp size={18} />
              上传图片
            </a>
            <a href="#video-detect" className="secondary-action">
              <Video size={18} />
              上传视频
            </a>
          </div>
        </div>
        <div className="slide-dots">
          {slides.map((slide, index) => (
            <button
              aria-label={`切换到${slide.kicker}`}
              className={index === activeSlide ? "active" : ""}
              key={slide.kicker}
              onClick={() => setActiveSlide(index)}
            />
          ))}
        </div>
      </section>

      <section className="metrics">
        <Metric icon={<ShieldCheck />} label="模型权重" value="Smoke Fire.pt" />
        <Metric icon={<Activity />} label="检测类别" value="smoke / fire" />
        <Metric icon={<RadioTower />} label="运行设备" value={health?.device || "检测中"} />
      </section>

      <section className="workspace">
        <DetectorPanel
          id="image-detect"
          type="image"
          title="图片检测"
          accept="image/*"
          endpoint="/api/detect/image"
          onResult={(result) => setHistory((items) => [{ type: "图片", result }, ...items].slice(0, 6))}
        />
        <DetectorPanel
          id="video-detect"
          type="video"
          title="视频检测"
          accept="video/*"
          endpoint="/api/detect/video"
          onResult={(result) => setHistory((items) => [{ type: "视频", result }, ...items].slice(0, 6))}
        />
      </section>

      <section className="info-grid">
        <CameraPanel />
        <HistoryPanel history={history} />
        <AboutPanel />
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

function Metric({ icon, label, value }) {
  return (
    <div className="metric">
      <span className="metric-icon">{icon}</span>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function DetectorPanel({ id, type, title, accept, endpoint, onResult }) {
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
      setError("请先选择文件。");
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
      const response = await fetch(`${API_BASE}${endpoint}`, {
        method: "POST",
        body: form,
      });
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

  return (
    <section className="panel" id={id}>
      <div className="panel-heading">
        <div>
          <span className="section-label">{title}</span>
          <h2>{type === "image" ? "上传图片并获取标注结果" : "上传视频并生成检测视频"}</h2>
        </div>
        {type === "image" ? <ImageUp size={26} /> : <Video size={26} />}
      </div>

      <form className="detector-form" onSubmit={submitDetection}>
        <label className="dropzone">
          <input
            accept={accept}
            type="file"
            onChange={(event) => setFile(event.target.files?.[0] || null)}
          />
          <span>{file ? file.name : "选择检测文件"}</span>
        </label>

        <div className="controls">
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
            输入尺寸
            <select value={imgsz} onChange={(event) => setImgsz(Number(event.target.value))}>
              <option value="512">512</option>
              <option value="640">640</option>
              <option value="960">960</option>
              <option value="1280">1280</option>
            </select>
          </label>
        </div>

        <button className="submit-button" disabled={loading} type="submit">
          {loading ? <Loader2 className="spin" size={18} /> : <PlayCircle size={18} />}
          {loading ? "检测中" : "开始检测"}
        </button>
      </form>

      {error && <div className="message error">{error}</div>}

      <div className="result-grid">
        {previewUrl && (
          <Preview title="原始文件" type={type} src={previewUrl} />
        )}
        {result && (
          <Preview
            title="检测结果"
            type={type}
            src={mediaUrl(type === "image" ? result.result_image_url : result.result_video_url)}
          />
        )}
      </div>

      {result && <ResultSummary result={result} />}
    </section>
  );
}

function Preview({ title, type, src }) {
  return (
    <div className="preview">
      <strong>{title}</strong>
      {type === "image" ? (
        <img src={src} alt={title} />
      ) : (
        <video controls src={src} />
      )}
    </div>
  );
}

function ResultSummary({ result }) {
  const summary = result.summary || {};
  const detections = result.detections || [];
  return (
    <div className={`summary ${result.alert_level}`}>
      <div>
        <span>告警等级</span>
        <strong>{levelText(result.alert_level)}</strong>
      </div>
      <p>{summary.message}</p>
      <div className="chips">
        <span>火焰 {summary.fire_count || 0}</span>
        <span>烟雾 {summary.smoke_count || 0}</span>
        <span>最高置信度 {((summary.max_confidence || 0) * 100).toFixed(1)}%</span>
      </div>
      {detections.length > 0 && (
        <div className="detection-list">
          {detections.slice(0, 8).map((item, index) => (
            <span key={`${item.class_name}-${index}`}>
              {item.frame ? `F${item.frame} ` : ""}
              {item.class_name} {(item.confidence * 100).toFixed(1)}%
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function CameraPanel() {
  return (
    <section className="panel compact" id="camera">
      <div className="panel-heading">
        <div>
          <span className="section-label">实时摄像头</span>
          <h2>本地窗口检测</h2>
        </div>
        <Camera size={25} />
      </div>
      <p>摄像头入口沿用项目现有脚本，适合 Windows 本地演示。</p>
      <code>python predict_camera.py</code>
    </section>
  );
}

function HistoryPanel({ history }) {
  return (
    <section className="panel compact">
      <div className="panel-heading">
        <div>
          <span className="section-label">检测记录</span>
          <h2>最近结果</h2>
        </div>
        <Activity size={25} />
      </div>
      {history.length === 0 ? (
        <p>暂无检测记录。</p>
      ) : (
        <div className="history-list">
          {history.map((item, index) => (
            <div key={index}>
              <strong>{item.type}</strong>
              <span>{levelText(item.result.alert_level)}</span>
              <small>{item.result.summary?.message}</small>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function AboutPanel() {
  return (
    <section className="panel compact">
      <div className="panel-heading">
        <div>
          <span className="section-label">系统说明</span>
          <h2>YOLO 烟火检测</h2>
        </div>
        <ShieldCheck size={25} />
      </div>
      <p>后端使用 FastAPI 加载 Ultralytics YOLO 权重，前端使用 React 展示上传、推理结果和告警摘要。</p>
      <div className="stack-tags">
        <span>React</span>
        <span>Vite</span>
        <span>FastAPI</span>
        <span>YOLO</span>
      </div>
    </section>
  );
}

function levelText(level) {
  return {
    danger: "危险",
    warning: "预警",
    safe: "安全",
  }[level] || "未知";
}

createRoot(document.getElementById("root")).render(<App />);
