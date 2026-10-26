'use client';

/**
 * Industrial Machine Fault Detection & Predictive Maintenance — Frontend
 * Stack: React / Next.js (App Router) + Tailwind CSS + lucide-react + Recharts
 *
 * Usage (Next.js App Router):
 *   app/page.jsx  ->  import Dashboard from '../components/FaultDetectionDashboard';
 *                     export default function Page() { return <Dashboard />; }
 *
 * Backend: set NEXT_PUBLIC_API_URL (e.g. http://localhost:8000). If an endpoint is
 * unreachable, the UI automatically falls back to a built-in mock so it always works.
 */

import React, { useState, useMemo, useRef, useEffect, useCallback } from 'react';
import {
  Activity, AlertTriangle, BarChart3, CheckCircle2, ChevronRight, Cpu, Download,
  FileUp, FlaskConical, Gauge as GaugeIcon, Loader2, Play, Siren, Table2,
  Thermometer, Upload, Wrench, Zap, Database, Sparkles, XCircle,
} from 'lucide-react';
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, BarChart, Bar, Cell, ReferenceDot, ReferenceLine, AreaChart, Area,
} from 'recharts';

/* ------------------------------------------------------------------ */
/* Config & constants                                                  */
/* ------------------------------------------------------------------ */

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? '';
const THRESHOLD = 0.9;

const SENSORS = [
  { key: 'Air_temperature_K', label: 'Air Temperature', unit: 'K', min: 295, max: 305, step: 0.1, def: 298.1, icon: Thermometer },
  { key: 'Process_temperature_K', label: 'Process Temperature', unit: 'K', min: 305, max: 315, step: 0.1, def: 308.6, icon: Thermometer },
  { key: 'Rotational_speed_rpm', label: 'Rotational Speed', unit: 'rpm', min: 1100, max: 2900, step: 1, def: 1551, icon: Activity },
  { key: 'Torque_Nm', label: 'Torque', unit: 'Nm', min: 3, max: 80, step: 0.1, def: 42.8, icon: Zap },
  { key: 'Tool_wear_min', label: 'Tool Wear', unit: 'min', min: 0, max: 260, step: 1, def: 0, icon: Wrench },
];

const DEFAULT_INPUT = {
  Type: 'M',
  ...Object.fromEntries(SENSORS.map((s) => [s.key, s.def])),
};

const TONES = {
  SAFE: {
    label: 'NORMAL', headline: 'Machine Operational',
    text: 'text-emerald-400', hex: '#34d399',
    badge: 'bg-emerald-500/10 text-emerald-300 border-emerald-500/40',
    card: 'border-emerald-500/40 shadow-[0_0_40px_-10px_rgba(52,211,153,0.45)]',
    action: 'Standard routine inspection.', icon: CheckCircle2,
  },
  WARNING: {
    label: 'WARNING', headline: 'Early Degradation Detected',
    text: 'text-amber-400', hex: '#fbbf24',
    badge: 'bg-amber-500/10 text-amber-300 border-amber-500/40',
    card: 'border-amber-500/40 shadow-[0_0_40px_-10px_rgba(251,191,36,0.5)]',
    action: 'Schedule inspection within 24 h.', icon: AlertTriangle,
  },
  CRITICAL: {
    label: 'CRITICAL', headline: 'Imminent Failure Alert!',
    text: 'text-rose-400', hex: '#fb7185',
    badge: 'bg-rose-500/10 text-rose-300 border-rose-500/50',
    card: 'border-rose-500/60 shadow-[0_0_60px_-8px_rgba(244,63,94,0.65)] animate-pulse',
    action: 'Emergency shutdown & replace tool.', icon: Siren,
  },
};

const statusFromProb = (p) => (p >= THRESHOLD ? 'CRITICAL' : p >= 0.5 ? 'WARNING' : 'SAFE');

const chartTooltip = {
  contentStyle: { background: '#0f172a', border: '1px solid #334155', borderRadius: 8, fontSize: 12 },
  labelStyle: { color: '#94a3b8' },
};

/* ------------------------------------------------------------------ */
/* API layer (live endpoint with mock fallback)                        */
/* ------------------------------------------------------------------ */

/** Heuristic stand-in for the MLP, loosely based on AI4I failure modes. */
function mockPredict(s) {
  const dT = s.Process_temperature_K - s.Air_temperature_K;
  const power = (s.Torque_Nm * s.Rotational_speed_rpm * 2 * Math.PI) / 60;
  const strain = s.Tool_wear_min * s.Torque_Nm;
  const limit = { L: 11000, M: 12000, H: 13000 }[s.Type] ?? 12000;
  let z = -5.2;
  z += Math.max(0, s.Torque_Nm - 45) * 0.06;
  z += Math.max(0, s.Tool_wear_min - 150) * 0.025;
  if (dT < 8.6 && s.Rotational_speed_rpm < 1380) z += 4.5;
  if (power < 3500 || power > 9000) z += 3.5;
  if (strain > limit) z += 5;
  const p = 1 / (1 + Math.exp(-z));
  const status = statusFromProb(p);
  return {
    failure_probability: +p.toFixed(4),
    prediction: p >= THRESHOLD ? 1 : 0,
    status,
    threshold: THRESHOLD,
    message:
      status === 'SAFE' ? 'Optimal machine conditions.'
      : status === 'WARNING' ? 'Sensor pattern shows early degradation.'
      : 'Failure signature detected across sensors.',
  };
}

const api = {
  async predict(input) {
    try {
      const res = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(input),
      });
      if (!res.ok) throw new Error(res.statusText);
      return { ...(await res.json()), _source: 'live' };
    } catch {
      await new Promise((r) => setTimeout(r, 700)); // simulate latency
      return { ...mockPredict(input), _source: 'mock' };
    }
  },
  async predictBatch(file, records) {
    try {
      const fd = new FormData();
      fd.append('file', file);
      const res = await fetch(`${API_BASE}/api/predict-batch`, { method: 'POST', body: fd });
      if (!res.ok) throw new Error(res.statusText);
      const data = await res.json();
      const arr = Array.isArray(data) ? data : data.results ?? data.predictions;
      if (!Array.isArray(arr) || arr.length !== records.length) throw new Error('shape');
      return {
        source: 'live',
        rows: records.map((r, i) => {
          const p = arr[i].failure_probability ?? 0;
          return { ...r, failure_probability: p, prediction: arr[i].prediction ?? (p >= THRESHOLD ? 1 : 0), status: arr[i].status ?? statusFromProb(p) };
        }),
      };
    } catch {
      await new Promise((r) => setTimeout(r, 500));
      return { source: 'mock', rows: records.map((r) => ({ ...r, ...mockPredict(r) })) };
    }
  },
};

/* ------------------------------------------------------------------ */
/* CSV utilities                                                       */
/* ------------------------------------------------------------------ */

function parseCSV(text) {
  const rows = [];
  let row = [], cell = '', q = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (q) {
      if (ch === '"') { if (text[i + 1] === '"') { cell += '"'; i++; } else q = false; }
      else cell += ch;
    } else if (ch === '"') q = true;
    else if (ch === ',') { row.push(cell); cell = ''; }
    else if (ch === '\n' || ch === '\r') {
      if (ch === '\r' && text[i + 1] === '\n') i++;
      row.push(cell); cell = '';
      if (row.some((c) => c.trim() !== '')) rows.push(row);
      row = [];
    } else cell += ch;
  }
  if (cell !== '' || row.length) { row.push(cell); if (row.some((c) => c.trim() !== '')) rows.push(row); }
  if (!rows.length) return { headers: [], rows: [] };
  const [head, ...body] = rows;
  return { headers: head.map((h) => h.trim()), rows: body };
}

const norm = (s) => s.toLowerCase().replace(/[^a-z0-9]/g, '');
const FIELD_MAP = {
  type: 'Type',
  airtemperaturek: 'Air_temperature_K',
  processtemperaturek: 'Process_temperature_K',
  rotationalspeedrpm: 'Rotational_speed_rpm',
  torquenm: 'Torque_Nm',
  toolwearmin: 'Tool_wear_min',
};

function toRecords(parsed) {
  const idx = {};
  parsed.headers.forEach((h, i) => { const k = FIELD_MAP[norm(h)]; if (k) idx[k] = i; });
  const missing = Object.values(FIELD_MAP).filter((k) => !(k in idx));
  if (missing.length) throw new Error(`Missing required columns: ${missing.join(', ')}`);
  return parsed.rows.map((r) => ({
    Type: (r[idx.Type] || 'M').trim().toUpperCase(),
    Air_temperature_K: parseFloat(r[idx.Air_temperature_K]),
    Process_temperature_K: parseFloat(r[idx.Process_temperature_K]),
    Rotational_speed_rpm: parseFloat(r[idx.Rotational_speed_rpm]),
    Torque_Nm: parseFloat(r[idx.Torque_Nm]),
    Tool_wear_min: parseFloat(r[idx.Tool_wear_min]),
  })).filter((r) => !Number.isNaN(r.Torque_Nm));
}

function downloadCSV(filename, rows) {
  if (!rows.length) return;
  const cols = Object.keys(rows[0]);
  const csv = [cols.join(','), ...rows.map((r) => cols.map((c) => r[c]).join(','))].join('\n');
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
  const a = document.createElement('a');
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

const DEMO_BATCH_CSV = `Type,Air temperature [K],Process temperature [K],Rotational speed [rpm],Torque [Nm],Tool wear [min]
M,298.1,308.6,1551,42.8,0
L,298.9,309.0,1410,65.7,191
H,300.5,309.4,1700,38.2,45
L,301.2,310.1,1325,70.4,215
M,297.4,308.1,2050,28.6,112
H,299.0,308.2,1290,62.0,225
L,302.0,311.2,1480,55.9,240
M,298.6,309.3,1620,44.0,30`;

/* ------------------------------------------------------------------ */
/* Shared UI atoms                                                     */
/* ------------------------------------------------------------------ */

function Card({ className = '', children }) {
  return <div className={`rounded-2xl border border-slate-800 bg-slate-900/70 backdrop-blur p-5 ${className}`}>{children}</div>;
}

function SectionTitle({ icon: Icon, children, right }) {
  return (
    <div className="mb-4 flex items-center justify-between">
      <h3 className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wider text-slate-300">
        <Icon className="h-4 w-4 text-cyan-400" /> {children}
      </h3>
      {right}
    </div>
  );
}

function StatusPill({ status }) {
  const t = TONES[status] ?? TONES.SAFE;
  const Icon = t.icon;
  return (
    <span className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-semibold ${t.badge}`}>
      <Icon className="h-3 w-3" /> {t.label}
    </span>
  );
}

function Dropzone({ onFile, label, hint }) {
  const [over, setOver] = useState(false);
  const ref = useRef(null);
  return (
    <div
      onClick={() => ref.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); const f = e.dataTransfer.files?.[0]; if (f) onFile(f); }}
      className={`flex cursor-pointer flex-col items-center justify-center gap-2 rounded-2xl border-2 border-dashed px-6 py-10 text-center transition-all duration-300 ${
        over ? 'border-cyan-400 bg-cyan-500/10 scale-[1.01]' : 'border-slate-700 bg-slate-900/40 hover:border-cyan-600 hover:bg-slate-900/70'
      }`}
    >
      <Upload className={`h-8 w-8 ${over ? 'text-cyan-300' : 'text-slate-500'}`} />
      <p className="text-sm font-medium text-slate-200">{label}</p>
      <p className="text-xs text-slate-500">{hint}</p>
      <input ref={ref} type="file" accept=".csv,text/csv" className="hidden"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); e.target.value = ''; }} />
    </div>
  );
}

function SensorSlider({ cfg, value, onChange }) {
  const Icon = cfg.icon;
  const pct = ((value - cfg.min) / (cfg.max - cfg.min)) * 100;
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950/50 p-4 transition-colors hover:border-slate-700">
      <div className="mb-3 flex items-center justify-between gap-3">
        <label className="flex items-center gap-2 text-sm text-slate-300">
          <Icon className="h-4 w-4 text-cyan-400" /> {cfg.label}
          <span className="text-xs text-slate-500">[{cfg.unit}]</span>
        </label>
        <input
          type="number" value={value} min={cfg.min} max={cfg.max} step={cfg.step}
          onChange={(e) => { const v = parseFloat(e.target.value); if (!Number.isNaN(v)) onChange(Math.min(cfg.max, Math.max(cfg.min, v))); }}
          className="w-24 rounded-lg border border-slate-700 bg-slate-900 px-2 py-1 text-right font-mono text-sm text-cyan-300 outline-none focus:border-cyan-500"
        />
      </div>
      <input
        type="range" min={cfg.min} max={cfg.max} step={cfg.step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="h-2 w-full cursor-pointer appearance-none rounded-full accent-cyan-400"
        style={{ background: `linear-gradient(to right, #22d3ee ${pct}%, #1e293b ${pct}%)` }}
      />
      <div className="mt-1 flex justify-between font-mono text-[10px] text-slate-600">
        <span>{cfg.min}</span><span>{cfg.max}</span>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Radial gauge                                                        */
/* ------------------------------------------------------------------ */

function RadialGauge({ value, status, idle }) {
  const r = 88, c = 2 * Math.PI * r, arc = c * 0.75;
  const tone = TONES[status] ?? TONES.SAFE;
  const color = idle ? '#334155' : tone.hex;
  const filled = arc * Math.min(1, Math.max(0, value));
  return (
    <div className="relative mx-auto h-60 w-60">
      <svg viewBox="0 0 220 220" className="h-full w-full rotate-[135deg]">
        <circle cx="110" cy="110" r={r} fill="none" stroke="#1e293b" strokeWidth="16" strokeLinecap="round" strokeDasharray={`${arc} ${c}`} />
        <circle
          cx="110" cy="110" r={r} fill="none" stroke={color} strokeWidth="16" strokeLinecap="round"
          strokeDasharray={`${filled} ${c}`}
          style={{ transition: 'stroke-dasharray 900ms cubic-bezier(.22,1,.36,1), stroke 400ms', filter: idle ? 'none' : `drop-shadow(0 0 8px ${color})` }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-[11px] uppercase tracking-widest text-slate-500">Failure probability</span>
        <span className={`font-mono text-5xl font-bold ${idle ? 'text-slate-600' : tone.text}`}>
          {idle ? '--' : (value * 100).toFixed(1)}
          {!idle && <span className="text-2xl">%</span>}
        </span>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Tab 1 — Live Diagnostic Cockpit                                     */
/* ------------------------------------------------------------------ */

function CockpitTab({ onSource }) {
  const [input, setInput] = useState(DEFAULT_INPUT);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [stale, setStale] = useState(false);

  const set = (k, v) => { setInput((p) => ({ ...p, [k]: v })); if (result) setStale(true); };

  const dT = input.Process_temperature_K - input.Air_temperature_K;
  const powerW = (input.Torque_Nm * input.Rotational_speed_rpm * 2 * Math.PI) / 60;

  const run = async () => {
    setLoading(true);
    const res = await api.predict(input);
    setResult(res); setStale(false); setLoading(false); onSource(res._source);
  };

  const status = result ? statusFromProb(result.failure_probability) : 'SAFE';
  const tone = TONES[status];
  const StatusIcon = tone.icon;

  return (
    <div className="grid gap-6 lg:grid-cols-5">
      <div className="space-y-4 lg:col-span-3">
        <Card>
          <SectionTitle icon={Cpu}>Machine Type</SectionTitle>
          <div className="grid grid-cols-3 gap-3">
            {[['H', 'High'], ['M', 'Medium'], ['L', 'Low']].map(([k, name]) => (
              <button key={k} onClick={() => set('Type', k)}
                className={`rounded-xl border px-4 py-3 text-left transition-all duration-200 ${
                  input.Type === k ? 'border-cyan-400 bg-cyan-500/10 shadow-[0_0_20px_-6px_rgba(34,211,238,0.6)]' : 'border-slate-800 bg-slate-950/50 hover:border-slate-600'
                }`}>
                <div className={`font-mono text-xl font-bold ${input.Type === k ? 'text-cyan-300' : 'text-slate-400'}`}>{k}</div>
                <div className="text-xs text-slate-500">{name} quality</div>
              </button>
            ))}
          </div>
        </Card>
        <div className="grid gap-4 sm:grid-cols-2">
          {SENSORS.map((s) => <SensorSlider key={s.key} cfg={s} value={input[s.key]} onChange={(v) => set(s.key, v)} />)}
          <button onClick={() => { setInput(DEFAULT_INPUT); setResult(null); setStale(false); }}
            className="rounded-xl border border-dashed border-slate-700 text-sm text-slate-500 transition-colors hover:border-slate-500 hover:text-slate-300">
            Reset to defaults
          </button>
        </div>
      </div>

      <div className="space-y-4 lg:col-span-2">
        <Card className={`transition-all duration-500 ${result && !stale ? tone.card : ''}`}>
          <SectionTitle icon={GaugeIcon}>Diagnostic Result</SectionTitle>
          <RadialGauge value={result?.failure_probability ?? 0} status={status} idle={!result} />
          <div className={`mt-2 text-center transition-opacity ${stale ? 'opacity-40' : ''}`}>
            {result ? (
              <>
                <div className="flex justify-center"><StatusPill status={status} /></div>
                <p className={`mt-3 flex items-center justify-center gap-2 text-lg font-semibold ${tone.text}`}>
                  <StatusIcon className="h-5 w-5" /> {tone.headline}
                </p>
                <p className="mt-1 text-xs text-slate-500">{result.message}</p>
                <div className="mt-4 rounded-xl border border-slate-800 bg-slate-950/60 p-3 text-left">
                  <p className="text-[10px] uppercase tracking-widest text-slate-500">Recommended action</p>
                  <p className="mt-1 flex items-center gap-1 text-sm text-slate-200"><ChevronRight className={`h-4 w-4 ${tone.text}`} />{tone.action}</p>
                </div>
                <p className="mt-2 text-[11px] text-slate-600">Decision threshold τ = {result.threshold} · model output {(result.failure_probability * 100).toFixed(2)}%</p>
              </>
            ) : (
              <p className="text-sm text-slate-500">Adjust the sensors, then run a diagnostic.</p>
            )}
            {stale && <p className="mt-2 text-xs text-amber-400">Inputs changed — re-run to refresh.</p>}
          </div>
          <button onClick={run} disabled={loading}
            className="mt-5 flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-cyan-500 to-emerald-500 py-3 font-semibold text-slate-950 shadow-lg shadow-cyan-500/20 transition-all hover:brightness-110 active:scale-[0.98] disabled:opacity-60">
            {loading ? <><Loader2 className="h-5 w-5 animate-spin" /> Analysing telemetry…</> : <><Play className="h-5 w-5" /> Run Fault Diagnostic</>}
          </button>
        </Card>

        <div className="grid grid-cols-2 gap-4">
          <Card className="!p-4">
            <p className="flex items-center gap-1 text-[11px] uppercase tracking-wider text-slate-500"><Thermometer className="h-3 w-3" /> Differential Temp ΔT</p>
            <p className={`mt-1 font-mono text-2xl font-bold ${dT < 8.6 ? 'text-amber-400' : 'text-cyan-300'}`}>{dT.toFixed(1)} <span className="text-sm text-slate-500">K</span></p>
            <p className="text-[11px] text-slate-600">T_process − T_air</p>
          </Card>
          <Card className="!p-4">
            <p className="flex items-center gap-1 text-[11px] uppercase tracking-wider text-slate-500"><Zap className="h-3 w-3" /> Power Dissipation</p>
            <p className={`mt-1 font-mono text-2xl font-bold ${powerW < 3500 || powerW > 9000 ? 'text-amber-400' : 'text-cyan-300'}`}>{(powerW / 1000).toFixed(2)} <span className="text-sm text-slate-500">kW</span></p>
            <p className="text-[11px] text-slate-600">P ∝ Torque × Speed</p>
          </Card>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Tab 2 — Batch CSV                                                   */
/* ------------------------------------------------------------------ */

function BatchTab({ onSource }) {
  const [rows, setRows] = useState([]);
  const [filter, setFilter] = useState('ALL');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [fileName, setFileName] = useState('');

  const process = useCallback(async (file) => {
    setError(''); setLoading(true); setFileName(file.name);
    try {
      const records = toRecords(parseCSV(await file.text()));
      if (!records.length) throw new Error('No valid rows found in this CSV.');
      const res = await api.predictBatch(file, records);
      setRows(res.rows); onSource(res.source);
    } catch (e) { setError(e.message); setRows([]); }
    setLoading(false);
  }, [onSource]);

  const loadDemo = () => process(new File([DEMO_BATCH_CSV], 'sample_test_records.csv', { type: 'text/csv' }));

  const failures = rows.filter((r) => r.prediction === 1).length;
  const warnings = rows.filter((r) => r.prediction === 0 && r.status === 'WARNING').length;
  const shown = rows.filter((r) => filter === 'ALL' ? true : filter === 'FAIL' ? r.prediction === 1 : r.prediction === 0);

  const exportRows = () => downloadCSV('predictions.csv', rows.map((r) => ({
    Type: r.Type, 'Air temperature [K]': r.Air_temperature_K, 'Process temperature [K]': r.Process_temperature_K,
    'Rotational speed [rpm]': r.Rotational_speed_rpm, 'Torque [Nm]': r.Torque_Nm, 'Tool wear [min]': r.Tool_wear_min,
    failure_probability: r.failure_probability, prediction: r.prediction, status: r.status,
  })));

  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-3">
        <div className="md:col-span-2"><Dropzone onFile={process} label="Drag & drop a CSV or click to browse" hint="Columns: Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min]" /></div>
        <Card className="flex flex-col justify-center gap-3">
          <button onClick={loadDemo} className="flex items-center justify-center gap-2 rounded-xl border border-cyan-500/40 bg-cyan-500/10 py-3 text-sm font-semibold text-cyan-300 transition-all hover:bg-cyan-500/20">
            <Sparkles className="h-4 w-4" /> Load Demo Sample CSV
          </button>
          <button onClick={exportRows} disabled={!rows.length}
            className="flex items-center justify-center gap-2 rounded-xl border border-slate-700 py-3 text-sm font-semibold text-slate-200 transition-all hover:border-emerald-500 hover:text-emerald-300 disabled:cursor-not-allowed disabled:opacity-40">
            <Download className="h-4 w-4" /> Export Predictions CSV
          </button>
        </Card>
      </div>

      {error && <div className="flex items-center gap-2 rounded-xl border border-rose-500/40 bg-rose-500/10 p-3 text-sm text-rose-300"><XCircle className="h-4 w-4" />{error}</div>}
      {loading && <div className="flex items-center gap-2 text-sm text-cyan-300"><Loader2 className="h-4 w-4 animate-spin" /> Running batch inference…</div>}

      {rows.length > 0 && (
        <Card className="!p-0 overflow-hidden">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 p-4">
            <div>
              <p className="flex items-center gap-2 text-sm font-semibold text-slate-200"><Table2 className="h-4 w-4 text-cyan-400" /> {fileName}</p>
              <p className="text-xs text-slate-500">{rows.length} records · <span className="text-rose-400">{failures} failure</span> · <span className="text-amber-400">{warnings} warning</span> · <span className="text-emerald-400">{rows.length - failures - warnings} normal</span></p>
            </div>
            <div className="flex rounded-lg border border-slate-800 bg-slate-950 p-1 text-xs">
              {[['ALL', 'All'], ['FAIL', 'Only Failures'], ['OK', 'Only Normal']].map(([k, l]) => (
                <button key={k} onClick={() => setFilter(k)} className={`rounded-md px-3 py-1.5 font-medium transition-colors ${filter === k ? 'bg-cyan-500/20 text-cyan-300' : 'text-slate-500 hover:text-slate-300'}`}>{l}</button>
              ))}
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="bg-slate-950/60 text-[11px] uppercase tracking-wider text-slate-500">
                <tr>{['#', 'Type', 'Air [K]', 'Process [K]', 'RPM', 'Torque [Nm]', 'Wear [min]', 'Failure prob.', 'Status'].map((h) => <th key={h} className="px-4 py-3 font-medium">{h}</th>)}</tr>
              </thead>
              <tbody className="divide-y divide-slate-800/70 font-mono text-slate-300">
                {shown.map((r, i) => {
                  const tone = TONES[r.status] ?? TONES.SAFE;
                  return (
                    <tr key={i} className={`transition-colors hover:bg-slate-800/40 ${r.prediction === 1 ? 'bg-rose-500/5' : ''}`}>
                      <td className="px-4 py-2.5 text-slate-600">{i + 1}</td>
                      <td className="px-4 py-2.5">{r.Type}</td>
                      <td className="px-4 py-2.5">{r.Air_temperature_K}</td>
                      <td className="px-4 py-2.5">{r.Process_temperature_K}</td>
                      <td className="px-4 py-2.5">{r.Rotational_speed_rpm}</td>
                      <td className="px-4 py-2.5">{r.Torque_Nm}</td>
                      <td className="px-4 py-2.5">{r.Tool_wear_min}</td>
                      <td className="px-4 py-2.5">
                        <div className="flex items-center gap-2">
                          <div className="h-1.5 w-20 overflow-hidden rounded-full bg-slate-800"><div className="h-full rounded-full transition-all duration-700" style={{ width: `${r.failure_probability * 100}%`, background: tone.hex }} /></div>
                          <span className={tone.text}>{(r.failure_probability * 100).toFixed(1)}%</span>
                        </div>
                      </td>
                      <td className="px-4 py-2.5 font-sans">{r.prediction === 1 ? <StatusPill status="CRITICAL" /> : <StatusPill status={r.status} />}</td>
                    </tr>
                  );
                })}
                {!shown.length && <tr><td colSpan={9} className="px-4 py-8 text-center font-sans text-slate-500">No records match this filter.</td></tr>}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Tab 3 — Model Performance & XAI                                     */
/* ------------------------------------------------------------------ */

/*
 * Headline numbers (confusion matrix, AUC, feature importance) come from your
 * evaluation report. The ROC / PR / training curves below are RECONSTRUCTED
 * shapes that match those headline numbers. To show your real curves, export
 * them from train_model.py (roc_curve, precision_recall_curve, training_history.csv)
 * and replace the generators below with the real arrays.
 */
const CM = { tn: 1898, fp: 34, fn: 14, tp: 54 };
const AUC = 0.973;

const ROC_DATA = (() => {
  const k = 1 / AUC - 1;
  const pts = [{ fpr: 0, tpr: 0 }];
  for (let i = 1; i <= 60; i++) { const f = Math.pow(i / 60, 2.2); pts.push({ fpr: +f.toFixed(4), tpr: +Math.pow(f, k).toFixed(4) }); }
  return pts;
})();

const PR_DATA = Array.from({ length: 51 }, (_, i) => {
  const r = i / 50;
  return { recall: r, precision: +(0.97 / (1 + Math.exp((r - 0.84) * 9)) + 0.03).toFixed(4) };
});

const TRAIN_DATA = Array.from({ length: 50 }, (_, i) => {
  const e = i + 1, n = Math.sin(e * 1.7);
  return {
    epoch: e,
    loss: +(0.52 * Math.exp(-e / 9) + 0.11 + 0.01 * n).toFixed(4),
    val_loss: +(0.46 * Math.exp(-e / 8) + 0.13 + 0.015 * Math.cos(e * 1.3)).toFixed(4),
    accuracy: +(0.97 - 0.12 * Math.exp(-e / 7) + 0.002 * n).toFixed(4),
    val_accuracy: +(0.968 - 0.1 * Math.exp(-e / 7) + 0.003 * Math.cos(e * 1.3)).toFixed(4),
  };
});

const IMPORTANCE = [
  { name: 'Torque [Nm]', value: 38 },
  { name: 'Rotational Speed [rpm]', value: 28 },
  { name: 'Tool Wear [min]', value: 18 },
  { name: 'Temperature Gradient', value: 11 },
  { name: 'Machine Type', value: 5 },
];

function ConfusionMatrix() {
  const cell = (label, value, sub, cls) => (
    <div className={`flex flex-col items-center justify-center rounded-xl border p-5 text-center ${cls}`}>
      <span className="text-[10px] uppercase tracking-widest opacity-70">{label}</span>
      <span className="font-mono text-3xl font-bold">{value.toLocaleString()}</span>
      <span className="text-[11px] opacity-70">{sub}</span>
    </div>
  );
  return (
    <div>
      <div className="mb-2 grid grid-cols-[auto_1fr_1fr] gap-2 text-center text-[11px] text-slate-500">
        <span /><span>Predicted Normal</span><span>Predicted Failure</span>
      </div>
      <div className="grid grid-cols-[auto_1fr_1fr] gap-2">
        <span className="flex items-center text-[11px] text-slate-500 [writing-mode:vertical-rl] rotate-180">Actual Normal</span>
        {cell('True Negative', CM.tn, 'Specificity 98.24%', 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300')}
        {cell('False Positive', CM.fp, 'False alarms', 'border-amber-500/40 bg-amber-500/10 text-amber-300')}
        <span className="flex items-center text-[11px] text-slate-500 [writing-mode:vertical-rl] rotate-180">Actual Failure</span>
        {cell('False Negative', CM.fn, 'Missed failures', 'border-rose-500/40 bg-rose-500/10 text-rose-300')}
        {cell('True Positive', CM.tp, 'Recall 79.41%', 'border-cyan-500/40 bg-cyan-500/10 text-cyan-300')}
      </div>
    </div>
  );
}

function PerformanceTab() {
  const kpis = [['ROC-AUC', '0.973'], ['Recall', '79.41%'], ['Precision', '61.36%'], ['F1-Score', '0.692'], ['Accuracy', '97.60%'], ['Threshold τ', '0.90']];
  const axis = { stroke: '#475569', fontSize: 11, tickLine: false };
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-6">
        {kpis.map(([k, v]) => (
          <Card key={k} className="!p-4 text-center"><p className="text-[11px] uppercase tracking-wider text-slate-500">{k}</p><p className="mt-1 font-mono text-2xl font-bold text-cyan-300">{v}</p></Card>
        ))}
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card><SectionTitle icon={Table2}>Confusion Matrix · N = 2,000</SectionTitle><ConfusionMatrix /></Card>
        <Card>
          <SectionTitle icon={BarChart3}>Feature Importance (Permutation)</SectionTitle>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={IMPORTANCE} layout="vertical" margin={{ left: 20, right: 24 }}>
              <CartesianGrid stroke="#1e293b" horizontal={false} />
              <XAxis type="number" unit="%" {...axis} />
              <YAxis type="category" dataKey="name" width={140} {...axis} />
              <Tooltip {...chartTooltip} formatter={(v) => [`${v}%`, 'Importance']} cursor={{ fill: '#1e293b55' }} />
              <Bar dataKey="value" radius={[0, 6, 6, 0]} animationDuration={900}>
                {IMPORTANCE.map((_, i) => <Cell key={i} fill={['#22d3ee', '#2dd4bf', '#34d399', '#fbbf24', '#94a3b8'][i]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card>
          <SectionTitle icon={Activity} right={<span className="rounded-md bg-cyan-500/10 px-2 py-0.5 font-mono text-xs text-cyan-300">AUC = {AUC}</span>}>ROC Curve</SectionTitle>
          <ResponsiveContainer width="100%" height={280}>
            <AreaChart data={ROC_DATA} margin={{ left: -10, right: 10, top: 5 }}>
              <defs><linearGradient id="roc" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#22d3ee" stopOpacity={0.35} /><stop offset="100%" stopColor="#22d3ee" stopOpacity={0} /></linearGradient></defs>
              <CartesianGrid stroke="#1e293b" />
              <XAxis dataKey="fpr" type="number" domain={[0, 1]} label={{ value: 'False Positive Rate', position: 'insideBottom', offset: -2, fill: '#64748b', fontSize: 11 }} {...axis} />
              <YAxis domain={[0, 1]} {...axis} />
              <Tooltip {...chartTooltip} />
              <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#475569" strokeDasharray="4 4" />
              <Area type="monotone" dataKey="tpr" name="TPR" stroke="#22d3ee" strokeWidth={2.5} fill="url(#roc)" dot={false} animationDuration={1000} />
              <ReferenceDot x={CM.fp / (CM.fp + CM.tn)} y={CM.tp / (CM.tp + CM.fn)} r={6} fill="#fbbf24" stroke="#0f172a" label={{ value: 'τ = 0.90', position: 'right', fill: '#fbbf24', fontSize: 11 }} />
            </AreaChart>
          </ResponsiveContainer>
        </Card>

        <Card>
          <SectionTitle icon={Activity}>Precision–Recall Curve</SectionTitle>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={PR_DATA} margin={{ left: -10, right: 10, top: 5 }}>
              <CartesianGrid stroke="#1e293b" />
              <XAxis dataKey="recall" type="number" domain={[0, 1]} label={{ value: 'Recall', position: 'insideBottom', offset: -2, fill: '#64748b', fontSize: 11 }} {...axis} />
              <YAxis domain={[0, 1]} {...axis} />
              <Tooltip {...chartTooltip} />
              <ReferenceLine y={0.0339} stroke="#475569" strokeDasharray="4 4" label={{ value: 'baseline 3.4%', fill: '#64748b', fontSize: 10, position: 'insideBottomRight' }} />
              <Line type="monotone" dataKey="precision" name="Precision" stroke="#34d399" strokeWidth={2.5} dot={false} animationDuration={1000} />
              <ReferenceDot x={0.7941} y={0.6136} r={6} fill="#fbbf24" stroke="#0f172a" />
            </LineChart>
          </ResponsiveContainer>
        </Card>

        <Card className="lg:col-span-2">
          <SectionTitle icon={Activity}>Training vs Validation · 50 epochs</SectionTitle>
          <div className="grid gap-6 md:grid-cols-2">
            {[['Loss (weighted BCE)', 'loss', 'val_loss', '#fb7185', '#fbbf24'], ['Accuracy', 'accuracy', 'val_accuracy', '#22d3ee', '#34d399']].map(([title, a, b, c1, c2]) => (
              <div key={title}>
                <p className="mb-2 text-xs text-slate-500">{title}</p>
                <ResponsiveContainer width="100%" height={240}>
                  <LineChart data={TRAIN_DATA} margin={{ left: -10, right: 10 }}>
                    <CartesianGrid stroke="#1e293b" />
                    <XAxis dataKey="epoch" {...axis} />
                    <YAxis domain={['auto', 'auto']} {...axis} />
                    <Tooltip {...chartTooltip} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Line type="monotone" dataKey={a} name="Train" stroke={c1} strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey={b} name="Validation" stroke={c2} strokeWidth={2} dot={false} strokeDasharray="5 3" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ))}
          </div>
        </Card>
      </div>
      <p className="text-center text-[11px] text-slate-600">ROC, PR and training curves are reconstructed to match the reported metrics — swap in the exported arrays from your reports/ folder for exact curves.</p>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Tab 4 — Custom Model Studio                                         */
/* ------------------------------------------------------------------ */

function makeDemoPumpCSV() {
  let seed = 7;
  const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
  const lines = ['Vibration_mm_s,Inlet_Pressure_bar,Outlet_Pressure_bar,Motor_Current_A,Bearing_Temp_C,Pump_Type,Fault'];
  for (let i = 0; i < 145; i++) {
    const fault = rnd() < 0.22 ? 1 : 0;
    const vib = (fault ? 6 + rnd() * 5 : 1.5 + rnd() * 3).toFixed(2);
    const inP = (2 + rnd() * 0.8 - (fault ? 0.6 : 0)).toFixed(2);
    const outP = (6 + rnd() * 1 - (fault ? 1.2 : 0)).toFixed(2);
    const cur = (fault ? 22 + rnd() * 8 : 15 + rnd() * 5).toFixed(1);
    const temp = (fault ? 78 + rnd() * 20 : 52 + rnd() * 15).toFixed(1);
    const type = ['Single-stage', 'Multi-stage', 'Submersible'][Math.floor(rnd() * 3)];
    lines.push([vib, inP, outP, cur, temp, type, fault].join(','));
  }
  return lines.join('\n');
}

function auditDataset(ds, target) {
  const ti = ds.headers.indexOf(target);
  const missing = ds.rows.reduce((n, r) => n + r.filter((c) => c.trim() === '').length, 0);
  const dupes = ds.rows.length - new Set(ds.rows.map((r) => r.join('|'))).size;
  const counts = {};
  if (ti >= 0) ds.rows.forEach((r) => { const v = (r[ti] ?? '').trim() || '(empty)'; counts[v] = (counts[v] || 0) + 1; });
  const classes = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const numeric = [], categorical = [];
  ds.headers.forEach((h, ci) => {
    if (ci === ti) return;
    const vals = ds.rows.map((r) => r[ci]).filter((v) => v !== undefined && v.trim() !== '');
    const unique = new Set(vals).size;
    if (/^(id|udi|index|product.?id)$/i.test(h) || (unique === ds.rows.length && ds.rows.length > 20 && !vals.every((v) => !Number.isNaN(parseFloat(v))))) return;
    (vals.length && vals.filter((v) => !Number.isNaN(parseFloat(v))).length / vals.length > 0.9 ? numeric : categorical).push(h);
  });
  const ratio = classes.length > 1 ? classes[0][1] / classes[classes.length - 1][1] : null;
  return { missing, dupes, classes, numeric, categorical, ratio };
}

function StudioTab() {
  const [ds, setDs] = useState(null);
  const [target, setTarget] = useState('');
  const [epochs, setEpochs] = useState(50);
  const [batch, setBatch] = useState(32);
  const [modelName, setModelName] = useState('custom_fault_model_v1');
  const [training, setTraining] = useState(false);
  const [history, setHistory] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const timer = useRef(null);

  useEffect(() => () => clearInterval(timer.current), []);

  const load = async (file) => {
    const parsed = parseCSV(await file.text());
    if (!parsed.headers.length) return;
    const guess = parsed.headers.find((h) => /fault|defect|failure|label|target/i.test(h)) ?? parsed.headers[parsed.headers.length - 1];
    setDs({ name: file.name, ...parsed }); setTarget(guess); setHistory([]); setMetrics(null);
  };

  const audit = useMemo(() => (ds && target ? auditDataset(ds, target) : null), [ds, target]);
  const quality = audit ? (audit.missing === 0 && audit.dupes === 0 ? (audit.ratio && audit.ratio > 10 ? 'WARNING' : 'SAFE') : 'WARNING') : 'SAFE';

  /* Simulated training. Replace the interval with a call to your /api/train endpoint
     (stream epoch metrics via SSE/WebSocket) to make it live. */
  const train = () => {
    clearInterval(timer.current);
    setTraining(true); setHistory([]); setMetrics(null);
    let e = 0;
    timer.current = setInterval(() => {
      e += 1;
      const n = Math.sin(e * 2.1) * 0.012;
      setHistory((h) => [...h, {
        epoch: e,
        loss: +(0.62 * Math.exp(-e / (epochs / 5)) + 0.08 + n).toFixed(4),
        val_loss: +(0.58 * Math.exp(-e / (epochs / 5.5)) + 0.11 + n * 1.4).toFixed(4),
      }]);
      if (e >= epochs) {
        clearInterval(timer.current); setTraining(false);
        setMetrics({ accuracy: 0.94 + Math.random() * 0.04, precision: 0.82 + Math.random() * 0.1, recall: 0.8 + Math.random() * 0.12, auc: 0.95 + Math.random() * 0.04 });
      }
    }, Math.max(40, 6000 / epochs));
  };

  const progress = history.length / epochs;

  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-3">
        <div className="md:col-span-2"><Dropzone onFile={load} label="Upload your sensor dataset (CSV)" hint="Pumps, compressors, turbines… any numeric/categorical sensors plus a fault label column" /></div>
        <Card className="flex flex-col justify-center gap-3">
          <button onClick={() => load(new File([makeDemoPumpCSV()], 'sample_custom_pump_dataset.csv'))}
            className="flex items-center justify-center gap-2 rounded-xl border border-cyan-500/40 bg-cyan-500/10 py-3 text-sm font-semibold text-cyan-300 hover:bg-cyan-500/20">
            <Sparkles className="h-4 w-4" /> Load Demo Pump Dataset
          </button>
          <p className="text-center text-[11px] text-slate-600">145 rows · centrifugal water pump telemetry (generated)</p>
        </Card>
      </div>

      {ds && audit && (
        <>
          <div className="grid gap-6 lg:grid-cols-3">
            <Card className="lg:col-span-2">
              <SectionTitle icon={Database} right={<StatusPill status={quality} />}>Data Quality Audit · {ds.name}</SectionTitle>
              <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                {[['Rows', ds.rows.length, 'text-cyan-300'], ['Columns', ds.headers.length, 'text-cyan-300'], ['Missing cells', audit.missing, audit.missing ? 'text-amber-400' : 'text-emerald-400'], ['Duplicates', audit.dupes, audit.dupes ? 'text-amber-400' : 'text-emerald-400']].map(([k, v, c]) => (
                  <div key={k} className="rounded-xl border border-slate-800 bg-slate-950/50 p-3"><p className="text-[11px] uppercase tracking-wider text-slate-500">{k}</p><p className={`font-mono text-2xl font-bold ${c}`}>{v}</p></div>
                ))}
              </div>
              <div className="mt-4">
                <p className="mb-2 text-xs text-slate-500">Class balance of <span className="font-mono text-slate-300">{target}</span>{audit.ratio && <> · imbalance ≈ {audit.ratio.toFixed(1)} : 1</>}</p>
                <div className="space-y-2">
                  {audit.classes.slice(0, 5).map(([k, v], i) => (
                    <div key={k} className="flex items-center gap-3 text-xs">
                      <span className="w-20 truncate font-mono text-slate-400">{k}</span>
                      <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-800"><div className="h-full rounded-full transition-all duration-700" style={{ width: `${(v / ds.rows.length) * 100}%`, background: i === 0 ? '#22d3ee' : '#fb7185' }} /></div>
                      <span className="w-24 text-right font-mono text-slate-400">{v} ({((v / ds.rows.length) * 100).toFixed(1)}%)</span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="mt-4 flex flex-wrap gap-1.5 text-[11px]">
                {audit.numeric.map((h) => <span key={h} className="rounded-md border border-cyan-500/30 bg-cyan-500/10 px-2 py-0.5 text-cyan-300">{h} · numeric</span>)}
                {audit.categorical.map((h) => <span key={h} className="rounded-md border border-violet-500/30 bg-violet-500/10 px-2 py-0.5 text-violet-300">{h} · categorical</span>)}
              </div>
            </Card>

            <Card>
              <SectionTitle icon={FlaskConical}>Training Setup</SectionTitle>
              <div className="space-y-4 text-sm">
                <label className="block"><span className="text-xs text-slate-500">Target column</span>
                  <select value={target} onChange={(e) => { setTarget(e.target.value); setMetrics(null); setHistory([]); }} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-200 outline-none focus:border-cyan-500">
                    {ds.headers.map((h) => <option key={h}>{h}</option>)}
                  </select></label>
                <label className="block"><span className="text-xs text-slate-500">Model name</span>
                  <input value={modelName} onChange={(e) => setModelName(e.target.value.replace(/[^\w-]/g, '_'))} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 font-mono text-slate-200 outline-none focus:border-cyan-500" /></label>
                <label className="block"><span className="flex justify-between text-xs text-slate-500">Epochs <span className="font-mono text-cyan-300">{epochs}</span></span>
                  <input type="range" min={10} max={100} step={5} value={epochs} onChange={(e) => setEpochs(+e.target.value)} className="mt-1 w-full accent-cyan-400" /></label>
                <div><span className="text-xs text-slate-500">Batch size</span>
                  <div className="mt-1 grid grid-cols-3 gap-2">{[16, 32, 64].map((b) => <button key={b} onClick={() => setBatch(b)} className={`rounded-lg border py-1.5 font-mono transition-colors ${batch === b ? 'border-cyan-400 bg-cyan-500/10 text-cyan-300' : 'border-slate-700 text-slate-500 hover:border-slate-500'}`}>{b}</button>)}</div></div>
                <button onClick={train} disabled={training}
                  className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-cyan-500 to-emerald-500 py-3 font-semibold text-slate-950 transition-all hover:brightness-110 active:scale-[0.98] disabled:opacity-60">
                  {training ? <><Loader2 className="h-5 w-5 animate-spin" /> Training…</> : <><FileUp className="h-5 w-5" /> Train Model</>}
                </button>
              </div>
            </Card>
          </div>

          {(training || history.length > 0) && (
            <Card>
              <SectionTitle icon={Activity} right={<span className="font-mono text-xs text-slate-400">Epoch {history.length}/{epochs}</span>}>Training Progress <span className="ml-2 rounded bg-slate-800 px-1.5 py-0.5 text-[10px] normal-case text-slate-400">simulated</span></SectionTitle>
              <div className="mb-4 h-2 overflow-hidden rounded-full bg-slate-800"><div className="h-full rounded-full bg-gradient-to-r from-cyan-400 to-emerald-400 transition-all duration-200" style={{ width: `${progress * 100}%` }} /></div>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={history} margin={{ left: -10, right: 10 }}>
                  <CartesianGrid stroke="#1e293b" />
                  <XAxis dataKey="epoch" type="number" domain={[1, epochs]} stroke="#475569" fontSize={11} tickLine={false} />
                  <YAxis stroke="#475569" fontSize={11} tickLine={false} />
                  <Tooltip {...chartTooltip} /><Legend wrapperStyle={{ fontSize: 11 }} />
                  <Line type="monotone" dataKey="loss" name="Train loss" stroke="#fb7185" strokeWidth={2} dot={false} isAnimationActive={false} />
                  <Line type="monotone" dataKey="val_loss" name="Val loss" stroke="#fbbf24" strokeWidth={2} dot={false} strokeDasharray="5 3" isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
              {metrics && (
                <div className="mt-4 grid grid-cols-2 gap-3 md:grid-cols-4">
                  {Object.entries(metrics).map(([k, v]) => <div key={k} className="rounded-xl border border-emerald-500/30 bg-emerald-500/5 p-3 text-center"><p className="text-[11px] uppercase tracking-wider text-slate-500">{k}</p><p className="font-mono text-xl font-bold text-emerald-300">{(v * 100).toFixed(1)}%</p></div>)}
                  <p className="col-span-full text-xs text-slate-500">Saved to <span className="font-mono text-slate-300">models/custom_models/{modelName}/</span> (model.keras, scaler.joblib, encoder.joblib, config.json) — simulated result.</p>
                </div>
              )}
            </Card>
          )}
        </>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Root                                                                */
/* ------------------------------------------------------------------ */

const TABS = [
  { id: 'cockpit', label: 'Live Diagnostic Cockpit', icon: GaugeIcon },
  { id: 'batch', label: 'Batch CSV Diagnostic', icon: Table2 },
  { id: 'perf', label: 'Model Performance & XAI', icon: BarChart3 },
  { id: 'studio', label: 'Custom Model Studio', icon: FlaskConical },
];

export default function FaultDetectionDashboard() {
  const [tab, setTab] = useState('cockpit');
  const [source, setSource] = useState(null);

  return (
    <div className="min-h-screen bg-slate-900 bg-[radial-gradient(ellipse_at_top,rgba(34,211,238,0.08),transparent_55%)] text-slate-200">
      <header className="sticky top-0 z-20 border-b border-slate-800 bg-slate-900/80 backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3 px-4 py-4 sm:px-6">
          <div className="flex items-center gap-3">
            <div className="rounded-xl bg-gradient-to-br from-cyan-400 to-emerald-400 p-2 shadow-lg shadow-cyan-500/30"><Cpu className="h-6 w-6 text-slate-950" /></div>
            <div>
              <h1 className="text-lg font-bold leading-tight text-white">Industrial Fault Detection</h1>
              <p className="text-xs text-slate-500">Predictive Maintenance · Deep Learning MLP</p>
            </div>
          </div>
          <div className="flex items-center gap-2 rounded-full border border-slate-800 bg-slate-950/60 px-3 py-1.5 text-xs">
            <span className={`h-2 w-2 rounded-full ${source === 'live' ? 'bg-emerald-400 shadow-[0_0_8px_#34d399]' : source === 'mock' ? 'bg-amber-400' : 'bg-slate-600'}`} />
            <span className="text-slate-400">{source === 'live' ? 'Live backend' : source === 'mock' ? 'Mock mode (backend unreachable)' : 'Awaiting first request'}</span>
          </div>
        </div>
        <nav className="mx-auto flex max-w-7xl gap-1 overflow-x-auto px-4 sm:px-6">
          {TABS.map(({ id, label, icon: Icon }) => (
            <button key={id} onClick={() => setTab(id)}
              className={`flex shrink-0 items-center gap-2 border-b-2 px-4 py-3 text-sm font-medium transition-all duration-200 ${tab === id ? 'border-cyan-400 text-cyan-300' : 'border-transparent text-slate-500 hover:text-slate-300'}`}>
              <Icon className="h-4 w-4" /> {label}
            </button>
          ))}
        </nav>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6">
        <div key={tab} className="animate-[fadeIn_.35s_ease-out]">
          {tab === 'cockpit' && <CockpitTab onSource={setSource} />}
          {tab === 'batch' && <BatchTab onSource={setSource} />}
          {tab === 'perf' && <PerformanceTab />}
          {tab === 'studio' && <StudioTab />}
        </div>
      </main>

      <footer className="border-t border-slate-800 py-6 text-center text-xs text-slate-600">
        Trained on UCI AI4I 2020 · TensorFlow/Keras MLP · Threshold τ = {THRESHOLD}
      </footer>

      <style>{`@keyframes fadeIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}`}</style>
    </div>
  );
}
