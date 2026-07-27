'use client'

import { useState, useEffect } from 'react'
import axios from 'axios'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js'
import { Scatter, Line, Bar } from 'react-chartjs-2'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
)

type Prediction = {
  model_name: string;
  prediction: number;
  probability: number;
  confidence: string;
  result_text: string;
}

type AnalysisResult = {
  target: string;
  predictions: Prediction[];
  features: Record<string, number>;
  explanation?: string[];
  data_source?: string;
  n_sectors?: number;
  sde?: number;
  sde_pass?: boolean;
  welch_p?: number;
  duration_ok?: boolean;
  duration_ratio?: number;
  density_ok?: boolean;
  density_ratio?: number;
  has_secondary?: boolean;
  secondary_depth?: number;
  secondary_snr?: number;
  stellar_r?: number;
  stellar_m?: number;
  plot_data: {
    time: number[];
    raw_flux: number[];
    flat_flux: number[];
    trend: number[];
  };
  folded_data?: {
    phase: number[];
    flux: number[];
    model_phase: number[];
    model_flux: number[];
  };
  periodogram_data?: {
    periods: number[];
    sde: number[];
  };
  odd_even_data?: {
    odd_phase: number[];
    odd_flux: number[];
    even_phase: number[];
    even_flux: number[];
  };
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'

// Preset benchmark target stars
const TARGET_PRESETS = [
  { name: 'TOI-270', tag: 'Known 3-Planet System', desc: 'Bright M-dwarf with confirmed planets b, c, d' },
  { name: 'TIC 307210830', tag: 'L 98-59 System', desc: 'Terrestrial multi-planet M-dwarf host' },
  { name: 'TIC 38846515', tag: 'Exoplanet Host', desc: 'Known transiting planet host' },
  { name: 'Ross 176', tag: 'Quiet Control Star', desc: 'No transits present (flat baseline)' },
]

export default function Home() {
  const [targetStar, setTargetStar] = useState('TOI-270')
  const [maxSectors, setMaxSectors] = useState(3)
  const [loading, setLoading] = useState(false)
  const [loadingStep, setLoadingStep] = useState(0)
  const [error, setError] = useState('')
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [activeTab, setActiveTab] = useState<'folded' | 'time' | 'periodogram' | 'oddeven' | 'features'>('folded')
  const [isApiConnected, setIsApiConnected] = useState<boolean | null>(null)

  // Check backend health on mount
  useEffect(() => {
    axios.get(`${API_BASE}/api/health`)
      .then(res => setIsApiConnected(res.data.status === 'ok'))
      .catch(() => setIsApiConnected(false))
  }, [])

  // Simulate pipeline execution steps during analysis request
  useEffect(() => {
    let interval: NodeJS.Timeout
    if (loading) {
      setLoadingStep(1)
      interval = setInterval(() => {
        setLoadingStep((prev) => (prev < 5 ? prev + 1 : prev))
      }, 1500)
    } else {
      setLoadingStep(0)
    }
    return () => clearInterval(interval)
  }, [loading])

  const handleAnalyze = async (overrideTarget?: string) => {
    const target = overrideTarget || targetStar
    if (!target) return

    setLoading(true)
    setError('')
    setResult(null)

    try {
      const response = await axios.post<AnalysisResult>(`${API_BASE}/api/analyze`, {
        target_star: target,
        max_sectors: maxSectors
      })
      setResult(response.data)
      // Default to folded phase tab if available, else time series
      setActiveTab('folded')
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'An error occurred during pipeline analysis.')
    } finally {
      setLoading(false)
    }
  }

  const loadDemoData = () => {
    // Generate realistic demo phase-folded data for TOI-270
    const phasePoints: number[] = []
    const fluxPoints: number[] = []
    const modelPhasePoints: number[] = []
    const modelFluxPoints: number[] = []

    for (let p = -0.5; p <= 0.5; p += 0.002) {
      const isTransit = Math.abs(p) < 0.015
      const dip = isTransit ? 0.00373 * Math.cos((p / 0.015) * (Math.PI / 2)) : 0
      const noise = (Math.random() - 0.5) * 0.0008
      phasePoints.push(p)
      fluxPoints.push(1 - dip + noise)

      modelPhasePoints.push(p)
      modelFluxPoints.push(1 - dip)
    }

    // Demo Periodogram data
    const periodsArr: number[] = []
    const sdeArr: number[] = []
    for (let p = 0.5; p <= 15.0; p += 0.025) {
      periodsArr.push(p)
      const isPeak = Math.abs(p - 5.6604) < 0.1
      sdeArr.push(isPeak ? 39.9 - Math.abs(p - 5.6604) * 200 : 3.0 + Math.random() * 2.5)
    }

    // Demo Odd/Even data
    const oddPhase: number[] = []
    const oddFlux: number[] = []
    const evenPhase: number[] = []
    const evenFlux: number[] = []
    for (let p = -0.15; p <= 0.15; p += 0.003) {
      const isTransit = Math.abs(p) < 0.015
      const dip = isTransit ? 0.00373 * Math.cos((p / 0.015) * (Math.PI / 2)) : 0
      oddPhase.push(p)
      oddFlux.push(1 - dip + (Math.random() - 0.5) * 0.0007)
      evenPhase.push(p)
      evenFlux.push(1 - dip + (Math.random() - 0.5) * 0.0007)
    }

    const demoResult: AnalysisResult = {
      target: 'TOI-270 (Interactive Demo)',
      predictions: [
        {
          model_name: 'Calibrated RF+XGBoost Ensemble (v3)',
          prediction: 1,
          probability: 0.942,
          confidence: 'High',
          result_text: 'Planet Candidate Detected'
        }
      ],
      features: {
        'period_days': 5.6604,
        'depth_ppm': 3732.4,
        'duration_hrs': 1.29,
        'model_snr': 41.7,
        'rp_rs': 0.0549,
        'log10_depth': 3.572,
        'log10_period': 0.753,
        'duration_over_period': 0.0095
      },
      explanation: [
        'Data source: TESS SPOC 2-minute cadence lightcurve (3 sectors merged, 48,210 data points).',
        'Best periodic signal: P = 5.6604 d, depth = 3732 ppm, duration = 1.29 h, Rp/Rs = 0.0549.',
        'Signal strength: SDE = 39.9 ≥ 7 — the periodic dip is statistically significant, not a noise fluctuation.',
        'Implied companion radius: 0.20 R_Jup (within the planetary regime).',
        'Odd/even test: alternating transits have consistent depths (Welch p = 0.842) — no sign of an eclipsing binary at twice the period.',
        'Duration check: transit lasts 0.92× the circular-orbit maximum — physically plausible.',
        'Density check: transit-implied stellar density is 1.14× the catalog value — consistent with target star.',
        'Secondary eclipse: none found at phase 0.5 (S/N = 0.8 < 3) — no sign of a self-luminous companion.',
        'ML classifier: calibrated ensemble assigns a 94.2% planet probability (High confidence).',
        'VERDICT: PLANET CANDIDATE — all 5 physics checks pass and the ML probability is 94.2%.'
      ],
      data_source: 'SPOC',
      n_sectors: 3,
      sde: 39.9,
      sde_pass: true,
      welch_p: 0.842,
      duration_ok: true,
      duration_ratio: 0.92,
      density_ok: true,
      density_ratio: 1.14,
      has_secondary: false,
      secondary_depth: 0.00001,
      secondary_snr: 0.8,
      stellar_r: 0.38,
      stellar_m: 0.39,
      plot_data: {
        time: Array.from({ length: 200 }, (_, i) => 1500 + i * 0.1),
        raw_flux: Array.from({ length: 200 }, () => 1 + (Math.random() * 0.01 - 0.005)),
        flat_flux: Array.from({ length: 200 }, (_, i) => (i % 40 > 18 && i % 40 < 22) ? 0.9962 : 1 + (Math.random() * 0.002 - 0.001)),
        trend: Array.from({ length: 200 }, () => 1)
      },
      folded_data: {
        phase: phasePoints,
        flux: fluxPoints,
        model_phase: modelPhasePoints,
        model_flux: modelFluxPoints
      },
      periodogram_data: {
        periods: periodsArr,
        sde: sdeArr
      },
      odd_even_data: {
        odd_phase: oddPhase,
        odd_flux: oddFlux,
        even_phase: evenPhase,
        even_flux: evenFlux
      }
    }

    setResult(demoResult)
    setTargetStar('TOI-270')
    setActiveTab('folded')
    setError('')
  }

  // Calculate planetary radius in Jupiter radii (R_Jup)
  const stellarRadius = result?.stellar_r ?? 1.0
  const rpRs = result?.features?.['rp_rs'] ?? 0
  const rpRjup = (rpRs * stellarRadius / 0.1028).toFixed(2)

  // Pipeline verdict evaluation
  const pred = result?.predictions?.[0]
  const isPlanetCandidate = pred?.prediction === 1 && result?.sde_pass && result?.duration_ok && result?.density_ok && !result?.has_secondary

  // --- CHART BUILDERS ---

  // 1. Phase-Folded Transit Chart
  const buildFoldedChartData = () => {
    if (!result?.folded_data) return { datasets: [] }
    const { phase, flux, model_phase, model_flux } = result.folded_data

    const datasets: any[] = [
      {
        label: 'Observed Flux (Phase Folded)',
        data: phase.map((p, i) => ({ x: p, y: flux[i] })),
        backgroundColor: 'rgba(56, 189, 248, 0.4)',
        borderColor: 'transparent',
        pointRadius: 2.5,
        pointHoverRadius: 5,
        type: 'scatter'
      }
    ]

    if (model_phase.length > 0 && model_flux.length > 0) {
      datasets.push({
        label: 'TLS Physical Transit Model Fit',
        data: model_phase.map((p, i) => ({ x: p, y: model_flux[i] })),
        borderColor: '#38bdf8',
        borderWidth: 3,
        pointRadius: 0,
        fill: false,
        type: 'line',
        tension: 0.1
      })
    }

    return { datasets }
  }

  // 2. Time Series Chart
  const buildTimeSeriesChartData = () => {
    if (!result?.plot_data) return { datasets: [] }
    const { time, flat_flux, raw_flux } = result.plot_data

    return {
      datasets: [
        {
          label: 'Wotan Flat Flux',
          data: time.map((t, i) => ({ x: t, y: flat_flux[i] })),
          backgroundColor: 'rgba(129, 140, 248, 0.5)',
          pointRadius: 2,
          type: 'scatter'
        },
        {
          label: 'Raw Flux Baseline',
          data: time.map((t, i) => ({ x: t, y: raw_flux[i] })),
          backgroundColor: 'rgba(148, 163, 184, 0.25)',
          pointRadius: 1.5,
          type: 'scatter'
        }
      ]
    }
  }

  // 3. Periodogram Chart
  const buildPeriodogramChartData = () => {
    if (!result?.periodogram_data) return { datasets: [] }
    const { periods, sde } = result.periodogram_data

    return {
      labels: periods.map(p => p.toFixed(2)),
      datasets: [
        {
          label: 'SDE Power',
          data: sde,
          borderColor: '#818cf8',
          backgroundColor: 'rgba(129, 140, 248, 0.15)',
          fill: true,
          borderWidth: 2,
          pointRadius: 0,
          type: 'line' as const
        }
      ]
    }
  }

  // 4. Odd vs Even Chart
  const buildOddEvenChartData = () => {
    if (!result?.odd_even_data) return { datasets: [] }
    const { odd_phase, odd_flux, even_phase, even_flux } = result.odd_even_data

    return {
      datasets: [
        {
          label: 'Odd Transits',
          data: odd_phase.map((p, i) => ({ x: p, y: odd_flux[i] })),
          backgroundColor: 'rgba(129, 140, 248, 0.6)',
          pointRadius: 3,
          type: 'scatter'
        },
        {
          label: 'Even Transits',
          data: even_phase.map((p, i) => ({ x: p, y: even_flux[i] })),
          backgroundColor: 'rgba(52, 211, 153, 0.6)',
          pointRadius: 3,
          type: 'scatter'
        }
      ]
    }
  }

  // 5. Feature Bar Chart
  const buildFeatureChartData = () => {
    if (!result?.features) return { labels: [], datasets: [] }
    const f = result.features
    return {
      labels: ['Period (d)', 'Depth (ppm / 100)', 'Duration (h)', 'Signal SNR', 'Rp/Rs (% x 10)'],
      datasets: [
        {
          label: 'Candidate Feature Values',
          data: [
            f.period_days || 0,
            (f.depth_ppm || 0) / 100,
            f.duration_hrs || 0,
            f.model_snr || 0,
            (f.rp_rs || 0) * 1000
          ],
          backgroundColor: [
            'rgba(56, 189, 248, 0.7)',
            'rgba(129, 140, 248, 0.7)',
            'rgba(52, 211, 153, 0.7)',
            'rgba(251, 191, 36, 0.7)',
            'rgba(244, 63, 94, 0.7)'
          ],
          borderWidth: 1
        }
      ]
    }
  }

  const chartOptionsScatter = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: { color: '#cbd5e1', font: { family: 'inherit', size: 12 } }
      },
      tooltip: {
        backgroundColor: '#0f172a',
        titleColor: '#38bdf8',
        bodyColor: '#e2e8f0',
        borderColor: 'rgba(56, 189, 248, 0.3)',
        borderWidth: 1
      }
    },
    scales: {
      x: {
        grid: { color: 'rgba(255, 255, 255, 0.05)' },
        ticks: { color: '#94a3b8' },
        title: { display: true, text: 'Phase / Time', color: '#94a3b8' }
      },
      y: {
        grid: { color: 'rgba(255, 255, 255, 0.05)' },
        ticks: { color: '#94a3b8' },
        title: { display: true, text: 'Relative Flux', color: '#94a3b8' }
      }
    }
  }

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 pb-16">
      {/* Top Glass Navigation Bar */}
      <header className="sticky top-0 z-30 glass-panel border-b border-slate-800/80 px-6 py-4 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-cyan-500 to-indigo-600 flex items-center justify-center text-xl shadow-lg shadow-cyan-500/20">
            🪐
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
              Exoplanet AI Control Center
              <span className="text-xs px-2.5 py-0.5 rounded-full font-mono bg-cyan-950 text-cyan-400 border border-cyan-800">
                v3.0 Live Engine
              </span>
            </h1>
            <p className="text-xs text-slate-400">
              Live TESS SPOC / TESScut Lightcurve Fetching $\rightarrow$ Wotan Detrending $\rightarrow$ TLS Transit Search $\rightarrow$ Physics Vetting $\rightarrow$ ML Ensemble
            </p>
          </div>
        </div>

        {/* System Health Status Indicator */}
        <div className="flex items-center gap-3 text-xs">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900/90 border border-slate-800">
            <span className={`h-2 w-2 rounded-full ${isApiConnected ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`}></span>
            <span className="text-slate-300 font-medium">
              {isApiConnected ? 'Backend Connected' : 'Standalone / Demo Mode'}
            </span>
          </div>
          <button
            onClick={loadDemoData}
            className="px-3.5 py-1.5 rounded-lg bg-indigo-900/60 hover:bg-indigo-800/80 text-indigo-200 border border-indigo-700/50 font-medium transition shadow-sm"
          >
            ⚡ Load Interactive Demo
          </button>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 mt-6 space-y-6">

        {/* Input & Target Preset Selector Panel */}
        <section className="glass-panel p-6 rounded-2xl shadow-xl">
          <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-6">
            <div className="flex-1 space-y-3">
              <label className="block text-xs font-semibold uppercase tracking-wider text-cyan-400">
                Target Star Identifier (TIC / TOI / Kepler ID)
              </label>
              <div className="flex items-center gap-3">
                <input
                  type="text"
                  value={targetStar}
                  onChange={(e) => setTargetStar(e.target.value)}
                  placeholder="e.g. TOI-270, TIC 307210830..."
                  className="flex-1 px-4 py-3 bg-slate-900 border border-slate-700/80 rounded-xl text-white font-mono placeholder:text-slate-500 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 transition shadow-inner"
                />
                <div className="flex items-center gap-2 px-4 py-3 bg-slate-900 border border-slate-700/80 rounded-xl">
                  <span className="text-xs text-slate-400 whitespace-nowrap">Sectors:</span>
                  <select
                    value={maxSectors}
                    onChange={(e) => setMaxSectors(Number(e.target.value))}
                    className="bg-transparent text-white font-mono text-sm focus:outline-none"
                  >
                    <option value={1} className="bg-slate-900">1 Sector</option>
                    <option value={2} className="bg-slate-900">2 Sectors</option>
                    <option value={3} className="bg-slate-900">3 Sectors (Default)</option>
                    <option value={5} className="bg-slate-900">5 Sectors</option>
                  </select>
                </div>
              </div>
            </div>

            <button
              onClick={() => handleAnalyze()}
              disabled={loading || !targetStar}
              className="px-8 py-3.5 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 disabled:opacity-50 text-white font-semibold shadow-lg shadow-cyan-500/25 flex items-center justify-center gap-2 transition active:scale-[0.98]"
            >
              {loading ? (
                <>
                  <span className="h-4 w-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></span>
                  Analyzing System...
                </>
              ) : (
                <>🚀 Run Live Prediction & Analysis</>
              )}
            </button>
          </div>

          {/* Preset Buttons */}
          <div className="mt-4 pt-4 border-t border-slate-800/60 flex flex-wrap items-center gap-2">
            <span className="text-xs text-slate-400 font-medium mr-2">Benchmark Targets:</span>
            {TARGET_PRESETS.map((preset) => (
              <button
                key={preset.name}
                onClick={() => {
                  setTargetStar(preset.name)
                  handleAnalyze(preset.name)
                }}
                disabled={loading}
                className="px-3 py-1.5 rounded-lg bg-slate-900/80 hover:bg-slate-800 border border-slate-700/60 text-xs text-slate-300 flex items-center gap-2 transition"
              >
                <span className="font-semibold text-white">{preset.name}</span>
                <span className="text-[10px] text-cyan-400 font-mono">[{preset.tag}]</span>
              </button>
            ))}
          </div>
        </section>

        {/* Live Execution Progress Tracker */}
        {loading && (
          <div className="glass-panel p-6 rounded-2xl border-cyan-500/40 space-y-4 animate-pulse">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold uppercase tracking-wider text-cyan-400 flex items-center gap-2">
                <span className="h-2 w-2 rounded-full bg-cyan-400 animate-ping"></span>
                Live Pipeline Execution in Progress...
              </span>
              <span className="font-mono text-slate-400">Step {loadingStep} of 5</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-5 gap-2">
              <div className={`p-3 rounded-xl border text-xs font-mono transition-all ${loadingStep >= 1 ? 'bg-cyan-950/70 border-cyan-500/60 text-cyan-300' : 'bg-slate-900/50 border-slate-800 text-slate-500'}`}>
                1. MAST TESS Fetch
              </div>
              <div className={`p-3 rounded-xl border text-xs font-mono transition-all ${loadingStep >= 2 ? 'bg-cyan-950/70 border-cyan-500/60 text-cyan-300' : 'bg-slate-900/50 border-slate-800 text-slate-500'}`}>
                2. Wotan Detrending
              </div>
              <div className={`p-3 rounded-xl border text-xs font-mono transition-all ${loadingStep >= 3 ? 'bg-cyan-950/70 border-cyan-500/60 text-cyan-300' : 'bg-slate-900/50 border-slate-800 text-slate-500'}`}>
                3. TLS Search
              </div>
              <div className={`p-3 rounded-xl border text-xs font-mono transition-all ${loadingStep >= 4 ? 'bg-cyan-950/70 border-cyan-500/60 text-cyan-300' : 'bg-slate-900/50 border-slate-800 text-slate-500'}`}>
                4. Physics Vetting
              </div>
              <div className={`p-3 rounded-xl border text-xs font-mono transition-all ${loadingStep >= 5 ? 'bg-cyan-950/70 border-cyan-500/60 text-cyan-300' : 'bg-slate-900/50 border-slate-800 text-slate-500'}`}>
                5. ML Ensemble
              </div>
            </div>
          </div>
        )}

        {/* Error Alert */}
        {error && (
          <div className="p-4 rounded-xl bg-rose-950/80 border border-rose-800 text-rose-200 text-sm flex items-center justify-between">
            <div>
              <p className="font-semibold">Analysis Error</p>
              <p className="text-xs text-rose-300">{error}</p>
            </div>
            <button
              onClick={loadDemoData}
              className="px-3 py-1.5 rounded-lg bg-rose-900 hover:bg-rose-800 text-xs font-medium"
            >
              Switch to Demo Mode
            </button>
          </div>
        )}

        {/* MAIN RESULTS DASHBOARD */}
        {result && (
          <div className="space-y-6">

            {/* Executive Verdict Banner */}
            <div className={`p-6 rounded-2xl border backdrop-blur-md shadow-2xl ${
              isPlanetCandidate
                ? 'bg-emerald-950/40 border-emerald-500/40 shadow-emerald-500/10'
                : result.sde_pass
                ? 'bg-amber-950/40 border-amber-500/40 shadow-amber-500/10'
                : 'bg-slate-900/80 border-slate-800'
            }`}>
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
                <div>
                  <div className="flex items-center gap-3">
                    <span className={`px-3.5 py-1 rounded-full text-xs font-bold uppercase tracking-wider ${
                      isPlanetCandidate
                        ? 'bg-emerald-500 text-slate-950 shadow-lg shadow-emerald-500/30'
                        : result.sde_pass
                        ? 'bg-amber-500 text-slate-950'
                        : 'bg-slate-700 text-slate-200'
                    }`}>
                      {pred?.result_text || 'Analysis Complete'}
                    </span>
                    <span className="text-xs font-mono text-slate-400">Target: {result.target}</span>
                  </div>

                  <h2 className="text-2xl font-extrabold text-white mt-2">
                    {isPlanetCandidate
                      ? 'High-Confidence Exoplanet Candidate Detected'
                      : result.sde_pass
                      ? 'Transit Signal Detected — Subject to Physics Vetting'
                      : 'No Statistically Significant Transit Signal'}
                  </h2>
                  <p className="text-xs text-slate-300 mt-1 max-w-3xl">
                    Analyzed using TESS {result.data_source || 'SPOC'} data across {result.n_sectors || 3} sector(s). Signal Efficiency SDE = {result.sde?.toFixed(1) || '0.0'}.
                  </p>
                </div>

                {/* Score & Confidence Metric Box */}
                <div className="flex items-center gap-4 bg-slate-950/60 p-4 rounded-xl border border-slate-800">
                  <div className="text-right">
                    <div className="text-xs text-slate-400 uppercase font-semibold">Planet Probability</div>
                    <div className="text-3xl font-black font-mono text-cyan-400 glow-cyan">
                      {((pred?.probability || 0) * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="h-10 w-px bg-slate-800"></div>
                  <div>
                    <div className="text-xs text-slate-400 uppercase font-semibold">Confidence</div>
                    <div className="text-sm font-bold text-white">{pred?.confidence || 'N/A'}</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Key Physical Metrics Cards Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
              <div className="glass-panel p-4 rounded-xl space-y-1">
                <div className="text-[10px] font-semibold text-slate-400 uppercase">Companion Radius</div>
                <div className="text-lg font-bold text-cyan-400 font-mono">{rpRjup} R<sub>Jup</sub></div>
                <div className="text-[10px] text-slate-400">Ratio Rp/Rs: {(result.features?.['rp_rs'] || 0).toFixed(4)}</div>
              </div>

              <div className="glass-panel p-4 rounded-xl space-y-1">
                <div className="text-[10px] font-semibold text-slate-400 uppercase">Orbital Period</div>
                <div className="text-lg font-bold text-indigo-400 font-mono">{(result.features?.['period_days'] || 0).toFixed(4)} d</div>
                <div className="text-[10px] text-slate-400">TLS Best Period</div>
              </div>

              <div className="glass-panel p-4 rounded-xl space-y-1">
                <div className="text-[10px] font-semibold text-slate-400 uppercase">Transit Depth</div>
                <div className="text-lg font-bold text-emerald-400 font-mono">{(result.features?.['depth_ppm'] || 0).toFixed(0)} ppm</div>
                <div className="text-[10px] text-slate-400">Dip: {((result.features?.['depth_ppm'] || 0) / 10000).toFixed(3)}%</div>
              </div>

              <div className="glass-panel p-4 rounded-xl space-y-1">
                <div className="text-[10px] font-semibold text-slate-400 uppercase">Transit Duration</div>
                <div className="text-lg font-bold text-amber-400 font-mono">{(result.features?.['duration_hrs'] || 0).toFixed(2)} h</div>
                <div className="text-[10px] text-slate-400">Ratio: {result.duration_ratio?.toFixed(2) || 1}x circular</div>
              </div>

              <div className="glass-panel p-4 rounded-xl space-y-1">
                <div className="text-[10px] font-semibold text-slate-400 uppercase">Signal Efficiency</div>
                <div className="text-lg font-bold text-purple-400 font-mono">SDE {result.sde?.toFixed(1) || 0}</div>
                <div className="text-[10px] text-slate-400">Threshold: $\ge$ 7.0</div>
              </div>

              <div className="glass-panel p-4 rounded-xl space-y-1">
                <div className="text-[10px] font-semibold text-slate-400 uppercase">Host Star</div>
                <div className="text-lg font-bold text-slate-200 font-mono">{result.stellar_r || 1.0} R<sub>$\odot$</sub></div>
                <div className="text-[10px] text-slate-400">Mass: {result.stellar_m || 1.0} M<sub>$\odot$</sub></div>
              </div>
            </div>

            {/* LIVE INTERACTIVE PLOTTING DASHBOARD */}
            <div className="glass-panel rounded-2xl p-6 shadow-2xl space-y-6">
              
              {/* Tab Selector Controls */}
              <div className="flex flex-wrap items-center justify-between border-b border-slate-800 pb-4 gap-4">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-bold text-white mr-2">Plotting Views:</span>
                  <button
                    onClick={() => setActiveTab('folded')}
                    className={`px-4 py-2 rounded-xl text-xs font-semibold transition flex items-center gap-2 ${
                      activeTab === 'folded'
                        ? 'bg-cyan-500 text-slate-950 shadow-md shadow-cyan-500/20'
                        : 'bg-slate-900 text-slate-300 hover:bg-slate-800'
                    }`}
                  >
                    🪐 Phase-Folded Transit Fit
                  </button>
                  <button
                    onClick={() => setActiveTab('time')}
                    className={`px-4 py-2 rounded-xl text-xs font-semibold transition flex items-center gap-2 ${
                      activeTab === 'time'
                        ? 'bg-cyan-500 text-slate-950 shadow-md shadow-cyan-500/20'
                        : 'bg-slate-900 text-slate-300 hover:bg-slate-800'
                    }`}
                  >
                    🌌 Full Light Curve
                  </button>
                  <button
                    onClick={() => setActiveTab('periodogram')}
                    className={`px-4 py-2 rounded-xl text-xs font-semibold transition flex items-center gap-2 ${
                      activeTab === 'periodogram'
                        ? 'bg-cyan-500 text-slate-950 shadow-md shadow-cyan-500/20'
                        : 'bg-slate-900 text-slate-300 hover:bg-slate-800'
                    }`}
                  >
                    📈 TLS Periodogram
                  </button>
                  <button
                    onClick={() => setActiveTab('oddeven')}
                    className={`px-4 py-2 rounded-xl text-xs font-semibold transition flex items-center gap-2 ${
                      activeTab === 'oddeven'
                        ? 'bg-cyan-500 text-slate-950 shadow-md shadow-cyan-500/20'
                        : 'bg-slate-900 text-slate-300 hover:bg-slate-800'
                    }`}
                  >
                    ⚖️ Odd vs Even Transits
                  </button>
                  <button
                    onClick={() => setActiveTab('features')}
                    className={`px-4 py-2 rounded-xl text-xs font-semibold transition flex items-center gap-2 ${
                      activeTab === 'features'
                        ? 'bg-cyan-500 text-slate-950 shadow-md shadow-cyan-500/20'
                        : 'bg-slate-900 text-slate-300 hover:bg-slate-800'
                    }`}
                  >
                    📊 Feature Breakdown
                  </button>
                </div>

                <div className="text-xs text-slate-400 font-mono">
                  {activeTab === 'folded' && 'Phased on P = ' + (result.features?.['period_days']?.toFixed(4) || '0') + ' days'}
                  {activeTab === 'time' && (result.plot_data?.time?.length || 0) + ' downsampled data points'}
                  {activeTab === 'periodogram' && 'Tested period range: 0.5 to 15.0 days'}
                </div>
              </div>

              {/* Chart Render Window */}
              <div className="h-[420px] w-full relative">
                {activeTab === 'folded' && (
                  <Scatter data={buildFoldedChartData()} options={chartOptionsScatter} />
                )}
                {activeTab === 'time' && (
                  <Scatter data={buildTimeSeriesChartData()} options={chartOptionsScatter} />
                )}
                {activeTab === 'periodogram' && (
                  <Line data={buildPeriodogramChartData()} options={chartOptionsScatter} />
                )}
                {activeTab === 'oddeven' && (
                  <Scatter data={buildOddEvenChartData()} options={chartOptionsScatter} />
                )}
                {activeTab === 'features' && (
                  <Bar
                    data={buildFeatureChartData()}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: {
                        legend: { display: false }
                      },
                      scales: {
                        x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#cbd5e1' } },
                        y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } }
                      }
                    }}
                  />
                )}
              </div>
            </div>

            {/* Bottom Grid: Physics Vetting Audit Matrix & Science Report */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

              {/* 5-Point Physics Vetting Checklist */}
              <div className="glass-panel p-6 rounded-2xl space-y-4">
                <h3 className="text-base font-bold text-white flex items-center justify-between border-b border-slate-800 pb-3">
                  <span>⚛️ 5-Point Physics Vetting Audit</span>
                  <span className="text-xs text-slate-400 font-mono">Deterministic Diagnostics</span>
                </h3>

                <div className="space-y-3">
                  {/* Check 1: SDE Signal Strength */}
                  <div className="flex items-center justify-between p-3 rounded-xl bg-slate-900/60 border border-slate-800">
                    <div>
                      <div className="text-xs font-semibold text-white">1. Signal Significance (TLS SDE)</div>
                      <div className="text-[11px] text-slate-400">Threshold: SDE $\ge$ 7.0 (Measured: {result.sde?.toFixed(1)})</div>
                    </div>
                    <span className={`px-2.5 py-1 rounded-md text-xs font-bold ${result.sde_pass ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40' : 'bg-rose-500/20 text-rose-300 border border-rose-500/40'}`}>
                      {result.sde_pass ? 'PASS' : 'FAIL'}
                    </span>
                  </div>

                  {/* Check 2: Odd / Even Transit Symmetry */}
                  <div className="flex items-center justify-between p-3 rounded-xl bg-slate-900/60 border border-slate-800">
                    <div>
                      <div className="text-xs font-semibold text-white">2. Odd/Even Depth Consistency</div>
                      <div className="text-[11px] text-slate-400">Welch p = {result.welch_p?.toFixed(3) || '1.0'} (Eclipsing binary check)</div>
                    </div>
                    <span className={`px-2.5 py-1 rounded-md text-xs font-bold ${result.welch_p && result.welch_p >= 0.01 ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40' : 'bg-amber-500/20 text-amber-300 border border-amber-500/40'}`}>
                      {result.welch_p && result.welch_p >= 0.01 ? 'PASS' : 'WARN'}
                    </span>
                  </div>

                  {/* Check 3: Transit Duration Plausibility */}
                  <div className="flex items-center justify-between p-3 rounded-xl bg-slate-900/60 border border-slate-800">
                    <div>
                      <div className="text-xs font-semibold text-white">3. Transit Duration Bounds</div>
                      <div className="text-[11px] text-slate-400">Measured vs circular orbit maximum: {result.duration_ratio?.toFixed(2) || '1'}x</div>
                    </div>
                    <span className={`px-2.5 py-1 rounded-md text-xs font-bold ${result.duration_ok ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40' : 'bg-rose-500/20 text-rose-300 border border-rose-500/40'}`}>
                      {result.duration_ok ? 'PASS' : 'FAIL'}
                    </span>
                  </div>

                  {/* Check 4: Stellar Density Consistency */}
                  <div className="flex items-center justify-between p-3 rounded-xl bg-slate-900/60 border border-slate-800">
                    <div>
                      <div className="text-xs font-semibold text-white">4. Stellar Mean Density Check</div>
                      <div className="text-[11px] text-slate-400">Transit-implied density vs catalog: {result.density_ratio?.toFixed(2) || '1'}x</div>
                    </div>
                    <span className={`px-2.5 py-1 rounded-md text-xs font-bold ${result.density_ok ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40' : 'bg-rose-500/20 text-rose-300 border border-rose-500/40'}`}>
                      {result.density_ok ? 'PASS' : 'FAIL'}
                    </span>
                  </div>

                  {/* Check 5: Secondary Eclipse Search */}
                  <div className="flex items-center justify-between p-3 rounded-xl bg-slate-900/60 border border-slate-800">
                    <div>
                      <div className="text-xs font-semibold text-white">5. Secondary Eclipse Check</div>
                      <div className="text-[11px] text-slate-400">Phase 0.5 depth check (S/N = {result.secondary_snr?.toFixed(1) || '0.0'})</div>
                    </div>
                    <span className={`px-2.5 py-1 rounded-md text-xs font-bold ${!result.has_secondary ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40' : 'bg-rose-500/20 text-rose-300 border border-rose-500/40'}`}>
                      {!result.has_secondary ? 'PASS (No Companion)' : 'FAIL (Self-Luminous)'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Natural Language Scientific Report Panel */}
              <div className="glass-panel p-6 rounded-2xl space-y-4">
                <h3 className="text-base font-bold text-white border-b border-slate-800 pb-3">
                  📝 Automated Science Analysis Report
                </h3>
                <div className="space-y-2 font-mono text-xs text-slate-300 overflow-y-auto max-h-[300px] pr-2">
                  {result.explanation && result.explanation.length > 0 ? (
                    result.explanation.map((line, idx) => (
                      <div key={idx} className="p-2.5 rounded-lg bg-slate-900/50 border border-slate-800/80 leading-relaxed">
                        • {line}
                      </div>
                    ))
                  ) : (
                    <p className="text-slate-500 italic">No detailed explanation generated.</p>
                  )}
                </div>
              </div>

            </div>

          </div>
        )}

      </div>
    </main>
  )
}