'use client'

import { useState } from 'react'
import axios from 'axios'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js'
import { Scatter } from 'react-chartjs-2'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
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
  plot_data: {
    time: number[];
    raw_flux: number[];
    flat_flux: number[];
    trend: number[];
  };
}

export default function Home() {
  const [targetStar, setTargetStar] = useState('TOI-270')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [selectedModel, setSelectedModel] = useState<string | null>(null)

  const handleAnalyze = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!targetStar) return

    setLoading(true)
    setError('')
    setResult(null)
    setSelectedModel(null)

    try {
      const response = await axios.post<AnalysisResult>('http://localhost:8000/api/analyze', {
        target_star: targetStar
      })
      const data = response.data
      setResult(data)
      
      // Auto-select the model with highest confidence
      if (data.predictions && data.predictions.length > 0) {
        const bestModel = data.predictions.reduce((prev, current) => {
          return (Math.abs(current.probability - 0.5) > Math.abs(prev.probability - 0.5)) ? current : prev
        });
        setSelectedModel(bestModel.model_name);
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'An error occurred during analysis.')
    } finally {
      setLoading(false)
    }
  }

  const loadDummyData = () => {
    const dummyResult = {
      target: 'TOI-270 (Dummy Data)',
      predictions: [
        {
          model_name: 'Legacy Model (v1)',
          prediction: 0,
          probability: 0.23,
          confidence: 'Medium',
          result_text: 'No Planet Transit Detected'
        },
        {
          model_name: 'Advanced Model (v2)',
          prediction: 1,
          probability: 0.96,
          confidence: 'High',
          result_text: 'Planet Candidate Detected'
        }
      ],
      features: {
        'Period': 3.36,
        'Duration': 0.12,
        'Depth': 0.005,
        'SNR': 12.5,
        'SDE Pass': 1.0,
        'Rp/Rs': 0.04,
        'Odd-Even Mismatch': 0.01,
        'Symmetry': 0.002
      },
      plot_data: {
        time: Array.from({length: 100}, (_, i) => i * 0.1),
        raw_flux: Array.from({length: 100}, () => 1 + (Math.random() * 0.02 - 0.01)),
        flat_flux: Array.from({length: 100}, (_, i) => (i > 45 && i < 55) ? 0.98 + (Math.random() * 0.005) : 1 + (Math.random() * 0.005)),
        trend: Array.from({length: 100}, () => 1)
      }
    }
    
    setResult(dummyResult);
    
    // Auto-select the model with highest confidence
    const bestModel = dummyResult.predictions.reduce((prev, current) => {
      return (Math.abs(current.probability - 0.5) > Math.abs(prev.probability - 0.5)) ? current : prev
    });
    setSelectedModel(bestModel.model_name);
    
    setError('');
  };

  return (
    <main className="min-h-screen bg-slate-950 text-slate-200">
      {/* Header */}
      <header className="border-b border-indigo-900/50 bg-slate-950 px-6 py-4 sticky top-0 z-10 flex items-center justify-between shadow-md shadow-indigo-500/5">
        <div className="flex items-center gap-3">
          <div className="h-2 w-2 rounded-full bg-indigo-500 animate-pulse"></div>
          <h1 className="text-xl font-bold tracking-tight text-indigo-400 font-mono">Exoplanet Detection Core</h1>
        </div>
      </header>

      <div className="flex h-[calc(100vh-65px)] overflow-hidden">
        {/* Sidebar */}
        <aside className="w-80 bg-slate-900 border-r border-indigo-900/50 p-6 flex flex-col gap-6 overflow-y-auto">
          <div>
            <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4">Configuration</h2>
            <form onSubmit={handleAnalyze} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1">Target Star Designation</label>
                <input 
                  type="text" 
                  value={targetStar}
                  onChange={(e) => setTargetStar(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700/50 rounded-md px-3 py-2 text-sm focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-all placeholder:text-slate-600"
                  placeholder="e.g., TOI-270"
                />
              </div>
              <button 
                type="submit" 
                disabled={loading}
                className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-800 disabled:text-slate-600 text-white font-medium py-2 px-4 rounded-md shadow-md shadow-indigo-500/20 transition-all"
              >
                {loading ? 'Analyzing...' : 'Analyze Target'}
              </button>
              <button 
                type="button" 
                onClick={loadDummyData}
                disabled={loading}
                className="w-full bg-slate-800 hover:bg-slate-700 disabled:bg-slate-900 disabled:text-slate-600 text-slate-200 font-medium py-2 px-4 rounded-md border border-slate-700/50 transition-all mt-2"
              >
                Load Dummy Results
              </button>
            </form>
          </div>

          <div>
            <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Example Targets</h3>
            <ul className="text-sm space-y-2 text-slate-400">
              <li className="hover:text-indigo-400 cursor-pointer transition-colors" onClick={() => setTargetStar('TOI-270')}>TOI-270 (Planet Host)</li>
              <li className="hover:text-indigo-400 cursor-pointer transition-colors" onClick={() => setTargetStar('TIC 38846515')}>TIC 38846515 (Planet Host)</li>
              <li className="hover:text-indigo-400 cursor-pointer transition-colors" onClick={() => setTargetStar('Ross 176')}>Ross 176 (No Transits)</li>
            </ul>
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 p-8 overflow-y-auto bg-[#040814] relative">
          {/* Subtle Background Glow */}
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-indigo-900/10 blur-[120px] rounded-full pointer-events-none"></div>
          
          <div className="max-w-6xl mx-auto relative z-10">
            {error && (
              <div className="bg-red-950/40 border border-red-900/50 text-red-300 px-4 py-3 rounded-md mb-8 flex items-center gap-3 shadow-lg shadow-red-900/10">
                <svg className="w-5 h-5 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                {error}
              </div>
            )}

            {!result && !loading && !error && (
              <div className="h-full flex flex-col items-center justify-center text-slate-500 min-h-[60vh]">
                <div className="w-20 h-20 rounded-full flex items-center justify-center mb-6 relative">
                  <div className="absolute inset-0 border border-indigo-500/20 rounded-full animate-[spin_4s_linear_infinite]"></div>
                  <div className="absolute inset-2 border border-blue-500/10 rounded-full animate-[spin_3s_linear_infinite_reverse]"></div>
                  <svg className="w-8 h-8 text-indigo-400/50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <p className="text-slate-400 text-lg">Initialize detection sequence.</p>
                <p className="text-slate-600 text-sm mt-2">Awaiting target parameters.</p>
              </div>
            )}

            {loading && (
              <div className="h-full flex flex-col items-center justify-center min-h-[60vh]">
                <div className="relative w-24 h-24 mb-6">
                  <div className="absolute inset-0 rounded-full border-t-2 border-indigo-500 animate-spin"></div>
                  <div className="absolute inset-2 rounded-full border-r-2 border-blue-400 animate-[spin_1.5s_linear_infinite_reverse]"></div>
                  <div className="absolute inset-4 rounded-full border-b-2 border-cyan-400 animate-[spin_2s_linear_infinite]"></div>
                </div>
                <p className="text-indigo-400 tracking-widest text-sm font-semibold uppercase animate-pulse">Processing Lightcurve Data...</p>
              </div>
            )}

            {result && !loading && (
              <div className="space-y-10 animate-in fade-in duration-500">
                <div className="flex items-end justify-between border-b border-indigo-900/40 pb-4">
                  <div>
                    <p className="text-indigo-500 text-sm font-semibold uppercase tracking-wider mb-1">Target Acquired</p>
                    <h2 className="text-3xl font-bold text-white tracking-tight">{result.target}</h2>
                  </div>
                </div>

                {/* Models Comparison Section */}
                <div className="space-y-5">
                  <div className="flex items-center gap-3">
                    <div className="h-4 w-1 bg-indigo-500 rounded-full"></div>
                    <h3 className="text-xl font-semibold text-slate-200">Prediction Engine Comparison</h3>
                    <span className="text-xs text-slate-500 ml-2 px-2 py-1 bg-slate-900 rounded border border-slate-800">Select model to focus details</span>
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {result.predictions.length === 0 && (
                      <div className="col-span-full text-slate-500 italic p-8 bg-slate-900/50 rounded-xl border border-slate-800/50 flex justify-center">No active models detected.</div>
                    )}
                    {result.predictions.map((pred, i) => {
                      const isSelected = selectedModel === pred.model_name;
                      const isPlanet = pred.prediction === 1;
                      
                      return (
                        <div 
                          key={i} 
                          onClick={() => setSelectedModel(pred.model_name)}
                          className={`
                            relative overflow-hidden cursor-pointer transition-all duration-300 p-6 rounded-xl border
                            ${isSelected 
                              ? isPlanet 
                                ? 'bg-emerald-950/20 border-emerald-500 ring-1 ring-emerald-500/50 shadow-lg shadow-emerald-500/10' 
                                : 'bg-indigo-950/30 border-indigo-500 ring-1 ring-indigo-500/50 shadow-lg shadow-indigo-500/10'
                              : 'bg-slate-900/60 border-slate-800 hover:border-slate-600 hover:bg-slate-800/60 opacity-70 hover:opacity-100 hover:shadow-md'
                            }
                          `}
                        >
                          {isSelected && (
                            <div className="absolute top-0 left-0 w-full h-[2px] bg-gradient-to-r from-transparent via-current to-transparent opacity-50" />
                          )}
                          
                          <div className="flex justify-between items-center mb-5 pb-3 border-b border-white/5">
                            <h4 className={`text-sm font-semibold tracking-wide ${isSelected ? 'text-white' : 'text-slate-400'}`}>
                              {pred.model_name}
                            </h4>
                            {isSelected && (
                              <svg className={`w-5 h-5 ${isPlanet ? 'text-emerald-400' : 'text-indigo-400'}`} fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                              </svg>
                            )}
                          </div>
                          
                          <div className="space-y-5">
                            <div>
                              <p className="text-[10px] text-slate-500 uppercase tracking-widest mb-1 font-semibold">Classification Result</p>
                              <p className={`text-lg font-bold tracking-tight ${isPlanet ? 'text-emerald-400' : 'text-slate-300'}`}>
                                {pred.result_text}
                              </p>
                            </div>
                            
                            <div className="flex items-end justify-between border-l-2 border-white/10 pl-4 py-1">
                              <div>
                                <p className="text-[10px] text-slate-500 uppercase tracking-widest mb-1 font-semibold">Probability</p>
                                <p className={`text-2xl font-black ${isSelected ? 'text-white' : 'text-slate-300'}`}>
                                  {(pred.probability * 100).toFixed(1)}%
                                </p>
                              </div>
                              <div className="text-right">
                                <p className="text-[10px] text-slate-500 uppercase tracking-widest mb-1 font-semibold">Confidence</p>
                                <p className={`text-sm font-bold tracking-wide ${
                                  pred.confidence === 'High' ? 'text-blue-400' : 
                                  pred.confidence === 'Medium' ? 'text-amber-400' : 'text-rose-400'
                                }`}>{pred.confidence}</p>
                              </div>
                            </div>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>

                {selectedModel && (
                  <div className="space-y-10 animate-in slide-in-from-bottom-4 duration-500 delay-100 fill-mode-both">
                    {/* Feature Extracted Metrics */}
                    <div className="space-y-5 pt-8">
                      <div className="flex items-center gap-3">
                        <div className="h-4 w-1 bg-blue-500 rounded-full"></div>
                        <h3 className="text-xl font-semibold text-slate-200">Extracted System Characteristics</h3>
                      </div>
                      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                        {Object.entries(result.features).slice(0, 8).map(([key, val], i) => (
                          <div key={i} className="bg-slate-900/80 p-5 rounded-xl border border-slate-800/80 hover:border-slate-700 hover:bg-slate-800/50 transition-colors group">
                            <p className="text-xs text-slate-500 tracking-wider uppercase mb-2 group-hover:text-indigo-400 transition-colors">{key}</p>
                            <p className="text-lg font-mono text-slate-200">{Number(val).toExponential(3)}</p>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Lightcurve Visualizations */}
                    <div className="space-y-5 pt-8">
                      <div className="flex items-center gap-3">
                        <div className="h-4 w-1 bg-cyan-500 rounded-full"></div>
                        <h3 className="text-xl font-semibold text-slate-200">De-trended Lightcurve Plot</h3>
                      </div>
                      <div className="bg-slate-900/80 border border-slate-800/80 p-6 rounded-2xl h-[450px] shadow-2xl shadow-black/50">
                        <Scatter 
                          options={{
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {
                              legend: { display: false },
                              tooltip: { 
                                enabled: true,
                                backgroundColor: '#0f172a',
                                titleColor: '#818cf8',
                                bodyColor: '#cbd5e1',
                                borderColor: '#1e293b',
                                borderWidth: 1,
                                padding: 12,
                                displayColors: false
                              }
                            },
                            elements: {
                              point: { radius: 2, hoverRadius: 5, backgroundColor: '#6366f1', borderWidth: 0 }
                            },
                            scales: {
                              x: { 
                                grid: { color: '#1e293b', tickColor: 'transparent' }, 
                                ticks: { color: '#64748b' },
                                title: { display: true, text: 'Time (Days)', color: '#94a3b8', font: { size: 12, weight: 'bold' } }
                              },
                              y: { 
                                grid: { color: '#1e293b', tickColor: 'transparent' },
                                ticks: { color: '#64748b' },
                                title: { display: true, text: 'Relative Flux', color: '#94a3b8', font: { size: 12, weight: 'bold' } }
                              }
                            }
                          }} 
                          data={{
                            datasets: [{
                              label: 'Relative Flux',
                              data: result.plot_data.time.map((t, i) => ({
                                x: t,
                                y: result.plot_data.flat_flux[i]
                              }))
                            }]
                          }} 
                        />
                      </div>
                    </div>
                  </div>
                )}

              </div>
            )}
          </div>
        </main>
      </div>
    </main>
  )
}