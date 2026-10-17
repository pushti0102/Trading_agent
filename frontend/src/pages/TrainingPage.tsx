import React, { useEffect, useState, useRef } from 'react';
import {
  Cpu,
  Play,
  Square,
  CheckCircle2,
  AlertCircle,
  Activity,
  Layers,
  Terminal,
  Zap,
  TrendingUp,
  ShieldAlert,
  ArrowRight,
  Database,
} from 'lucide-react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
} from 'recharts';
import { api, TrainingStatusResponse } from '../services/api';
import { Badge } from '../components/Badge';
import { MetricCard } from '../components/MetricCard';
import { LoadingState } from '../components/LoadingState';

interface TrainingPageProps {
  onNavigate: (tab: any) => void;
}

export const TrainingPage: React.FC<TrainingPageProps> = ({ onNavigate }) => {
  const [data, setData] = useState<TrainingStatusResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [episodesCount, setEpisodesCount] = useState<number>(10);
  const [isStarting, setIsStarting] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const pollTimerRef = useRef<any>(null);

  useEffect(() => {
    loadStatus();
    // Poll status every 2 seconds if training is active
    pollTimerRef.current = setInterval(loadStatusSilently, 2500);
    return () => {
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    };
  }, []);

  const loadStatus = async () => {
    try {
      setLoading(true);
      setErrorMsg(null);
      const res = await api.getTrainingStatus();
      setData(res);
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to fetch training status');
    } finally {
      setLoading(false);
    }
  };

  const loadStatusSilently = async () => {
    try {
      const res = await api.getTrainingStatus();
      setData(res);
    } catch (err) {
      // silent
    }
  };

  const handleStart = async () => {
    try {
      setIsStarting(true);
      setErrorMsg(null);
      await api.startTraining(episodesCount);
      await loadStatusSilently();
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to start training');
    } finally {
      setIsStarting(false);
    }
  };

  const handleStop = async () => {
    try {
      await api.stopTraining();
      await loadStatusSilently();
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to stop training');
    }
  };

  if (loading) {
    return <LoadingState message="Checking RL training status..." subMessage="Reading training checkpoints and historical training logs" />;
  }

  const isRunning = data?.status === 'running';
  const episodes = data?.episodes || [];
  const model = data?.model;

  // Best training episode
  const bestEpisode = episodes.length > 0
    ? [...episodes].sort((a, b) => b.return_pct - a.return_pct)[0]
    : null;

  return (
    <div className="space-y-6">
      {/* Title & Hyperparameters Bar */}
      <div className="bg-ivory p-6 rounded-xl border border-borderBase shadow-card flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <h1 className="text-xl font-bold text-charcoal tracking-tight">DQN Reinforcement Learning Training</h1>
            <Badge variant={isRunning ? 'gain' : 'olive'}>
              {isRunning ? 'Training In Progress' : 'Model Ready'}
            </Badge>
          </div>
          <p className="text-xs text-charcoal-muted mt-1 leading-relaxed max-w-2xl">
            Train the Deep Q-Network agent on the designated training split (70% = 346 sessions, 2016-01-01 through 2017-05-26).
            The holdout test period is strictly held back from gradient updates to guarantee zero data leakage.
          </p>
        </div>

        {/* Action Controls */}
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2">
            <span className="text-xs font-semibold text-charcoal">Episodes:</span>
            <select
              value={episodesCount}
              onChange={(e) => setEpisodesCount(Number(e.target.value))}
              disabled={isRunning}
              className="px-2.5 py-1.5 text-xs font-mono font-semibold bg-cream-100 border border-borderBase rounded-lg text-charcoal focus:outline-none focus:border-olive-500 disabled:opacity-50"
            >
              <option value={5}>5 Episodes (~1.5 min)</option>
              <option value={10}>10 Episodes (~3 min)</option>
              <option value={15}>15 Episodes (~4.5 min)</option>
              <option value={20}>20 Episodes (~6 min)</option>
            </select>
          </div>

          {isRunning ? (
            <button
              onClick={handleStop}
              className="flex items-center space-x-1.5 px-4 py-2 bg-loss text-ivory hover:bg-red-700 rounded-lg text-xs font-bold shadow-sm transition-colors"
            >
              <Square className="w-3.5 h-3.5 fill-current" />
              <span>Stop Training</span>
            </button>
          ) : (
            <button
              onClick={handleStart}
              disabled={isStarting}
              className="flex items-center space-x-1.5 px-4 py-2 bg-olive-500 hover:bg-olive-600 text-ivory rounded-lg text-xs font-bold shadow-sm transition-colors disabled:opacity-50"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>{isStarting ? 'Starting...' : 'Start Training'}</span>
            </button>
          )}

          <button
            onClick={() => onNavigate('evaluation')}
            className="flex items-center space-x-1 px-3 py-2 bg-cream-200 hover:bg-olive-100 text-olive-800 border border-borderBase rounded-lg text-xs font-semibold transition-colors"
          >
            <span>Evaluate Model</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {errorMsg && (
        <div className="p-4 rounded-lg bg-loss-bg border border-loss-border text-xs text-loss flex items-center gap-2">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Live Training Status Card (Visible when running) */}
      {isRunning && data?.progress && (
        <div className="bg-olive-50 p-5 rounded-xl border border-olive-200 shadow-sm space-y-3">
          <div className="flex items-center justify-between text-xs">
            <span className="font-bold text-olive-900 flex items-center gap-2">
              <Activity className="w-4 h-4 text-olive-600 animate-spin" />
              {data.progress.message}
            </span>
            <span className="font-mono font-semibold text-olive-800">
              Episode {data.progress.current_episode} of {data.progress.total_episodes}
            </span>
          </div>

          <div className="w-full bg-olive-200 rounded-full h-2.5 overflow-hidden">
            <div
              className="bg-olive-600 h-full rounded-full transition-all duration-300"
              style={{
                width: `${Math.min(
                  100,
                  Math.round(
                    (data.progress.current_episode / Math.max(1, data.progress.total_episodes)) * 100
                  )
                )}%`,
              }}
            ></div>
          </div>
        </div>
      )}

      {/* Model Spec & Architecture Metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <MetricCard
          label="Learning Rate"
          value="0.001"
          subValue="Adam Optimizer"
        />
        <MetricCard
          label="Discount (γ)"
          value="0.95"
          subValue="Future reward weighting"
        />
        <MetricCard
          label="Exploration (ε)"
          value={episodes.length > 0 ? episodes[episodes.length - 1].epsilon.toFixed(3) : '0.050'}
          subValue="Epsilon decay 0.995"
        />
        <MetricCard
          label="Batch Size"
          value="64"
          subValue="Replay memory: 10,000"
        />
        <MetricCard
          label="Best Train Return"
          value={bestEpisode ? `${bestEpisode.return_pct >= 0 ? '+' : ''}${bestEpisode.return_pct.toFixed(2)}%` : '—'}
          subValue={bestEpisode ? `Episode ${bestEpisode.episode}` : '—'}
          trend={bestEpisode && bestEpisode.return_pct >= 0 ? 'up' : 'down'}
        />
        <MetricCard
          label="Model Checkpoint"
          value={model?.exists ? 'Saved' : 'Not Found'}
          subValue={model?.exists ? `${(model.size_bytes / 1024).toFixed(0)} KB • dqn_trader.pt` : 'Run train_rl.py'}
          badge={model?.exists ? 'active' : 'missing'}
        />
      </div>

      {/* Training Performance Charts: Return % Progression & Mean Loss Curve */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Return per Episode */}
        <div className="bg-ivory p-6 rounded-xl border border-borderBase shadow-card space-y-4">
          <div className="flex items-center justify-between border-b border-cream-200 pb-3">
            <div>
              <h3 className="text-sm font-bold text-charcoal">Episode Training Return (%)</h3>
              <p className="text-xs text-charcoal-muted mt-0.5">Portfolio return achieved across training episodes</p>
            </div>
            <Badge variant="olive">{episodes.length} Episodes</Badge>
          </div>

          <div className="h-[260px] w-full pt-2">
            {episodes.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={episodes} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E5E1D3" vertical={false} />
                  <XAxis dataKey="episode" tick={{ fontSize: 11, fill: '#77796D' }} tickLine={false} stroke="#E5E1D3" />
                  <YAxis tick={{ fontSize: 11, fill: '#77796D' }} tickLine={false} stroke="#E5E1D3" tickFormatter={(v) => `${v}%`} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#FFFDF7', borderColor: '#E5E1D3', borderRadius: '8px', fontSize: '12px' }}
                    formatter={(val: any) => [`${Number(val).toFixed(2)}%`, 'Return']}
                  />
                  <Line type="monotone" dataKey="return_pct" stroke="#606B3F" strokeWidth={2.2} dot={{ r: 4, fill: '#606B3F' }} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-xs text-charcoal-muted">
                No training logs available yet. Click Start Training above.
              </div>
            )}
          </div>
        </div>

        {/* Loss Curve */}
        <div className="bg-ivory p-6 rounded-xl border border-borderBase shadow-card space-y-4">
          <div className="flex items-center justify-between border-b border-cream-200 pb-3">
            <div>
              <h3 className="text-sm font-bold text-charcoal">Smooth L1 Loss Curve</h3>
              <p className="text-xs text-charcoal-muted mt-0.5">Mean TD error loss per training episode</p>
            </div>
            <Badge variant="sage">Convergence</Badge>
          </div>

          <div className="h-[260px] w-full pt-2">
            {episodes.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={episodes} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E5E1D3" vertical={false} />
                  <XAxis dataKey="episode" tick={{ fontSize: 11, fill: '#77796D' }} tickLine={false} stroke="#E5E1D3" />
                  <YAxis
                    tick={{ fontSize: 11, fill: '#77796D' }}
                    tickLine={false}
                    stroke="#E5E1D3"
                    tickFormatter={(v) => Number(v).toExponential(1)}
                  />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#FFFDF7', borderColor: '#E5E1D3', borderRadius: '8px', fontSize: '12px' }}
                    formatter={(val: any) => [Number(val).toExponential(4), 'Loss']}
                  />
                  <Line type="monotone" dataKey="mean_loss" stroke="#3F492C" strokeWidth={2.2} dot={{ r: 4, fill: '#3F492C' }} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-xs text-charcoal-muted">
                No loss data recorded yet.
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Episode Log Table and Real-time Console */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Episodes Table (7 Cols) */}
        <div className="lg:col-span-7 bg-ivory p-6 rounded-xl border border-borderBase shadow-card space-y-3">
          <div className="flex items-center justify-between border-b border-cream-200 pb-3">
            <div>
              <h3 className="text-sm font-bold text-charcoal">Training Episode History</h3>
              <p className="text-xs text-charcoal-muted mt-0.5">Outputs logged to rl_training_results.csv</p>
            </div>
          </div>

          <div className="overflow-x-auto max-h-[300px] overflow-y-auto">
            <table className="w-full text-left text-xs">
              <thead className="sticky top-0 bg-cream-100 border-b border-borderBase text-[11px] uppercase tracking-wider text-charcoal-muted font-semibold">
                <tr>
                  <th className="py-2.5 px-3">Ep</th>
                  <th className="py-2.5 px-3 text-right">Final Value</th>
                  <th className="py-2.5 px-3 text-right">Return %</th>
                  <th className="py-2.5 px-3 text-right">Reward</th>
                  <th className="py-2.5 px-3 text-right">Orders</th>
                  <th className="py-2.5 px-3 text-right">Loss</th>
                  <th className="py-2.5 px-3 text-right">ε</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-cream-200">
                {episodes.map((ep) => {
                  const isGain = ep.return_pct >= 0;
                  return (
                    <tr key={ep.episode} className="hover:bg-cream-50 transition-colors font-mono">
                      <td className="py-2.5 px-3 font-bold text-charcoal">#{ep.episode}</td>
                      <td className="py-2.5 px-3 text-right text-charcoal">
                        ₹{ep.final_portfolio_value.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                      </td>
                      <td className="py-2.5 px-3 text-right font-semibold">
                        <span className={isGain ? 'text-gain' : 'text-loss'}>
                          {isGain ? '+' : ''}{ep.return_pct.toFixed(2)}%
                        </span>
                      </td>
                      <td className="py-2.5 px-3 text-right text-charcoal-muted">{ep.total_reward.toFixed(3)}</td>
                      <td className="py-2.5 px-3 text-right text-charcoal">{ep.executed_orders}</td>
                      <td className="py-2.5 px-3 text-right text-charcoal-muted">{ep.mean_loss.toExponential(2)}</td>
                      <td className="py-2.5 px-3 text-right text-olive-800">{ep.epsilon.toFixed(3)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Live Logs Console (5 Cols) */}
        <div className="lg:col-span-5 bg-charcoal p-5 rounded-xl border border-charcoal text-ivory font-mono text-xs flex flex-col justify-between shadow-card">
          <div>
            <div className="flex items-center justify-between border-b border-charcoal-muted/30 pb-3 mb-3">
              <span className="flex items-center gap-2 font-bold text-cream-100 text-[11px] uppercase tracking-wider">
                <Terminal className="w-3.5 h-3.5 text-olive-400" />
                Live Training Terminal Output
              </span>
              <span className="text-[10px] text-cream-200/60 font-mono">train_rl.py</span>
            </div>

            <div className="space-y-1.5 max-h-[260px] overflow-y-auto text-[11px] text-cream-100/90 leading-relaxed pr-1">
              {data?.logs && data.logs.length > 0 ? (
                data.logs.map((log, idx) => (
                  <div key={idx} className="flex items-start gap-2">
                    <span className="text-olive-400 select-none">&gt;</span>
                    <span>{log}</span>
                  </div>
                ))
              ) : (
                <div className="text-cream-200/50 py-10 text-center text-xs">
                  Awaiting training session execution...
                </div>
              )}
            </div>
          </div>

          <div className="pt-3 border-t border-charcoal-muted/30 text-[10px] text-cream-200/60 flex items-center justify-between">
            <span>Target Network Sync: Every 100 updates</span>
            <span>Device: CPU</span>
          </div>
        </div>
      </div>
    </div>
  );
};
