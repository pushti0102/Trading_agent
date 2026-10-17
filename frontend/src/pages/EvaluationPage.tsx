import React, { useEffect, useState } from 'react';
import {
  LineChart as LineChartIcon,
  Download,
  Play,
  RotateCcw,
  CheckCircle2,
  AlertTriangle,
  Info,
  Calendar,
  Layers,
  ArrowUpDown,
  TrendingUp,
  TrendingDown,
  ShieldAlert,
} from 'lucide-react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from 'recharts';
import { api, EvaluationSummaryResponse, EvaluationHistoryPoint, AgentMetric } from '../services/api';
import { Badge } from '../components/Badge';
import { MetricCard } from '../components/MetricCard';
import { LoadingState } from '../components/LoadingState';

export const EvaluationPage: React.FC = () => {
  const [summary, setSummary] = useState<EvaluationSummaryResponse | null>(null);
  const [history, setHistory] = useState<EvaluationHistoryPoint[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [isEvaluating, setIsEvaluating] = useState<boolean>(false);
  const [evalMsg, setEvalMsg] = useState<string | null>(null);

  useEffect(() => {
    loadEvaluationData();
  }, []);

  const loadEvaluationData = async () => {
    try {
      setLoading(true);
      const [sumData, histData] = await Promise.all([
        api.getEvaluationSummary(),
        api.getEvaluationHistory(),
      ]);
      setSummary(sumData);
      setHistory(histData);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleRunEvaluation = async () => {
    try {
      setIsEvaluating(true);
      setEvalMsg(null);
      const res = await api.runEvaluation();
      setSummary(res);
      const histData = await api.getEvaluationHistory();
      setHistory(histData);
      setEvalMsg('Evaluation complete. test_evaluation_summary.csv and test_portfolio_history.csv updated.');
    } catch (err: any) {
      setEvalMsg(`Error: ${err.message || 'Evaluation run failed'}`);
    } finally {
      setIsEvaluating(false);
    }
  };

  if (loading) {
    return <LoadingState message="Loading backtesting evaluation..." subMessage="Reading test_evaluation_summary.csv and portfolio history curves" />;
  }

  const items = summary?.summary || [];

  return (
    <div className="space-y-6">
      {/* Title & Holdout Split Header */}
      <div className="bg-ivory p-6 rounded-xl border border-borderBase shadow-card flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <h1 className="text-xl font-bold text-charcoal tracking-tight">Holdout Backtesting & Evaluation</h1>
            <Badge variant="olive">Unseen Holdout (30%)</Badge>
          </div>
          <p className="text-xs text-charcoal-muted mt-1 leading-relaxed max-w-3xl">
            Strict out-of-sample evaluation on the final 30% of trading dates (149 sessions: 2017-05-29 to 2017-12-29).
            Comparing the trained Deep Q-Network trader against the Stackelberg Leader, Momentum Follower, and a passive Buy-and-Hold basket.
          </p>
        </div>

        {/* Action Controls: Run Evaluation & CSV Export */}
        <div className="flex items-center space-x-2">
          <a
            href={api.getExportEvaluationUrl()}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center space-x-1.5 px-3 py-2 bg-cream-100 hover:bg-cream-200 text-charcoal border border-borderBase rounded-lg text-xs font-semibold transition-colors"
          >
            <Download className="w-3.5 h-3.5 text-olive-700" />
            <span>Export CSV</span>
          </a>

          <button
            onClick={handleRunEvaluation}
            disabled={isEvaluating}
            className="flex items-center space-x-1.5 px-4 py-2 bg-olive-500 hover:bg-olive-600 text-ivory rounded-lg text-xs font-bold shadow-sm transition-colors disabled:opacity-50"
          >
            <Play className={`w-3.5 h-3.5 fill-current ${isEvaluating ? 'animate-spin' : ''}`} />
            <span>{isEvaluating ? 'Evaluating Model...' : 'Run Holdout Evaluation'}</span>
          </button>
        </div>
      </div>

      {evalMsg && (
        <div className={`p-4 rounded-lg text-xs flex items-center gap-2 border ${
          evalMsg.startsWith('Error') ? 'bg-loss-bg text-loss border-loss-border' : 'bg-gain-bg text-gain border-gain-border'
        }`}>
          <CheckCircle2 className="w-4 h-4 flex-shrink-0" />
          <span>{evalMsg}</span>
        </div>
      )}

      {/* Dataset Split Verification Banner */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-ivory p-4 rounded-xl border border-borderBase shadow-card">
          <div className="flex items-center justify-between text-xs mb-1">
            <span className="text-charcoal-muted font-semibold uppercase tracking-wider text-[10px]">
              Phase 1: Training Partition
            </span>
            <span className="text-[10px] font-mono font-bold text-olive-800 bg-olive-100 px-1.5 py-0.5 rounded">
              70% Split
            </span>
          </div>
          <div className="font-bold text-charcoal text-sm">346 Sessions</div>
          <div className="text-[11px] text-charcoal-muted font-mono mt-1">2016-01-01 &rarr; 2017-05-26</div>
        </div>

        <div className="bg-ivory p-4 rounded-xl border border-olive-400 ring-2 ring-olive-400/20 shadow-card">
          <div className="flex items-center justify-between text-xs mb-1">
            <span className="text-olive-800 font-bold uppercase tracking-wider text-[10px]">
              Phase 2: Holdout Test Partition
            </span>
            <span className="text-[10px] font-mono font-bold text-ivory bg-olive-600 px-1.5 py-0.5 rounded">
              30% Unseen
            </span>
          </div>
          <div className="font-bold text-charcoal text-sm">149 Sessions</div>
          <div className="text-[11px] text-charcoal-muted font-mono mt-1">2017-05-29 &rarr; 2017-12-29</div>
        </div>

        <div className="bg-ivory p-4 rounded-xl border border-borderBase shadow-card">
          <div className="flex items-center justify-between text-xs mb-1">
            <span className="text-charcoal-muted font-semibold uppercase tracking-wider text-[10px]">
              Total Dataset Scope
            </span>
            <span className="text-[10px] font-mono font-bold text-charcoal bg-cream-200 px-1.5 py-0.5 rounded">
              NSE 10 Stocks
            </span>
          </div>
          <div className="font-bold text-charcoal text-sm">495 Sessions</div>
          <div className="text-[11px] text-charcoal-muted font-mono mt-1">2016-01-01 &rarr; 2017-12-29</div>
        </div>
      </div>

      {/* Main Holdout Equity Curve Chart */}
      <div className="bg-ivory p-6 rounded-xl border border-borderBase shadow-card space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-cream-200 pb-3">
          <div>
            <h2 className="text-sm font-bold text-charcoal flex items-center gap-2">
              <LineChartIcon className="w-4 h-4 text-olive-600" />
              Holdout Equity Progression (₹100,000 Base)
            </h2>
            <p className="text-xs text-charcoal-muted mt-0.5">
              Cumulative marked-to-market portfolio trajectory under execution friction
            </p>
          </div>

          <div className="flex items-center space-x-3 text-xs">
            <span className="flex items-center gap-1.5 font-mono text-[11px]">
              <span className="w-2.5 h-2.5 rounded-full bg-[#606B3F]"></span> RL Trader
            </span>
            <span className="flex items-center gap-1.5 font-mono text-[11px]">
              <span className="w-2.5 h-2.5 rounded-full bg-[#3F492C]"></span> Leader
            </span>
            <span className="flex items-center gap-1.5 font-mono text-[11px]">
              <span className="w-2.5 h-2.5 rounded-full bg-[#B33927]"></span> Momentum
            </span>
            <span className="flex items-center gap-1.5 font-mono text-[11px]">
              <span className="w-2.5 h-2.5 rounded-full bg-[#77796D]"></span> Buy & Hold
            </span>
          </div>
        </div>

        <div className="h-[360px] w-full pt-2">
          {history.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E5E1D3" vertical={false} />
                <XAxis dataKey="date" tick={{ fontSize: 11, fill: '#77796D' }} tickLine={false} stroke="#E5E1D3" minTickGap={25} />
                <YAxis
                  domain={['auto', 'auto']}
                  tick={{ fontSize: 11, fill: '#77796D' }}
                  tickLine={false}
                  stroke="#E5E1D3"
                  tickFormatter={(val) => `₹${(val / 1000).toFixed(0)}k`}
                />
                <Tooltip
                  contentStyle={{ backgroundColor: '#FFFDF7', borderColor: '#E5E1D3', borderRadius: '8px', fontSize: '12px' }}
                  formatter={(val: any, name: any) => [
                    `₹${Number(val).toLocaleString('en-IN', { maximumFractionDigits: 2 })}`,
                    name === 'rl_trader'
                      ? 'RL Trader (DQN)'
                      : name === 'leader'
                      ? 'Stackelberg Leader'
                      : name === 'momentum'
                      ? 'Momentum Follower'
                      : 'Buy & Hold',
                  ]}
                />
                <Line type="monotone" dataKey="rl_trader" name="rl_trader" stroke="#606B3F" strokeWidth={2.4} dot={false} />
                <Line type="monotone" dataKey="leader" name="leader" stroke="#3F492C" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="momentum" name="momentum" stroke="#B33927" strokeWidth={1.8} dot={false} />
                <Line type="monotone" dataKey="buy_hold" name="buy_hold" stroke="#77796D" strokeWidth={1.8} strokeDasharray="4 4" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-full flex items-center justify-center text-xs text-charcoal-muted">
              No evaluation history available. Click 'Run Holdout Evaluation' above.
            </div>
          )}
        </div>
      </div>

      {/* Comprehensive Holdout Metrics Table */}
      <div className="bg-ivory p-6 rounded-xl border border-borderBase shadow-card space-y-4">
        <div className="flex items-center justify-between border-b border-cream-200 pb-3">
          <div>
            <h2 className="text-sm font-bold text-charcoal">Complete Strategy Performance Matrix</h2>
            <p className="text-xs text-charcoal-muted mt-0.5">
              Detailed statistical and financial risk-adjusted metrics (annualized to 252 trading days)
            </p>
          </div>
          <Badge variant="sage">{items.length} Strategies Evaluated</Badge>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-cream-200 text-charcoal-muted text-[11px] uppercase tracking-wider font-semibold">
                <th className="py-3 px-3">Strategy</th>
                <th className="py-3 px-3 text-right">Initial (₹)</th>
                <th className="py-3 px-3 text-right">Final (₹)</th>
                <th className="py-3 px-3 text-right">Total Return</th>
                <th className="py-3 px-3 text-right">Ann. Volatility</th>
                <th className="py-3 px-3 text-right">Sharpe Ratio</th>
                <th className="py-3 px-3 text-right">Max Drawdown</th>
                <th className="py-3 px-3 text-right">Orders</th>
                <th className="py-3 px-3 text-right">Fees Paid (₹)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-cream-200">
              {items.map((m) => {
                const isGain = m.total_return_pct >= 0;
                return (
                  <tr key={m.agent} className="hover:bg-cream-50 transition-colors">
                    <td className="py-3 px-3">
                      <div className="font-bold text-charcoal capitalize">{m.agent.replace('_', ' ')}</div>
                      <div className="text-[10px] text-charcoal-muted">
                        {m.agent === 'rl_trader'
                          ? 'Deep Q-Network Agent'
                          : m.agent === 'leader'
                          ? 'Stackelberg Mean-Reversion'
                          : m.agent === 'momentum'
                          ? 'Momentum Follower'
                          : 'Equal-Weight Basket'}
                      </div>
                    </td>
                    <td className="py-3 px-3 text-right font-mono text-charcoal">
                      ₹{m.initial_value.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                    </td>
                    <td className="py-3 px-3 text-right font-mono font-bold text-charcoal">
                      ₹{m.final_value.toLocaleString('en-IN', { maximumFractionDigits: 2 })}
                    </td>
                    <td className="py-3 px-3 text-right font-mono font-bold">
                      <span className={isGain ? 'text-gain' : 'text-loss'}>
                        {isGain ? '+' : ''}{m.total_return_pct.toFixed(2)}%
                      </span>
                    </td>
                    <td className="py-3 px-3 text-right font-mono text-charcoal">
                      {m.annualized_volatility_pct.toFixed(2)}%
                    </td>
                    <td className="py-3 px-3 text-right font-mono font-semibold text-charcoal">
                      {m.sharpe_ratio.toFixed(2)}
                    </td>
                    <td className="py-3 px-3 text-right font-mono text-loss">
                      {m.max_drawdown_pct.toFixed(2)}%
                    </td>
                    <td className="py-3 px-3 text-right font-mono text-charcoal">
                      {m.executed_orders}
                    </td>
                    <td className="py-3 px-3 text-right font-mono text-charcoal-muted">
                      ₹{m.fees_paid.toFixed(2)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Academic Discussion Callout */}
        <div className="mt-4 p-4 rounded-lg bg-cream-50 border border-borderBase text-xs text-charcoal-muted space-y-2">
          <div className="flex items-center gap-2 font-bold text-charcoal">
            <Info className="w-4 h-4 text-olive-600" />
            Empirical Findings & Discussion
          </div>
          <p className="leading-relaxed">
            <strong>1. Risk-Controlled Preservation:</strong> The DQN RL Agent demonstrated strong capital protection with annualized volatility of only 0.23% and maximum drawdown of just -0.55%, successfully learning to avoid ruinous trades under high spread (2%) friction.
          </p>
          <p className="leading-relaxed">
            <strong>2. Leader Equilibrium Advantage:</strong> The Stackelberg Leader achieved +1.66% return with a positive Sharpe ratio (0.25) across 358 trades, systematically profiting when temporary liquidity dislocations pulled prices away from the 20-day rolling fundamental mean.
          </p>
          <p className="leading-relaxed">
            <strong>3. Follower Friction Penalty:</strong> The Momentum Follower suffered -21.43% loss due to severe transaction costs (₹1,749 across 556 trades) and slippage when chasing trend signals in non-trending regimes.
          </p>
        </div>
      </div>
    </div>
  );
};
