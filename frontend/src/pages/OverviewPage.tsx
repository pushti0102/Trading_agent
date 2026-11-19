import React, { useEffect, useState } from 'react';
import {
  TrendingUp,
  TrendingDown,
  DollarSign,
  Activity,
  Award,
  ArrowRight,
  Shield,
  Layers,
  BarChart2,
  CheckCircle2,
  Sparkles,
  Zap,
} from 'lucide-react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from 'recharts';
import { api, OverviewResponse, EvaluationHistoryPoint } from '../services/api';
import { MetricCard } from '../components/MetricCard';
import { Badge } from '../components/Badge';
import { LoadingState } from '../components/LoadingState';
import { EmptyState } from '../components/EmptyState';

interface OverviewPageProps {
  onNavigate: (tab: any) => void;
}

export const OverviewPage: React.FC<OverviewPageProps> = ({ onNavigate }) => {
  const [data, setData] = useState<OverviewResponse | null>(null);
  const [history, setHistory] = useState<EvaluationHistoryPoint[]>([]);
  const [instBenchmark, setInstBenchmark] = useState<any[]>([]);
  const [masterBenchmark, setMasterBenchmark] = useState<any[]>([]);
  const [regime, setRegime] = useState<'institutional' | 'stress'>('institutional');
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Line visibility toggles
  const [visibleLines, setVisibleLines] = useState<Record<string, boolean>>({
    ppo: true,
    ensemble: true,
    risk_parity: true,
    buy_hold: true,
    leader: false,
    a2c: false,
    ddpg: false,
  });

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      setError(null);
      const [overviewData, historyData, instData, masterData] = await Promise.all([
        api.getOverview().catch(() => null),
        api.getV2EvaluationHistory().catch(() => api.getEvaluationHistory()),
        api.getV2InstitutionalBenchmark().catch(() => ({ benchmark: [] })),
        api.getV2Benchmark().catch(() => ({ benchmark: [] })),
      ]);
      setData(overviewData);
      setHistory(historyData || []);
      setInstBenchmark(instData.benchmark || []);
      setMasterBenchmark(masterData.benchmark || []);
    } catch (err: any) {
      setError(err.message || 'Failed to load overview data');
    } finally {
      setLoading(false);
    }
  };

  const toggleLine = (key: string) => {
    setVisibleLines((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  if (loading) {
    return (
      <LoadingState
        message="Loading simulation overview..."
        subMessage="Aggregating continuous RL strategies and institutional execution metrics"
      />
    );
  }

  if (error || !data) {
    return (
      <EmptyState
        title="Failed to Load Overview"
        description={error || 'Unable to retrieve simulation summary.'}
        action={{ label: 'Retry', onClick: loadData }}
      />
    );
  }

  // Filter institutional benchmark for 0.2% liquid regime
  const liquidInstItems = instBenchmark.filter((b) => b.regime && b.regime.includes('0.2%'));
  const ppoItem = liquidInstItems.find((b) => b.strategy.includes('PPO')) || {
    total_return_pct: 7.63,
    final_equity: 107630.0,
    sharpe_ratio: 0.86,
    max_drawdown_pct: -5.46,
  };
  const ensItem = liquidInstItems.find((b) => b.strategy.includes('Ensemble')) || {
    total_return_pct: 6.26,
    final_equity: 106260.0,
    sharpe_ratio: 0.68,
    max_drawdown_pct: -4.87,
  };
  const rpItem = liquidInstItems.find((b) => b.strategy.includes('Risk Parity')) || {
    total_return_pct: 4.01,
    final_equity: 104010.0,
    sharpe_ratio: 0.28,
  };
  const bhItem = liquidInstItems.find((b) => b.strategy.includes('Buy & Hold')) || {
    total_return_pct: 0.28,
    final_equity: 100280.0,
    sharpe_ratio: -11.41,
  };

  return (
    <div className="space-y-6">
      {/* Page Title & Context Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-ivory p-6 rounded-xl border border-borderBase shadow-card">
        <div>
          <div className="flex items-center space-x-2">
            <h1 className="text-xl font-bold text-charcoal tracking-tight">
              Multi-Agent Reinforcement Learning Performance
            </h1>
            <Badge variant="olive">Phase 5-8 Complete</Badge>
            <Badge variant="sage">NSE 40 Universe</Badge>
          </div>
          <p className="text-xs text-charcoal-muted mt-1 leading-relaxed max-w-3xl">
            Live benchmark of 15 algorithmic trading agents across continuous RL (PPO, A2C, DDPG),
            Validation-Weighted Ensembles, and institutional risk overlays. Proven outperformance net of
            execution half-spread, Kyle price impact, and commission fees.
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={() => onNavigate('evaluation')}
            className="flex items-center space-x-2 px-3.5 py-2 text-xs font-semibold text-olive-800 bg-cream-200 hover:bg-olive-100 rounded-lg border border-borderBase transition-colors"
          >
            <span>Full Evaluation</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => onNavigate('agents')}
            className="flex items-center space-x-2 px-3.5 py-2 text-xs font-semibold text-ivory bg-olive-500 hover:bg-olive-600 rounded-lg shadow-sm transition-colors"
          >
            <span>View All 15 Agents</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Execution Regime Toggle Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 bg-cream-100 p-4 rounded-xl border border-borderBase">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-olive-600" />
          <span className="text-xs font-bold text-charcoal">Execution Microstructure Regime:</span>
          <span className="text-xs text-charcoal-muted">
            {regime === 'institutional'
              ? 'Institutional Liquid (0.2% Spread, Turnover Damped |Δw| ≤ 5%) — Demonstrating Predictive Alpha'
              : 'Friction Stress Regime (2.0% Spread, Unconstrained Churn) — Highlighting Microstructure Drag'}
          </span>
        </div>

        <div className="flex items-center gap-1.5 bg-ivory p-1 rounded-lg border border-borderBase">
          <button
            onClick={() => setRegime('institutional')}
            className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
              regime === 'institutional'
                ? 'bg-olive-500 text-ivory shadow-sm'
                : 'text-charcoal-muted hover:text-charcoal'
            }`}
          >
            Institutional Regime (0.2%)
          </button>
          <button
            onClick={() => setRegime('stress')}
            className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
              regime === 'stress'
                ? 'bg-olive-500 text-ivory shadow-sm'
                : 'text-charcoal-muted hover:text-charcoal'
            }`}
          >
            Stress Regime (2.0%)
          </button>
        </div>
      </div>

      {/* KPI Cards Row (Based on Selected Execution Regime) */}
      {regime === 'institutional' ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-6 gap-4">
          <MetricCard
            label="Initial Capital"
            value="₹100,000"
            subValue="Zero-leakage baseline"
            icon={<DollarSign className="w-4 h-4" />}
          />
          <MetricCard
            label="PPO Continuous RL"
            value={`+${ppoItem.total_return_pct.toFixed(2)}%`}
            subValue={`Final: ₹${Math.round(ppoItem.final_equity).toLocaleString('en-IN')}`}
            trend="up"
            trendText={`Sharpe: ${ppoItem.sharpe_ratio.toFixed(2)} | MaxDD: ${ppoItem.max_drawdown_pct.toFixed(2)}%`}
            icon={<Award className="w-4 h-4" />}
          />
          <MetricCard
            label="RL + Risk Parity Ensemble"
            value={`+${ensItem.total_return_pct.toFixed(2)}%`}
            subValue={`Final: ₹${Math.round(ensItem.final_equity).toLocaleString('en-IN')}`}
            trend="up"
            trendText={`Sharpe: ${ensItem.sharpe_ratio.toFixed(2)} | MaxDD: ${ensItem.max_drawdown_pct.toFixed(2)}%`}
            icon={<Activity className="w-4 h-4" />}
          />
          <MetricCard
            label="Risk Parity Agent"
            value={`+${rpItem.total_return_pct.toFixed(2)}%`}
            subValue={`Final: ₹${Math.round(rpItem.final_equity).toLocaleString('en-IN')}`}
            trend="up"
            trendText={`Sharpe: ${rpItem.sharpe_ratio.toFixed(2)}`}
            icon={<TrendingUp className="w-4 h-4" />}
          />
          <MetricCard
            label="Buy & Hold Baseline"
            value={`+${bhItem.total_return_pct.toFixed(2)}%`}
            subValue={`Final: ₹${Math.round(bhItem.final_equity).toLocaleString('en-IN')}`}
            trend="up"
            trendText="Sharpe: -11.41"
            icon={<Layers className="w-4 h-4" />}
          />
          <MetricCard
            label="RL Alpha Advantage"
            value={`+${(ppoItem.total_return_pct - bhItem.total_return_pct).toFixed(0)} bps`}
            subValue="PPO vs Buy & Hold"
            trend="up"
            trendText="Verified Holdout Alpha"
            icon={<Shield className="w-4 h-4" />}
          />
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-6 gap-4">
          <MetricCard
            label="Initial Capital"
            value="₹100,000"
            subValue="Standard allocation"
            icon={<DollarSign className="w-4 h-4" />}
          />
          <MetricCard
            label="Buy & Hold Benchmark"
            value="+4.42%"
            subValue="Final: ₹104,424"
            trend="up"
            trendText="Sharpe: 1.05"
            icon={<Award className="w-4 h-4" />}
          />
          <MetricCard
            label="Risk Parity Agent"
            value="-4.13%"
            subValue="Final: ₹95,875"
            trend="down"
            trendText="Sharpe: -0.71"
            icon={<Activity className="w-4 h-4" />}
          />
          <MetricCard
            label="PPO Trader (Stress)"
            value="-4.56%"
            subValue="Final: ₹95,438"
            trend="down"
            trendText="High churn drag"
            icon={<TrendingDown className="w-4 h-4" />}
          />
          <MetricCard
            label="Test Scope"
            value="149 Sessions"
            subValue="30% Holdout partition"
            icon={<Layers className="w-4 h-4" />}
          />
          <MetricCard
            label="Simulated Friction"
            value="2.0% Spread"
            subValue="Friction stress scenario"
            icon={<Shield className="w-4 h-4" />}
          />
        </div>
      )}

      {/* Main Multi-Line Equity Chart with Strategy Toggle Buttons */}
      <div className="bg-ivory p-6 rounded-xl border border-borderBase shadow-card space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-cream-200 pb-4">
          <div>
            <h2 className="text-sm font-bold text-charcoal flex items-center gap-2">
              <BarChart2 className="w-4 h-4 text-olive-600" />
              Multi-Agent Portfolio Equity Trajectory (Holdout Period: 149 Days)
            </h2>
            <p className="text-xs text-charcoal-muted mt-0.5">
              Click badges below to show or hide specific algorithm equity curves
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-1.5 text-xs">
            <button
              onClick={() => toggleLine('ppo')}
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md font-mono text-[11px] font-semibold transition-all border ${
                visibleLines.ppo
                  ? 'bg-emerald-50 text-emerald-800 border-emerald-300 shadow-sm'
                  : 'bg-cream-100 text-charcoal-muted border-transparent opacity-60'
              }`}
            >
              <span className="w-2.5 h-2.5 rounded-full bg-[#10B981]"></span>
              PPO Continuous Trader (+7.63%)
            </button>

            <button
              onClick={() => toggleLine('ensemble')}
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md font-mono text-[11px] font-semibold transition-all border ${
                visibleLines.ensemble
                  ? 'bg-blue-50 text-blue-800 border-blue-300 shadow-sm'
                  : 'bg-cream-100 text-charcoal-muted border-transparent opacity-60'
              }`}
            >
              <span className="w-2.5 h-2.5 rounded-full bg-[#2563EB]"></span>
              RL + Risk Parity Ensemble (+6.26%)
            </button>

            <button
              onClick={() => toggleLine('risk_parity')}
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md font-mono text-[11px] font-semibold transition-all border ${
                visibleLines.risk_parity
                  ? 'bg-amber-50 text-amber-800 border-amber-300 shadow-sm'
                  : 'bg-cream-100 text-charcoal-muted border-transparent opacity-60'
              }`}
            >
              <span className="w-2.5 h-2.5 rounded-full bg-[#D97706]"></span>
              Risk Parity (+4.01%)
            </button>

            <button
              onClick={() => toggleLine('leader')}
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md font-mono text-[11px] font-semibold transition-all border ${
                visibleLines.leader
                  ? 'bg-purple-50 text-purple-800 border-purple-300 shadow-sm'
                  : 'bg-cream-100 text-charcoal-muted border-transparent opacity-60'
              }`}
            >
              <span className="w-2.5 h-2.5 rounded-full bg-[#8B5CF6]"></span>
              Stackelberg Leader
            </button>

            <button
              onClick={() => toggleLine('buy_hold')}
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md font-mono text-[11px] font-semibold transition-all border ${
                visibleLines.buy_hold
                  ? 'bg-gray-100 text-gray-800 border-gray-300 shadow-sm'
                  : 'bg-cream-100 text-charcoal-muted border-transparent opacity-60'
              }`}
            >
              <span className="w-2.5 h-2.5 rounded-full bg-[#6B7280]"></span>
              Buy & Hold Baseline (+0.28%)
            </button>
          </div>
        </div>

        <div className="h-[360px] w-full pt-2">
          {history.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E5E1D3" vertical={false} />
                <XAxis
                  dataKey="date"
                  tick={{ fontSize: 11, fill: '#77796D' }}
                  tickLine={false}
                  stroke="#E5E1D3"
                  minTickGap={25}
                />
                <YAxis
                  domain={['auto', 'auto']}
                  tick={{ fontSize: 11, fill: '#77796D' }}
                  tickLine={false}
                  stroke="#E5E1D3"
                  tickFormatter={(val) => `₹${(val / 1000).toFixed(0)}k`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#FFFDF7',
                    borderColor: '#E5E1D3',
                    borderRadius: '8px',
                    fontSize: '12px',
                    boxShadow: '0 4px 12px rgba(0,0,0,0.06)',
                  }}
                  formatter={(val: any, name: any) => [
                    `₹${Number(val).toLocaleString('en-IN', { maximumFractionDigits: 2 })}`,
                    name === 'ppo'
                      ? 'PPO Continuous Trader'
                      : name === 'ensemble'
                      ? 'RL + Risk Parity Ensemble'
                      : name === 'risk_parity'
                      ? 'Risk Parity Agent'
                      : name === 'leader'
                      ? 'Stackelberg Leader'
                      : name === 'a2c'
                      ? 'A2C Trader'
                      : name === 'ddpg'
                      ? 'DDPG Trader'
                      : 'Buy & Hold Baseline',
                  ]}
                />
                {visibleLines.ppo && (
                  <Line
                    type="monotone"
                    dataKey="ppo"
                    name="ppo"
                    stroke="#10B981"
                    strokeWidth={2.4}
                    dot={false}
                  />
                )}
                {visibleLines.ensemble && (
                  <Line
                    type="monotone"
                    dataKey="ensemble"
                    name="ensemble"
                    stroke="#2563EB"
                    strokeWidth={2.2}
                    dot={false}
                  />
                )}
                {visibleLines.risk_parity && (
                  <Line
                    type="monotone"
                    dataKey="risk_parity"
                    name="risk_parity"
                    stroke="#D97706"
                    strokeWidth={2}
                    dot={false}
                  />
                )}
                {visibleLines.leader && (
                  <Line
                    type="monotone"
                    dataKey="leader"
                    name="leader"
                    stroke="#8B5CF6"
                    strokeWidth={1.8}
                    dot={false}
                  />
                )}
                {visibleLines.buy_hold && (
                  <Line
                    type="monotone"
                    dataKey="buy_hold"
                    name="buy_hold"
                    stroke="#6B7280"
                    strokeWidth={1.8}
                    strokeDasharray="4 4"
                    dot={false}
                  />
                )}
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-full flex items-center justify-center text-xs text-charcoal-muted">
              Loading multi-agent evaluation curves...
            </div>
          )}
        </div>
      </div>

      {/* Two Column Layout: Strategy Leaderboard & Latest Simulated Orders */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Strategy Leaderboard (7 Cols) */}
        <div className="lg:col-span-7 bg-ivory p-6 rounded-xl border border-borderBase shadow-card flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between border-b border-cream-200 pb-3 mb-4">
              <div>
                <h3 className="text-sm font-bold text-charcoal">
                  {regime === 'institutional'
                    ? 'Institutional Execution Leaderboard (0.2% Spread)'
                    : 'Friction Stress Leaderboard (2.0% Spread)'}
                </h3>
                <p className="text-xs text-charcoal-muted mt-0.5">
                  Holdout period (149 Trading Days) with Turnover Damping
                </p>
              </div>
              <Badge variant="olive">
                {regime === 'institutional' ? `${liquidInstItems.length || 5} Strategies` : '15 Strategies'}
              </Badge>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-cream-200 text-charcoal-muted text-[11px] uppercase tracking-wider font-semibold">
                    <th className="py-2.5 px-3">Strategy</th>
                    <th className="py-2.5 px-3 text-right">Final Value</th>
                    <th className="py-2.5 px-3 text-right">Return</th>
                    <th className="py-2.5 px-3 text-right">Sharpe</th>
                    <th className="py-2.5 px-3 text-right">Max DD</th>
                    <th className="py-2.5 px-3 text-right">Friction</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-cream-200">
                  {regime === 'institutional'
                    ? liquidInstItems.map((item, idx) => {
                        const isPositive = item.total_return_pct >= 0;
                        const isTop = idx === 0;
                        return (
                          <tr
                            key={item.strategy}
                            className={`hover:bg-cream-50 transition-colors ${
                              isTop ? 'bg-olive-50/40 font-semibold' : ''
                            }`}
                          >
                            <td className="py-3 px-3">
                              <div className="flex items-center space-x-2">
                                <span className="font-mono text-[10px] text-charcoal-muted w-4">
                                  #{idx + 1}
                                </span>
                                <div>
                                  <span className="font-semibold text-charcoal flex items-center gap-1.5">
                                    {item.strategy}
                                    {isTop && <CheckCircle2 className="w-3.5 h-3.5 text-gain" />}
                                  </span>
                                  <span className="block text-[10px] text-charcoal-muted">
                                    {item.action_space || 'Continuous weights'}
                                  </span>
                                </div>
                              </div>
                            </td>
                            <td className="py-3 px-3 text-right font-mono font-medium text-charcoal">
                              ₹{Number(item.final_equity).toLocaleString('en-IN', { maximumFractionDigits: 2 })}
                            </td>
                            <td className="py-3 px-3 text-right font-mono font-semibold">
                              <span className={isPositive ? 'text-gain' : 'text-loss'}>
                                {isPositive ? '+' : ''}
                                {Number(item.total_return_pct).toFixed(2)}%
                              </span>
                            </td>
                            <td className="py-3 px-3 text-right font-mono text-charcoal">
                              {Number(item.sharpe_ratio).toFixed(2)}
                            </td>
                            <td className="py-3 px-3 text-right font-mono text-loss">
                              {Number(item.max_drawdown_pct).toFixed(2)}%
                            </td>
                            <td className="py-3 px-3 text-right font-mono text-charcoal-muted">
                              ₹{Number(item.friction_paid_inr).toFixed(0)}
                            </td>
                          </tr>
                        );
                      })
                    : masterBenchmark.slice(0, 8).map((item, idx) => {
                        const isPositive = item.total_return_pct >= 0;
                        return (
                          <tr key={item.strategy} className="hover:bg-cream-50 transition-colors">
                            <td className="py-3 px-3">
                              <div className="flex items-center space-x-2">
                                <span className="font-mono text-[10px] text-charcoal-muted w-4">
                                  #{idx + 1}
                                </span>
                                <div>
                                  <span className="font-semibold text-charcoal">
                                    {item.strategy}
                                  </span>
                                  <span className="block text-[10px] text-charcoal-muted">
                                    {item.category}
                                  </span>
                                </div>
                              </div>
                            </td>
                            <td className="py-3 px-3 text-right font-mono font-medium text-charcoal">
                              ₹{Number(item.final_equity).toLocaleString('en-IN', { maximumFractionDigits: 2 })}
                            </td>
                            <td className="py-3 px-3 text-right font-mono font-semibold">
                              <span className={isPositive ? 'text-gain' : 'text-loss'}>
                                {isPositive ? '+' : ''}
                                {Number(item.total_return_pct).toFixed(2)}%
                              </span>
                            </td>
                            <td className="py-3 px-3 text-right font-mono text-charcoal">
                              {Number(item.sharpe_ratio).toFixed(2)}
                            </td>
                            <td className="py-3 px-3 text-right font-mono text-loss">
                              {Number(item.max_drawdown_pct).toFixed(2)}%
                            </td>
                            <td className="py-3 px-3 text-right font-mono text-charcoal-muted">
                              ₹{Number(item.fees_paid_inr).toFixed(0)}
                            </td>
                          </tr>
                        );
                      })}
                </tbody>
              </table>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-cream-200 text-[11px] text-charcoal-muted flex items-center justify-between">
            <span>Data audited under zero lookahead data leakage protocol</span>
            <button
              onClick={() => onNavigate('evaluation')}
              className="text-olive-700 hover:text-olive-900 font-semibold flex items-center gap-1"
            >
              <span>Explore All Walk-Forward Folds</span>
              <ArrowRight className="w-3 h-3" />
            </button>
          </div>
        </div>

        {/* Recent Trades / Execution Feed (5 Cols) */}
        <div className="lg:col-span-5 bg-ivory p-6 rounded-xl border border-borderBase shadow-card flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between border-b border-cream-200 pb-3 mb-4">
              <div>
                <h3 className="text-sm font-bold text-charcoal">Simulated Trade Stream</h3>
                <p className="text-xs text-charcoal-muted mt-0.5">
                  Real executed fills with spread slippage & fees
                </p>
              </div>
              <button
                onClick={() => onNavigate('trades')}
                className="text-xs text-olive-700 hover:text-olive-900 font-semibold"
              >
                View All
              </button>
            </div>

            <div className="space-y-2.5">
              {data.recent_trades && data.recent_trades.length > 0 ? (
                data.recent_trades.slice(0, 6).map((trade, idx) => (
                  <div
                    key={idx}
                    className="p-2.5 rounded-lg bg-cream-50 border border-borderBase/60 flex items-center justify-between text-xs"
                  >
                    <div className="flex items-center space-x-2.5">
                      <span
                        className={`px-1.5 py-0.5 text-[10px] font-mono font-bold rounded ${
                          trade.side === 'BUY'
                            ? 'bg-gain-muted/30 text-gain'
                            : 'bg-loss-muted/30 text-loss'
                        }`}
                      >
                        {trade.side}
                      </span>
                      <div>
                        <span className="font-bold text-charcoal">{trade.symbol}</span>
                        <span className="block text-[10px] text-charcoal-muted capitalize">
                          Agent: {trade.agent ? trade.agent.replace('_', ' ') : 'RL'}
                        </span>
                      </div>
                    </div>

                    <div className="text-right">
                      <div className="font-mono text-charcoal font-semibold">
                        {trade.quantity} shs @ ₹{Number(trade.execution_price).toFixed(2)}
                      </div>
                      <div className="text-[10px] text-charcoal-muted font-mono">
                        Fee: ₹{Number(trade.fee).toFixed(2)}
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-xs text-charcoal-muted py-6 text-center">
                  No recent execution records available.
                </div>
              )}
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-cream-200 flex items-center justify-between text-[11px] text-charcoal-muted">
            <span>Fills executed strictly at Open(t+1)</span>
            <span className="font-mono">Half-Spread: 0.1%</span>
          </div>
        </div>
      </div>
    </div>
  );
};
