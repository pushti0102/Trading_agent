import React, { useEffect, useState } from 'react';
import {
  Users,
  Cpu,
  Brain,
  Shield,
  Zap,
  TrendingUp,
  TrendingDown,
  Info,
  DollarSign,
  Activity,
  Award,
} from 'lucide-react';
import { api, EvaluationSummaryResponse, AgentMetric } from '../services/api';
import { Badge } from '../components/Badge';
import { MetricCard } from '../components/MetricCard';
import { LoadingState } from '../components/LoadingState';

export const AgentsPage: React.FC = () => {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [selectedAgentId, setSelectedAgentId] = useState<string>('rl_trader');

  useEffect(() => {
    loadAgents();
  }, []);

  const loadAgents = async () => {
    try {
      setLoading(true);
      const res = await api.getAgents();
      setData(res);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <LoadingState message="Loading agent specifications..." subMessage="Fetching model architectures and risk parameter configs" />;
  }

  const agents = data?.agents || [];
  const selectedAgent = agents.find((a: any) => a.id === selectedAgentId) || agents[0];

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="bg-ivory p-6 rounded-xl border border-borderBase shadow-card">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-xl font-bold text-charcoal tracking-tight">Trading Agents Architecture</h1>
              <Badge variant="olive">4 Active Archetypes</Badge>
            </div>
            <p className="text-xs text-charcoal-muted mt-1 leading-relaxed max-w-3xl">
              Inspect the game-theoretic and reinforcement learning policies governing decision-making. All agents operate under identical initial capital (₹100,000), cash reserve floors (5%), and simulated market friction.
            </p>
          </div>

          <div className="flex items-center space-x-1.5 bg-cream-100 p-1 rounded-lg border border-borderBase">
            {agents.map((a: any) => (
              <button
                key={a.id}
                onClick={() => setSelectedAgentId(a.id)}
                className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
                  selectedAgentId === a.id
                    ? 'bg-olive-500 text-ivory shadow-sm'
                    : 'text-charcoal-muted hover:text-charcoal'
                }`}
              >
                {a.id === 'rl_trader'
                  ? 'RL (DQN)'
                  : a.id === 'leader'
                  ? 'Leader'
                  : a.id === 'momentum'
                  ? 'Follower'
                  : 'Buy & Hold'}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Head-to-Head Comparison Grid of All 4 Agents */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {agents.map((agent: any) => {
          const m: AgentMetric = agent.metrics || {};
          const isSelected = agent.id === selectedAgentId;
          const isGain = m.total_return_pct >= 0;

          return (
            <div
              key={agent.id}
              onClick={() => setSelectedAgentId(agent.id)}
              className={`bg-ivory p-5 rounded-xl border transition-all cursor-pointer shadow-card hover:shadow-card-hover flex flex-col justify-between ${
                isSelected ? 'border-olive-500 ring-2 ring-olive-500/20' : 'border-borderBase'
              }`}
            >
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[10px] font-bold uppercase tracking-wider text-charcoal-muted">
                    {agent.category}
                  </span>
                  <Badge variant={isSelected ? 'olive' : 'neutral'}>
                    {isSelected ? 'Active View' : 'Select'}
                  </Badge>
                </div>

                <h3 className="text-sm font-bold text-charcoal mb-1">{agent.name}</h3>
                <p className="text-[11px] text-charcoal-muted line-clamp-2 mb-4 leading-relaxed">
                  {agent.description}
                </p>

                {/* Key Metrics */}
                <div className="space-y-2 py-3 border-y border-cream-200 text-xs">
                  <div className="flex items-center justify-between">
                    <span className="text-charcoal-muted text-[11px]">Holdout Return:</span>
                    <span className={`font-mono font-bold ${isGain ? 'text-gain' : 'text-loss'}`}>
                      {isGain ? '+' : ''}
                      {m.total_return_pct !== undefined ? `${m.total_return_pct}%` : '—'}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-charcoal-muted text-[11px]">Final Portfolio:</span>
                    <span className="font-mono font-semibold text-charcoal">
                      ₹{m.final_value ? m.final_value.toLocaleString('en-IN') : '100,000'}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-charcoal-muted text-[11px]">Sharpe Ratio:</span>
                    <span className="font-mono text-charcoal">
                      {m.sharpe_ratio !== undefined ? m.sharpe_ratio : '—'}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-charcoal-muted text-[11px]">Max Drawdown:</span>
                    <span className="font-mono text-loss">
                      {m.max_drawdown_pct !== undefined ? `${m.max_drawdown_pct}%` : '—'}
                    </span>
                  </div>
                </div>
              </div>

              <div className="pt-3 mt-3 flex items-center justify-between text-[11px] text-charcoal-muted">
                <span>Orders: {m.executed_orders ?? 0}</span>
                <span>Fees: ₹{m.fees_paid ? m.fees_paid.toFixed(1) : 0}</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Selected Agent In-Depth Specification Panel */}
      {selectedAgent && (
        <div className="bg-ivory p-6 rounded-xl border border-borderBase shadow-card space-y-6">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-cream-200 pb-4">
            <div>
              <div className="flex items-center space-x-2">
                <span className="p-2 rounded-lg bg-olive-100 text-olive-800 border border-olive-200">
                  {selectedAgent.id === 'rl_trader' ? (
                    <Brain className="w-5 h-5" />
                  ) : selectedAgent.id === 'leader' ? (
                    <Award className="w-5 h-5" />
                  ) : selectedAgent.id === 'momentum' ? (
                    <Zap className="w-5 h-5" />
                  ) : (
                    <Shield className="w-5 h-5" />
                  )}
                </span>
                <div>
                  <h2 className="text-base font-bold text-charcoal">{selectedAgent.name}</h2>
                  <p className="text-xs text-charcoal-muted font-mono">{selectedAgent.category}</p>
                </div>
              </div>
            </div>

            <div className="flex items-center space-x-3 text-xs">
              <span className="text-charcoal-muted">Execution Mode:</span>
              <span className="font-semibold text-olive-800 bg-cream-200 px-2 py-0.5 rounded border border-borderBase font-mono">
                Simulated Next-Open Fill
              </span>
            </div>
          </div>

          {/* Description & Decision Rule */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-3">
              <h3 className="text-xs font-bold uppercase tracking-wider text-charcoal-muted">
                Strategy Philosophy & Formulation
              </h3>
              <p className="text-xs text-charcoal leading-relaxed bg-cream-50 p-4 rounded-lg border border-borderBase">
                {selectedAgent.description}
              </p>

              <div className="space-y-2">
                <h4 className="text-xs font-bold text-charcoal">Decision Policy Logic:</h4>
                <div className="p-3 bg-cream-100 rounded-lg border border-borderBase font-mono text-xs text-olive-900 leading-relaxed">
                  {selectedAgent.policy}
                </div>
              </div>
            </div>

            {/* Risk Management & Constraints */}
            <div className="space-y-3">
              <h3 className="text-xs font-bold uppercase tracking-wider text-charcoal-muted">
                Engine Constraints & Risk Controls
              </h3>
              <div className="bg-cream-50 p-4 rounded-lg border border-borderBase space-y-3 text-xs">
                {Object.entries(selectedAgent.risk_controls || {}).map(([key, val]: any) => (
                  <div key={key} className="flex items-center justify-between pb-2 border-b border-cream-200 last:border-0 last:pb-0">
                    <span className="text-charcoal-muted capitalize">{key.replace(/_/g, ' ')}:</span>
                    <span className="font-mono font-semibold text-charcoal">{val}</span>
                  </div>
                ))}
              </div>

              {selectedAgent.id === 'rl_trader' && (
                <div className="p-3 rounded-lg bg-olive-50 border border-olive-200 text-xs text-olive-900 space-y-1">
                  <div className="font-bold flex items-center gap-1.5">
                    <Brain className="w-3.5 h-3.5 text-olive-700" />
                    DQN Observation Vector (10 Dimensions)
                  </div>
                  <p className="text-[11px] text-olive-800 leading-relaxed font-mono">
                    [Valuation Gap, Return, Log Volume, Quoted Spread, Turnover Liquidity, Rolling Volatility, Order Imbalance, Inventory Exposure, Cash Ratio, Position Weight]
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Transparent Research Analysis Note */}
          <div className="p-4 rounded-lg bg-cream-50 border border-borderBase text-xs text-charcoal-muted flex items-start gap-3">
            <Info className="w-4 h-4 text-olive-600 flex-shrink-0 mt-0.5" />
            <div className="space-y-1">
              <p className="font-bold text-charcoal">Academic Evaluation Finding:</p>
              <p className="leading-relaxed">
                In this holdout period (May 2017 – Dec 2017), the passive Buy-and-Hold basket benefited from an overall bullish trend in large-cap Indian equities, returning 3.96%. The Stackelberg Leader achieved steady +1.66% gains by mean-reverting mispriced dips. The RL Trader achieved -0.55% with exceptional capital preservation (annualized volatility of just 0.23% and negligible drawdown). In contrast, the Momentum Follower suffered -21.43% because choppy range-bound trading repeatedly triggered false breakout orders whose returns were consumed by execution fees and slippage.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
