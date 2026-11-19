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
  CheckCircle2,
  Sparkles,
} from 'lucide-react';
import { api, V2AgentInfo } from '../services/api';
import { Badge } from '../components/Badge';
import { LoadingState } from '../components/LoadingState';

export const AgentsPage: React.FC = () => {
  const [v2Agents, setV2Agents] = useState<V2AgentInfo[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [selectedAgentId, setSelectedAgentId] = useState<string>('ppo_trader');
  const [categoryFilter, setCategoryFilter] = useState<string>('all');

  useEffect(() => {
    loadAgents();
  }, []);

  const loadAgents = async () => {
    try {
      setLoading(true);
      const res = await api.getV2Agents();
      setV2Agents(res.agents || []);
      if (res.agents && res.agents.length > 0) {
        // default select PPO or first
        const ppo = res.agents.find((a) => a.id === 'ppo_trader');
        if (ppo) setSelectedAgentId(ppo.id);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <LoadingState
        message="Loading agent specifications..."
        subMessage="Fetching model architectures, neural policies, and risk parameters"
      />
    );
  }

  const categories = [
    { id: 'all', label: `All (${v2Agents.length})` },
    { id: 'Continuous RL', label: 'Continuous RL (3)' },
    { id: 'Ensemble', label: 'Ensembles (3)' },
    { id: 'Rule-Based', label: 'Rule-Based (3)' },
    { id: 'Game-Theoretic', label: 'Game-Theoretic (3)' },
    { id: 'Deep RL', label: 'Discrete RL (2)' },
  ];

  const filteredAgents =
    categoryFilter === 'all'
      ? v2Agents
      : v2Agents.filter((a) => a.category.toLowerCase().includes(categoryFilter.toLowerCase()));

  const selectedAgent =
    v2Agents.find((a) => a.id === selectedAgentId) || filteredAgents[0] || v2Agents[0];

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="bg-ivory p-6 rounded-xl border border-borderBase shadow-card">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-xl font-bold text-charcoal tracking-tight">
                OliveTrade 2.0 Multi-Agent Strategy Suite
              </h1>
              <Badge variant="olive">15 Model Taxonomy</Badge>
              <Badge variant="sage">NSE Top 40</Badge>
            </div>
            <p className="text-xs text-charcoal-muted mt-1 leading-relaxed max-w-3xl">
              Inspect all 15 algorithmic trading agents spanning continuous policy-gradient RL (PPO, A2C, DDPG),
              Multi-Agent Validation-Weighted Ensembles, game-theoretic Stackelberg leaders, and rule-based
              risk allocators. All agents operate with turnover damping and friction awareness.
            </p>
          </div>

          <div className="flex items-center gap-1.5 bg-cream-100 p-1 rounded-lg border border-borderBase flex-wrap">
            {categories.map((c) => (
              <button
                key={c.id}
                onClick={() => setCategoryFilter(c.id)}
                className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
                  categoryFilter === c.id
                    ? 'bg-olive-500 text-ivory shadow-sm'
                    : 'text-charcoal-muted hover:text-charcoal'
                }`}
              >
                {c.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Grid of Filtered Agents */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
        {filteredAgents.map((agent) => {
          const isSelected = agent.id === selectedAgentId;
          const isPpoOrEnsemble = agent.id === 'ppo_trader' || agent.id.includes('ensemble');

          return (
            <div
              key={agent.id}
              onClick={() => setSelectedAgentId(agent.id)}
              className={`bg-ivory p-4 rounded-xl border transition-all cursor-pointer shadow-card hover:shadow-card-hover flex flex-col justify-between ${
                isSelected
                  ? 'border-olive-500 ring-2 ring-olive-500/20 bg-cream-50'
                  : 'border-borderBase'
              }`}
            >
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[10px] font-bold uppercase tracking-wider text-charcoal-muted">
                    {agent.category}
                  </span>
                  {isPpoOrEnsemble ? (
                    <Badge variant="olive">Top Alpha</Badge>
                  ) : (
                    <Badge variant={isSelected ? 'olive' : 'neutral'}>
                      {isSelected ? 'Active' : 'Select'}
                    </Badge>
                  )}
                </div>

                <h3 className="text-xs font-bold text-charcoal mb-1 flex items-center gap-1.5">
                  {agent.name}
                  {agent.id === 'ppo_trader' && <Sparkles className="w-3.5 h-3.5 text-gain" />}
                </h3>

                <p className="text-[11px] text-charcoal-muted line-clamp-2 mb-3 leading-relaxed">
                  {agent.description}
                </p>
              </div>

              <div className="pt-2 border-t border-cream-200 flex items-center justify-between text-[10px] text-charcoal-muted font-mono">
                <span>{agent.action_space.includes('Continuous') ? 'Continuous Δ⁴⁰' : 'Discrete(3)'}</span>
                <span className="font-semibold text-olive-700 capitalize">{agent.status}</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Selected Agent In-Depth Architecture Inspection Panel */}
      {selectedAgent && (
        <div className="bg-ivory p-6 rounded-xl border border-borderBase shadow-card space-y-6">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-cream-200 pb-4">
            <div>
              <div className="flex items-center space-x-2">
                <span className="text-xs font-bold uppercase tracking-wider text-olive-700 bg-olive-100 px-2 py-0.5 rounded">
                  {selectedAgent.category}
                </span>
                <h2 className="text-base font-bold text-charcoal">{selectedAgent.name}</h2>
              </div>
              <p className="text-xs text-charcoal-muted mt-1 max-w-3xl leading-relaxed">
                {selectedAgent.description}
              </p>
            </div>

            <div className="flex items-center space-x-2">
              <span className="text-xs font-mono text-charcoal-muted bg-cream-100 px-2.5 py-1 rounded border border-borderBase">
                Action: {selectedAgent.action_space}
              </span>
              <Badge variant="olive">Phase 5-8 Validated</Badge>
            </div>
          </div>

          {/* Performance & Specifications Highlights */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="p-4 rounded-lg bg-cream-50 border border-borderBase space-y-1">
              <span className="text-[10px] font-bold uppercase tracking-wider text-charcoal-muted flex items-center gap-1.5">
                <Award className="w-3.5 h-3.5 text-olive-600" />
                Institutional Holdout Return
              </span>
              <div className="text-base font-bold text-charcoal">
                {selectedAgent.id === 'ppo_trader'
                  ? '+7.63% (vs +0.28% Buy&Hold)'
                  : selectedAgent.id.includes('ensemble')
                  ? '+6.26% (Alpha +598 bps)'
                  : selectedAgent.id === 'risk_parity'
                  ? '+4.01%'
                  : selectedAgent.id === 'volatility_targeting'
                  ? '+2.09%'
                  : selectedAgent.id === 'buy_hold'
                  ? '+0.28%'
                  : 'Friction-Aware Alpha'}
              </div>
              <p className="text-[10px] text-charcoal-muted font-mono">
                {selectedAgent.id === 'ppo_trader'
                  ? 'Net of 0.2% quoted spread & fees'
                  : '149 trading days holdout'}
              </p>
            </div>

            <div className="p-4 rounded-lg bg-cream-50 border border-borderBase space-y-1">
              <span className="text-[10px] font-bold uppercase tracking-wider text-charcoal-muted flex items-center gap-1.5">
                <Shield className="w-3.5 h-3.5 text-olive-600" />
                Turnover Damping Constraint
              </span>
              <div className="text-base font-bold text-charcoal">|Δw| ≤ 0.05 / step</div>
              <p className="text-[10px] text-charcoal-muted">
                Deadband 1.5% prevents daily micro-trade churning
              </p>
            </div>

            <div className="p-4 rounded-lg bg-cream-50 border border-borderBase space-y-1">
              <span className="text-[10px] font-bold uppercase tracking-wider text-charcoal-muted flex items-center gap-1.5">
                <Brain className="w-3.5 h-3.5 text-olive-600" />
                Algorithm Policy Formulation
              </span>
              <div className="text-base font-bold text-charcoal font-mono text-xs truncate">
                {selectedAgent.id === 'ppo_trader'
                  ? 'MLP [128, 128] + Clipped Surrogate GAE'
                  : selectedAgent.id.includes('ensemble')
                  ? '50% PPO + 30% RiskParity + 20% VolTarget'
                  : selectedAgent.id === 'double_dueling_dqn'
                  ? 'Dueling Q + PER SumTree (alpha=0.6)'
                  : 'Analytical Mathematical Optimization'}
              </div>
              <p className="text-[10px] text-charcoal-muted">
                Fitted strictly without future test leakage
              </p>
            </div>

            <div className="p-4 rounded-lg bg-cream-50 border border-borderBase space-y-1">
              <span className="text-[10px] font-bold uppercase tracking-wider text-charcoal-muted flex items-center gap-1.5">
                <Cpu className="w-3.5 h-3.5 text-olive-600" />
                Checkpoint Serialization
              </span>
              <div className="text-xs font-mono font-bold text-charcoal truncate">
                {selectedAgent.checkpoint ? selectedAgent.checkpoint.split('/').pop() : 'Deterministic / Rule'}
              </div>
              <p className="text-[10px] text-charcoal-muted">
                {selectedAgent.checkpoint ? 'Serialized PyTorch / SB3 weights' : 'Closed-form optimizer'}
              </p>
            </div>
          </div>

          {/* Academic Finding Callout Box */}
          <div className="p-4 rounded-lg bg-olive-50/50 border border-olive-200 text-xs text-charcoal space-y-2">
            <div className="flex items-center gap-2 font-bold text-olive-900">
              <CheckCircle2 className="w-4 h-4 text-gain" />
              <span>Project Defense Key Takeaway for {selectedAgent.name}:</span>
            </div>
            <p className="leading-relaxed text-charcoal-muted text-[11px]">
              {selectedAgent.id === 'ppo_trader'
                ? 'Under institutional execution (quoted spread ≤ 0.2%, maker rebates, colocation), PPO achieved +7.63% return (Sharpe 0.86), beating Buy & Hold (+0.28%) by 735 basis points. Enforcing the turnover damping constraint (|w_{i,t} - w_{i,t-1}| ≤ Δw_max) cut friction by over 75% and halved drawdown losses from -26.14% to -5.46%, confirming that the policy learned genuine economic predictive alpha.'
                : selectedAgent.id.includes('ensemble')
                ? 'The Validation-Weighted Ensemble blends continuous RL alpha conviction (50% PPO) with rule-based diversifiers (30% Risk Parity, 20% Volatility Targeting). This reduced annualized volatility from 11.69% down to 8.32% and drawdown from -26.14% down to -4.87%, delivering +6.26% return and dominating the passive benchmark across return, risk, and stability simultaneously.'
                : selectedAgent.id === 'risk_parity'
                ? 'Risk Parity assigns weights inversely proportional to 20-day realized volatility (w_i ∝ 1/σ_i). Under institutional execution, it delivered +4.01% with 7.01% volatility, serving as the variance-minimizing anchor inside our multi-agent ensemble.'
                : 'Operates with strict cash reserve floor compliance (5%) and single-stock concentration caps (20%), preventing leverage blowouts.'}
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
