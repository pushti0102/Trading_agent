import React, { useEffect, useState } from 'react';
import {
  Network,
  ArrowRight,
  ShieldCheck,
  CheckCircle2,
  Clock,
  Layers,
  Info,
  GitCommit,
  BarChart2,
} from 'lucide-react';
import { api, StackelbergAnalysisResponse } from '../services/api';
import { Badge } from '../components/Badge';
import { LoadingState } from '../components/LoadingState';

export const StackelbergPage: React.FC = () => {
  const [data, setData] = useState<StackelbergAnalysisResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [filterAction, setFilterAction] = useState<string>('ALL');

  useEffect(() => {
    loadGameData();
  }, []);

  const loadGameData = async () => {
    try {
      setLoading(true);
      const res = await api.getStackelbergAnalysis();
      setData(res);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <LoadingState message="Loading Stackelberg Game analysis..." subMessage="Computing leader-follower decision matrices and empirical action frequencies" />;
  }

  if (!data) return null;

  const matrix = data.action_matrix || {};
  const totalPairDecisions = Object.values(matrix).reduce((acc, curr) => acc + curr, 0);

  const decisions = (data.sample_decisions || []).filter((d) => {
    if (filterAction === 'ALL') return true;
    if (filterAction === 'BUY') return d.leader_action === 'BUY' || d.follower_action === 'BUY';
    if (filterAction === 'SELL') return d.leader_action === 'SELL' || d.follower_action === 'SELL';
    return true;
  });

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="bg-ivory p-6 rounded-xl border border-borderBase shadow-card">
        <div className="flex items-center space-x-2">
          <h1 className="text-xl font-bold text-charcoal tracking-tight">Stackelberg Leader–Follower Interaction</h1>
          <Badge variant="olive">Sequential Game Dynamics</Badge>
        </div>
        <p className="text-xs text-charcoal-muted mt-1 leading-relaxed max-w-3xl">
          Visualizing the asymmetric hierarchical game formulation. The Leader (fundamental mean-reversion) commits an order first. The Follower (momentum trader) observes the proposed action and recent price trend before executing a response.
        </p>
      </div>

      {/* Sequential Interaction Flow Diagram (5 Steps) */}
      <div className="bg-ivory p-6 rounded-xl border border-borderBase shadow-card space-y-4">
        <h2 className="text-sm font-bold text-charcoal flex items-center gap-2 border-b border-cream-200 pb-3">
          <GitCommit className="w-4 h-4 text-olive-600" />
          5-Stage Sequential Game Workflow
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-3 pt-2">
          {/* Step 1 */}
          <div className="p-4 rounded-lg bg-cream-50 border border-borderBase flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="w-6 h-6 rounded-full bg-olive-700 text-ivory font-bold text-xs flex items-center justify-center font-mono">
                  1
                </span>
                <span className="text-[10px] font-bold text-charcoal-muted uppercase">Observation</span>
              </div>
              <h3 className="text-xs font-bold text-charcoal mb-1">State Evaluation</h3>
              <p className="text-[11px] text-charcoal-muted leading-relaxed">
                Leader observes current stock price <span className="font-mono text-olive-800">Close(t)</span> and 20-day rolling fundamental mean <span className="font-mono text-olive-800">FV(t)</span>.
              </p>
            </div>
            <div className="mt-3 pt-2 border-t border-cream-200 text-[10px] font-mono text-olive-700">
              Gap = P(t) / FV(t) - 1
            </div>
          </div>

          {/* Step 2 */}
          <div className="p-4 rounded-lg bg-cream-50 border border-borderBase flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="w-6 h-6 rounded-full bg-olive-700 text-ivory font-bold text-xs flex items-center justify-center font-mono">
                  2
                </span>
                <span className="text-[10px] font-bold text-charcoal-muted uppercase">Commitment</span>
              </div>
              <h3 className="text-xs font-bold text-charcoal mb-1">Leader Action</h3>
              <p className="text-[11px] text-charcoal-muted leading-relaxed">
                Leader commits action: BUY if <span className="font-mono">P &lt; 0.99*FV</span>; SELL if <span className="font-mono">P &gt; 1.01*FV</span>; else HOLD.
              </p>
            </div>
            <div className="mt-3 pt-2 border-t border-cream-200 text-[10px] font-mono text-olive-700">
              Action = [BUY | HOLD | SELL]
            </div>
          </div>

          {/* Step 3 */}
          <div className="p-4 rounded-lg bg-cream-50 border border-borderBase flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="w-6 h-6 rounded-full bg-olive-700 text-ivory font-bold text-xs flex items-center justify-center font-mono">
                  3
                </span>
                <span className="text-[10px] font-bold text-charcoal-muted uppercase">Follower</span>
              </div>
              <h3 className="text-xs font-bold text-charcoal mb-1">Follower Reaction</h3>
              <p className="text-[11px] text-charcoal-muted leading-relaxed">
                Follower observes Leader's action + 1-day Return. Follows BUY if return &ge; 0; follows SELL if held; or trades &plusmn;1% momentum breakouts.
              </p>
            </div>
            <div className="mt-3 pt-2 border-t border-cream-200 text-[10px] font-mono text-olive-700">
              Response = f(a_L, Ret)
            </div>
          </div>

          {/* Step 4 */}
          <div className="p-4 rounded-lg bg-cream-50 border border-borderBase flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="w-6 h-6 rounded-full bg-olive-700 text-ivory font-bold text-xs flex items-center justify-center font-mono">
                  4
                </span>
                <span className="text-[10px] font-bold text-charcoal-muted uppercase">Execution</span>
              </div>
              <h3 className="text-xs font-bold text-charcoal mb-1">Market Friction</h3>
              <p className="text-[11px] text-charcoal-muted leading-relaxed">
                Orders fill at Open(t+1) with 2% spread + price impact <span className="font-mono">(0.0005 * Qty)</span> + 0.1% transaction fee.
              </p>
            </div>
            <div className="mt-3 pt-2 border-t border-cream-200 text-[10px] font-mono text-olive-700">
              P_exec = P_ref * (1 ± Adj)
            </div>
          </div>

          {/* Step 5 */}
          <div className="p-4 rounded-lg bg-cream-50 border border-borderBase flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="w-6 h-6 rounded-full bg-olive-700 text-ivory font-bold text-xs flex items-center justify-center font-mono">
                  5
                </span>
                <span className="text-[10px] font-bold text-charcoal-muted uppercase">Accounting</span>
              </div>
              <h3 className="text-xs font-bold text-charcoal mb-1">Mark-to-Market</h3>
              <p className="text-[11px] text-charcoal-muted leading-relaxed">
                Portfolios are marked at Close(t+1). Inventories, cash balances, and agent rewards are synchronously updated.
              </p>
            </div>
            <div className="mt-3 pt-2 border-t border-cream-200 text-[10px] font-mono text-olive-700">
              Value = Cash + Σ (Q * Close)
            </div>
          </div>
        </div>
      </div>

      {/* Empirical Action Frequency Matrix (3x3 Grid) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-5 bg-ivory p-6 rounded-xl border border-borderBase shadow-card space-y-4">
          <div>
            <h2 className="text-sm font-bold text-charcoal">Empirical Action Profile Matrix</h2>
            <p className="text-xs text-charcoal-muted mt-0.5">
              Observed joint action distributions across {totalPairDecisions} simulated stock-day decisions
            </p>
          </div>

          {/* 3x3 Matrix Grid */}
          <div className="border border-borderBase rounded-lg overflow-hidden text-xs">
            <div className="grid grid-cols-4 bg-cream-200 font-semibold text-charcoal p-2 border-b border-borderBase text-center text-[11px]">
              <div>Leader \ Follower</div>
              <div className="text-gain">BUY</div>
              <div className="text-charcoal-muted">HOLD</div>
              <div className="text-loss">SELL</div>
            </div>

            {['BUY', 'HOLD', 'SELL'].map((lAction) => (
              <div key={lAction} className="grid grid-cols-4 border-b border-cream-200 last:border-0 text-center font-mono">
                <div className="bg-cream-100 p-3 font-bold text-charcoal flex items-center justify-center">
                  {lAction}
                </div>
                {['BUY', 'HOLD', 'SELL'].map((fAction) => {
                  const key = `${lAction}_${fAction}`;
                  const count = matrix[key] || 0;
                  const pct = totalPairDecisions > 0 ? ((count / totalPairDecisions) * 100).toFixed(1) : '0.0';
                  return (
                    <div key={fAction} className="p-3 bg-ivory hover:bg-cream-50 transition-colors flex flex-col justify-center">
                      <span className="font-bold text-charcoal">{count}</span>
                      <span className="text-[10px] text-charcoal-muted font-normal">{pct}%</span>
                    </div>
                  );
                })}
              </div>
            ))}
          </div>

          <div className="p-3 rounded-lg bg-cream-50 border border-borderBase text-xs text-charcoal-muted">
            <span className="font-semibold text-charcoal">Observation:</span> The majority of steps are{' '}
            <span className="font-mono text-olive-800 font-semibold">HOLD-HOLD</span> when prices reside within the ±1% fundamental band. Simultaneous{' '}
            <span className="font-mono text-olive-800 font-semibold">BUY-BUY</span> occurs when the Leader signals undervalued entry and recent momentum is non-negative.
          </div>
        </div>

        {/* Verified Action Log Table */}
        <div className="lg:col-span-7 bg-ivory p-6 rounded-xl border border-borderBase shadow-card flex flex-col justify-between">
          <div>
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-cream-200 pb-3 mb-4">
              <div>
                <h3 className="text-sm font-bold text-charcoal">Actual Leader-Follower Decisions Log</h3>
                <p className="text-xs text-charcoal-muted mt-0.5">Real outputs from the backend StackelbergGame implementation</p>
              </div>

              <div className="flex items-center space-x-1.5 text-xs">
                {['ALL', 'BUY', 'SELL'].map((f) => (
                  <button
                    key={f}
                    onClick={() => setFilterAction(f)}
                    className={`px-2.5 py-1 rounded text-xs font-semibold ${
                      filterAction === f
                        ? 'bg-olive-500 text-ivory'
                        : 'bg-cream-100 text-charcoal hover:bg-cream-200 border border-borderBase'
                    }`}
                  >
                    {f}
                  </button>
                ))}
              </div>
            </div>

            <div className="overflow-x-auto max-h-[360px] overflow-y-auto">
              <table className="w-full text-left text-xs">
                <thead className="sticky top-0 bg-cream-100 border-b border-borderBase text-[11px] uppercase tracking-wider text-charcoal-muted font-semibold">
                  <tr>
                    <th className="py-2 px-3">Date</th>
                    <th className="py-2 px-3">Symbol</th>
                    <th className="py-2 px-3 text-right">Price</th>
                    <th className="py-2 px-3 text-right">FV</th>
                    <th className="py-2 px-3 text-center">Gap %</th>
                    <th className="py-2 px-3 text-center">Leader</th>
                    <th className="py-2 px-3 text-center">Follower</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-cream-200">
                  {decisions.map((d, i) => (
                    <tr key={i} className="hover:bg-cream-50 transition-colors">
                      <td className="py-2.5 px-3 font-mono text-[11px] text-charcoal-muted">{d.date}</td>
                      <td className="py-2.5 px-3 font-bold text-charcoal">{d.symbol}</td>
                      <td className="py-2.5 px-3 text-right font-mono text-charcoal">₹{d.price.toFixed(1)}</td>
                      <td className="py-2.5 px-3 text-right font-mono text-charcoal-muted">₹{d.fundamental_value.toFixed(1)}</td>
                      <td className="py-2.5 px-3 text-center font-mono">
                        <span className={d.valuation_gap_pct < -1 ? 'text-gain font-semibold' : d.valuation_gap_pct > 1 ? 'text-loss font-semibold' : 'text-charcoal-muted'}>
                          {d.valuation_gap_pct > 0 ? '+' : ''}{d.valuation_gap_pct.toFixed(1)}%
                        </span>
                      </td>
                      <td className="py-2.5 px-3 text-center">
                        <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold ${
                          d.leader_action === 'BUY' ? 'bg-gain-bg text-gain border border-gain-border' :
                          d.leader_action === 'SELL' ? 'bg-loss-bg text-loss border border-loss-border' :
                          'bg-cream-200 text-charcoal-muted'
                        }`}>
                          {d.leader_action}
                        </span>
                      </td>
                      <td className="py-2.5 px-3 text-center">
                        <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold ${
                          d.follower_action === 'BUY' ? 'bg-gain-bg text-gain border border-gain-border' :
                          d.follower_action === 'SELL' ? 'bg-loss-bg text-loss border border-loss-border' :
                          'bg-cream-200 text-charcoal-muted'
                        }`}>
                          {d.follower_action}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-cream-200 text-[11px] text-charcoal-muted">
            All actions logged directly from <span className="font-mono text-olive-800">StackelbergGame.leader_action</span> and <span className="font-mono text-olive-800">follower_action</span>.
          </div>
        </div>
      </div>
    </div>
  );
};
