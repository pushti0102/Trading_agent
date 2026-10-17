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
  PieChart as PieIcon,
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
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      setError(null);
      const [overviewData, historyData] = await Promise.all([
        api.getOverview(),
        api.getEvaluationHistory(),
      ]);
      setData(overviewData);
      setHistory(historyData);
    } catch (err: any) {
      setError(err.message || 'Failed to load overview data');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <LoadingState message="Loading simulation overview..." subMessage="Aggregating strategy returns and market metrics" />;
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

  // Identify top agent and RL Trader agent
  const bestAgent = data.best_strategy;
  const rlAgent = data.agents_summary?.rl_trader;
  const leaderAgent = data.agents_summary?.leader;
  const initialCash = 100000;

  return (
    <div className="space-y-6">
      {/* Page Title & Context Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-ivory p-6 rounded-xl border border-borderBase shadow-card">
        <div>
          <div className="flex items-center space-x-2">
            <h1 className="text-xl font-bold text-charcoal tracking-tight">
              Simulation Performance Overview
            </h1>
            <Badge variant="olive">Holdout Test 30%</Badge>
          </div>
          <p className="text-xs text-charcoal-muted mt-1 leading-relaxed max-w-2xl">
            Real performance comparison across RL (DQN), Stackelberg Leader-Follower, and Buy-and-Hold
            strategies executed with simulated bid-ask spread (2%), temporary market impact, and transaction fees (0.1%).
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
            onClick={() => onNavigate('market')}
            className="flex items-center space-x-2 px-3.5 py-2 text-xs font-semibold text-ivory bg-olive-500 hover:bg-olive-600 rounded-lg shadow-sm transition-colors"
          >
            <span>Replay Market</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* KPI Cards Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-6 gap-4">
        <MetricCard
          label="Initial Capital"
          value="₹100,000"
          subValue="Baseline per agent"
          icon={<DollarSign className="w-4 h-4" />}
        />
        <MetricCard
          label="Best Final Value"
          value={bestAgent ? `₹${bestAgent.final_value.toLocaleString('en-IN')}` : '—'}
          subValue={bestAgent ? `Agent: ${bestAgent.agent}` : '—'}
          trend={bestAgent && bestAgent.total_return_pct >= 0 ? 'up' : 'down'}
          trendText={bestAgent ? `${bestAgent.total_return_pct >= 0 ? '+' : ''}${bestAgent.total_return_pct}%` : undefined}
          icon={<Award className="w-4 h-4" />}
        />
        <MetricCard
          label="RL Trader Return"
          value={rlAgent ? `${rlAgent.total_return_pct >= 0 ? '+' : ''}${rlAgent.total_return_pct}%` : '—'}
          subValue={rlAgent ? `Final: ₹${rlAgent.final_value.toLocaleString('en-IN')}` : '—'}
          trend={rlAgent && rlAgent.total_return_pct >= 0 ? 'up' : 'down'}
          trendText={rlAgent ? `Sharpe: ${rlAgent.sharpe_ratio}` : undefined}
          icon={<Activity className="w-4 h-4" />}
        />
        <MetricCard
          label="Leader Return"
          value={leaderAgent ? `${leaderAgent.total_return_pct >= 0 ? '+' : ''}${leaderAgent.total_return_pct}%` : '—'}
          subValue={leaderAgent ? `Final: ₹${leaderAgent.final_value.toLocaleString('en-IN')}` : '—'}
          trend={leaderAgent && leaderAgent.total_return_pct >= 0 ? 'up' : 'down'}
          trendText={leaderAgent ? `Max DD: ${leaderAgent.max_drawdown_pct}%` : undefined}
          icon={<TrendingUp className="w-4 h-4" />}
        />
        <MetricCard
          label="Test Sessions"
          value={data.dataset_period.holdout_dates}
          subValue="Unseen holdout trading days"
          icon={<Layers className="w-4 h-4" />}
        />
        <MetricCard
          label="Simulated Friction"
          value="2.0% Spread"
          subValue="0.1% fees + price impact"
          icon={<Shield className="w-4 h-4" />}
        />
      </div>

      {/* Main Multi-Line Equity Chart */}
      <div className="bg-ivory p-6 rounded-xl border border-borderBase shadow-card space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-cream-200 pb-4">
          <div>
            <h2 className="text-sm font-bold text-charcoal flex items-center gap-2">
              <BarChart2 className="w-4 h-4 text-olive-600" />
              Strategy Portfolio Value Over Time (Holdout Period)
            </h2>
            <p className="text-xs text-charcoal-muted mt-0.5">
              Comparative equity curves for RL Trader, Stackelberg Leader, Momentum Follower, and Buy-and-Hold
            </p>
          </div>
          <div className="flex items-center space-x-3 text-xs">
            <span className="flex items-center gap-1.5 font-mono text-[11px] text-charcoal">
              <span className="w-2.5 h-2.5 rounded-full bg-[#606B3F]"></span> RL Trader
            </span>
            <span className="flex items-center gap-1.5 font-mono text-[11px] text-charcoal">
              <span className="w-2.5 h-2.5 rounded-full bg-[#3F492C]"></span> Leader
            </span>
            <span className="flex items-center gap-1.5 font-mono text-[11px] text-charcoal">
              <span className="w-2.5 h-2.5 rounded-full bg-[#B33927]"></span> Momentum
            </span>
            <span className="flex items-center gap-1.5 font-mono text-[11px] text-charcoal">
              <span className="w-2.5 h-2.5 rounded-full bg-[#77796D]"></span> Buy & Hold
            </span>
          </div>
        </div>

        <div className="h-[340px] w-full pt-2">
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
                    name === 'rl_trader'
                      ? 'RL Trader (DQN)'
                      : name === 'leader'
                      ? 'Stackelberg Leader'
                      : name === 'momentum'
                      ? 'Momentum Follower'
                      : 'Buy & Hold',
                  ]}
                />
                <Line
                  type="monotone"
                  dataKey="rl_trader"
                  name="rl_trader"
                  stroke="#606B3F"
                  strokeWidth={2.2}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="leader"
                  name="leader"
                  stroke="#3F492C"
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="momentum"
                  name="momentum"
                  stroke="#B33927"
                  strokeWidth={1.8}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="buy_hold"
                  name="buy_hold"
                  stroke="#77796D"
                  strokeWidth={1.8}
                  strokeDasharray="4 4"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-full flex items-center justify-center text-xs text-charcoal-muted">
              No holdout history available. Please run evaluation from the Backtesting tab.
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
                <h3 className="text-sm font-bold text-charcoal">Strategy Performance Leaderboard</h3>
                <p className="text-xs text-charcoal-muted mt-0.5">
                  Holdout period ({data.dataset_period.holdout_start} to {data.dataset_period.holdout_end})
                </p>
              </div>
              <Badge variant="sage">{data.leaderboard.length} Strategies</Badge>
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
                    <th className="py-2.5 px-3 text-right">Trades</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-cream-200">
                  {data.leaderboard.map((item, idx) => {
                    const isPositive = item.total_return_pct >= 0;
                    return (
                      <tr key={item.agent} className="hover:bg-cream-50 transition-colors">
                        <td className="py-3 px-3">
                          <div className="flex items-center space-x-2">
                            <span className="font-mono text-[10px] text-charcoal-muted w-4">
                              #{idx + 1}
                            </span>
                            <div>
                              <span className="font-semibold text-charcoal capitalize">
                                {item.agent.replace('_', ' ')}
                              </span>
                              <span className="block text-[10px] text-charcoal-muted">
                                {item.agent === 'rl_trader'
                                  ? 'Deep Q-Network'
                                  : item.agent === 'leader'
                                  ? 'Fundamental Mean-Reversion'
                                  : item.agent === 'momentum'
                                  ? 'Trend & Signal Follower'
                                  : 'Passive Basket'}
                              </span>
                            </div>
                          </div>
                        </td>
                        <td className="py-3 px-3 text-right font-mono font-medium text-charcoal">
                          ₹{item.final_value.toLocaleString('en-IN', { maximumFractionDigits: 2 })}
                        </td>
                        <td className="py-3 px-3 text-right font-mono font-semibold">
                          <span className={isPositive ? 'text-gain' : 'text-loss'}>
                            {isPositive ? '+' : ''}
                            {item.total_return_pct.toFixed(2)}%
                          </span>
                        </td>
                        <td className="py-3 px-3 text-right font-mono text-charcoal">
                          {item.sharpe_ratio.toFixed(2)}
                        </td>
                        <td className="py-3 px-3 text-right font-mono text-loss">
                          {item.max_drawdown_pct.toFixed(2)}%
                        </td>
                        <td className="py-3 px-3 text-right font-mono text-charcoal-muted">
                          {item.executed_orders}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-cream-200 text-[11px] text-charcoal-muted flex items-center justify-between">
            <span>Trading constraints: max order size 100, 5% cash reserve</span>
            <button
              onClick={() => onNavigate('agents')}
              className="text-olive-700 hover:text-olive-900 font-semibold"
            >
              Inspect Agent Specifications &rarr;
            </button>
          </div>
        </div>

        {/* Latest Simulated Orders (5 Cols) */}
        <div className="lg:col-span-5 bg-ivory p-6 rounded-xl border border-borderBase shadow-card flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between border-b border-cream-200 pb-3 mb-4">
              <div>
                <h3 className="text-sm font-bold text-charcoal">Recent Executed Orders</h3>
                <p className="text-xs text-charcoal-muted mt-0.5">Simulated order book transactions</p>
              </div>
              <button
                onClick={() => onNavigate('trades')}
                className="text-xs text-olive-700 hover:text-olive-900 font-semibold"
              >
                View All
              </button>
            </div>

            {data.recent_trades.length > 0 ? (
              <div className="space-y-2.5">
                {data.recent_trades.slice(0, 6).map((trade, idx) => (
                  <div
                    key={idx}
                    className="p-2.5 rounded-lg bg-cream-50 border border-borderBase flex items-center justify-between text-xs"
                  >
                    <div className="flex items-center space-x-2.5">
                      <span
                        className={`font-mono text-[10px] font-bold px-1.5 py-0.5 rounded ${
                          trade.side === 'BUY'
                            ? 'bg-gain-bg text-gain border border-gain-border'
                            : 'bg-loss-bg text-loss border border-loss-border'
                        }`}
                      >
                        {trade.side}
                      </span>
                      <div>
                        <span className="font-bold text-charcoal">{trade.symbol}</span>
                        <span className="text-[10px] text-charcoal-muted block font-mono">
                          {trade.agent} • {trade.date}
                        </span>
                      </div>
                    </div>

                    <div className="text-right font-mono">
                      <div className="font-semibold text-charcoal">
                        {trade.quantity} @ ₹{trade.execution_price.toFixed(2)}
                      </div>
                      <div className="text-[10px] text-charcoal-muted">
                        Fee: ₹{trade.fee.toFixed(2)}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="py-12 text-center text-xs text-charcoal-muted">
                No simulated executions recorded yet.
              </div>
            )}
          </div>

          <div className="mt-4 pt-3 border-t border-cream-200 text-[11px] text-charcoal-muted text-center">
            Transactions incorporate half-spread + impact penalty
          </div>
        </div>
      </div>

      {/* Selected Stocks Replay Prices Snapshot */}
      <div className="bg-ivory p-6 rounded-xl border border-borderBase shadow-card">
        <div className="flex items-center justify-between border-b border-cream-200 pb-3 mb-4">
          <div>
            <h3 className="text-sm font-bold text-charcoal">NSE Constituents Replay Monitor</h3>
            <p className="text-xs text-charcoal-muted mt-0.5">
              Historical prices and liquidity metrics from the curated 10-stock NSE dataset
            </p>
          </div>
          <button
            onClick={() => onNavigate('market')}
            className="text-xs text-olive-700 hover:text-olive-900 font-semibold flex items-center gap-1"
          >
            <span>Open Market Explorer</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
          {data.market_snapshot.map((stock) => {
            const isGain = stock.daily_return_pct >= 0;
            return (
              <div
                key={stock.symbol}
                onClick={() => onNavigate('market')}
                className="p-3.5 rounded-lg bg-cream-50 border border-borderBase hover:border-olive-300 cursor-pointer transition-colors"
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="font-bold text-xs text-charcoal">{stock.symbol}</span>
                  <span
                    className={`text-[10px] font-mono font-semibold ${
                      isGain ? 'text-gain' : 'text-loss'
                    }`}
                  >
                    {isGain ? '+' : ''}
                    {stock.daily_return_pct.toFixed(2)}%
                  </span>
                </div>
                <div className="text-base font-bold font-mono text-charcoal">
                  ₹{stock.latest_close.toFixed(2)}
                </div>
                <div className="mt-2 pt-2 border-t border-cream-200 flex items-center justify-between text-[10px] text-charcoal-muted font-mono">
                  <span>Vol 20d: {stock.volatility_20d_pct.toFixed(1)}%</span>
                  <span>Turnover: ₹{(stock.median_turnover / 1e7).toFixed(1)}Cr</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
