import React, { useEffect, useState, useRef } from 'react';
import {
  Play,
  Pause,
  RotateCcw,
  SkipForward,
  TrendingUp,
  TrendingDown,
  BarChart3,
  Calendar,
  Layers,
  Info,
  Clock,
  Sparkles,
} from 'lucide-react';
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
} from 'recharts';
import { api, StockInfo, StockHistoryPoint, SimulationState } from '../services/api';
import { Badge } from '../components/Badge';
import { MetricCard } from '../components/MetricCard';
import { LoadingState } from '../components/LoadingState';

export const MarketExplorerPage: React.FC = () => {
  const [stocks, setStocks] = useState<StockInfo[]>([]);
  const [selectedSymbol, setSelectedSymbol] = useState<string>('RELIANCE');
  const [history, setHistory] = useState<StockHistoryPoint[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [historyLoading, setHistoryLoading] = useState<boolean>(false);

  // Simulation Replay State
  const [simState, setSimState] = useState<SimulationState | null>(null);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [replaySpeed, setReplaySpeed] = useState<number>(1000); // 1s per step
  const playTimerRef = useRef<any>(null);

  // Load stocks list and initial simulation state
  useEffect(() => {
    initData();
    return () => {
      if (playTimerRef.current) clearInterval(playTimerRef.current);
    };
  }, []);

  const initData = async () => {
    try {
      setLoading(true);
      const [stocksData, stateData] = await Promise.all([
        api.getMarketStocks(),
        api.getSimulationState(),
      ]);
      setStocks(stocksData);
      setSimState(stateData);
      if (stocksData.length > 0) {
        loadHistory(stocksData[0].symbol);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const loadHistory = async (symbol: string) => {
    try {
      setHistoryLoading(true);
      setSelectedSymbol(symbol);
      const data = await api.getMarketHistory(symbol);
      setHistory(data);
    } catch (err) {
      console.error(err);
    } finally {
      setHistoryLoading(false);
    }
  };

  // Step simulation forward
  const handleStep = async () => {
    try {
      const res = await api.stepSimulation();
      if (res && res.state) {
        setSimState(res.state);
        if (res.is_finished) {
          setIsPlaying(false);
          if (playTimerRef.current) clearInterval(playTimerRef.current);
        }
      }
    } catch (err) {
      console.error('Step error', err);
      setIsPlaying(false);
      if (playTimerRef.current) clearInterval(playTimerRef.current);
    }
  };

  // Reset simulation
  const handleReset = async (split: string = 'holdout') => {
    setIsPlaying(false);
    if (playTimerRef.current) clearInterval(playTimerRef.current);
    try {
      const state = await api.resetSimulation(split);
      setSimState(state);
    } catch (err) {
      console.error(err);
    }
  };

  // Toggle play/pause
  const togglePlay = () => {
    if (isPlaying) {
      setIsPlaying(false);
      if (playTimerRef.current) clearInterval(playTimerRef.current);
    } else {
      setIsPlaying(true);
      playTimerRef.current = setInterval(() => {
        handleStep();
      }, replaySpeed);
    }
  };

  // Update speed
  const handleSpeedChange = (ms: number) => {
    setReplaySpeed(ms);
    if (isPlaying) {
      if (playTimerRef.current) clearInterval(playTimerRef.current);
      playTimerRef.current = setInterval(() => {
        handleStep();
      }, ms);
    }
  };

  if (loading) {
    return <LoadingState message="Loading Market Explorer..." subMessage="Fetching stock observations and order replay state" />;
  }

  const currentStock = stocks.find((s) => s.symbol === selectedSymbol) || stocks[0];
  const lastHistoryPoint = history[history.length - 1];

  // Current replay progress
  const progressPercent = simState && simState.total_steps > 0
    ? Math.min(100, Math.round((simState.current_step / simState.total_steps) * 100))
    : 0;

  return (
    <div className="space-y-6">
      {/* Top Controller Bar: Stock Selector & Replay Controls */}
      <div className="bg-ivory p-5 rounded-xl border border-borderBase shadow-card space-y-4">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          {/* Stock selector pills */}
          <div>
            <div className="text-[11px] font-bold uppercase tracking-wider text-charcoal-muted mb-2">
              Select NSE Stock ({stocks.length} Available)
            </div>
            <div className="flex flex-wrap gap-1.5">
              {stocks.map((stock) => {
                const isSelected = stock.symbol === selectedSymbol;
                const isGain = stock.daily_return_pct >= 0;
                return (
                  <button
                    key={stock.symbol}
                    onClick={() => loadHistory(stock.symbol)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all flex items-center space-x-1.5 ${
                      isSelected
                        ? 'bg-olive-500 text-ivory shadow-sm'
                        : 'bg-cream-100 hover:bg-cream-200 text-charcoal border border-borderBase'
                    }`}
                  >
                    <span>{stock.symbol}</span>
                    <span
                      className={`text-[10px] font-mono ${
                        isSelected
                          ? 'text-olive-100'
                          : isGain
                          ? 'text-gain'
                          : 'text-loss'
                      }`}
                    >
                      {isGain ? '+' : ''}
                      {stock.daily_return_pct.toFixed(1)}%
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Interactive Replay Controls */}
          <div className="bg-cream-50 p-3 rounded-lg border border-borderBase flex flex-col sm:flex-row sm:items-center gap-3">
            <div className="flex items-center space-x-2">
              <button
                onClick={togglePlay}
                className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-colors ${
                  isPlaying
                    ? 'bg-amber-600 hover:bg-amber-700 text-ivory'
                    : 'bg-olive-600 hover:bg-olive-700 text-ivory'
                }`}
              >
                {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                <span>{isPlaying ? 'Pause' : 'Play Replay'}</span>
              </button>

              <button
                onClick={handleStep}
                disabled={isPlaying}
                title="Advance 1 trading day"
                className="flex items-center space-x-1 px-2.5 py-1.5 bg-ivory hover:bg-cream-200 border border-borderBase rounded-lg text-xs font-semibold text-charcoal disabled:opacity-40"
              >
                <SkipForward className="w-3.5 h-3.5" />
                <span>Next Day</span>
              </button>

              <button
                onClick={() => handleReset('holdout')}
                title="Reset simulation replay"
                className="p-1.5 bg-ivory hover:bg-cream-200 border border-borderBase rounded-lg text-charcoal-muted hover:text-charcoal"
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </button>
            </div>

            {/* Speed selectors */}
            <div className="flex items-center space-x-1 pl-2 border-l border-cream-200 text-xs">
              <span className="text-[10px] text-charcoal-muted font-medium">Speed:</span>
              {[
                { label: '1x', ms: 1200 },
                { label: '2x', ms: 600 },
                { label: '5x', ms: 250 },
              ].map((sp) => (
                <button
                  key={sp.label}
                  onClick={() => handleSpeedChange(sp.ms)}
                  className={`px-1.5 py-0.5 rounded text-[10px] font-mono font-medium ${
                    replaySpeed === sp.ms
                      ? 'bg-olive-500 text-ivory font-bold'
                      : 'text-charcoal-muted hover:bg-cream-200'
                  }`}
                >
                  {sp.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Timeline Progress Bar */}
        <div className="pt-2 border-t border-cream-200 flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-xs">
          <div className="flex items-center space-x-2 text-charcoal-muted">
            <Clock className="w-3.5 h-3.5 text-olive-600" />
            <span>Replay Session:</span>
            <span className="font-mono font-bold text-olive-900 bg-cream-200 px-2 py-0.5 rounded border border-borderBase">
              {simState?.current_date || '2017-05-29'}
            </span>
            <span className="text-[11px] font-mono">
              Step {simState?.current_step || 0} / {simState?.total_steps || 149} ({progressPercent}%)
            </span>
          </div>

          <div className="w-full sm:w-64 bg-cream-200 rounded-full h-2 overflow-hidden border border-borderBase">
            <div
              className="bg-olive-500 h-full rounded-full transition-all duration-300"
              style={{ width: `${progressPercent}%` }}
            ></div>
          </div>
        </div>
      </div>

      {/* Selected Stock Metrics Row */}
      {currentStock && (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          <MetricCard
            label="Latest Close"
            value={`₹${currentStock.latest_close.toFixed(2)}`}
            subValue={`Open: ₹${currentStock.open.toFixed(2)}`}
          />
          <MetricCard
            label="Daily Return"
            value={`${currentStock.daily_return_pct >= 0 ? '+' : ''}${currentStock.daily_return_pct.toFixed(2)}%`}
            subValue={`Prev: ₹${currentStock.previous_close.toFixed(2)}`}
            trend={currentStock.daily_return_pct >= 0 ? 'up' : 'down'}
          />
          <MetricCard
            label="20-Day Volatility"
            value={`${currentStock.volatility_20d_pct.toFixed(1)}%`}
            subValue="Annualized std dev"
          />
          <MetricCard
            label="Median Turnover"
            value={`₹${(currentStock.median_turnover / 1e7).toFixed(1)} Cr`}
            subValue="Daily liquidity proxy"
          />
          <MetricCard
            label="Median Volume"
            value={`${(currentStock.median_volume / 1000).toFixed(0)}k`}
            subValue="Shares traded/day"
          />
          <MetricCard
            label="2-Year Return"
            value={`${currentStock.total_return_pct >= 0 ? '+' : ''}${currentStock.total_return_pct.toFixed(1)}%`}
            subValue="Jan 2016 - Dec 2017"
            trend={currentStock.total_return_pct >= 0 ? 'up' : 'down'}
          />
        </div>
      )}

      {/* Main Stock Chart: Price + 20-Day Fundamental Anchor + Volume Bar Chart */}
      <div className="bg-ivory p-6 rounded-xl border border-borderBase shadow-card space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-cream-200 pb-3">
          <div>
            <h2 className="text-sm font-bold text-charcoal flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-olive-600" />
              <span>{selectedSymbol} — Historical Price & 20-Day Fundamental Anchor</span>
            </h2>
            <p className="text-xs text-charcoal-muted mt-0.5">
              The Stackelberg Leader utilizes the 20-day rolling mean as its fundamental anchor to identify mean-reversion mispricings (±1%).
            </p>
          </div>

          <div className="flex items-center space-x-3 text-xs">
            <span className="flex items-center gap-1.5 font-mono text-[11px]">
              <span className="w-2.5 h-2.5 rounded-full bg-[#606B3F]"></span> Close Price
            </span>
            <span className="flex items-center gap-1.5 font-mono text-[11px]">
              <span className="w-2.5 h-2.5 rounded-full bg-[#A7AE8A]"></span> 20d Fundamental Anchor
            </span>
            <span className="flex items-center gap-1.5 font-mono text-[11px]">
              <span className="w-2.5 h-2.5 rounded-sm bg-[#CCD5B5]"></span> Volume (Scaled)
            </span>
          </div>
        </div>

        <div className="h-[380px] w-full pt-2">
          {historyLoading ? (
            <div className="h-full flex items-center justify-center text-xs text-charcoal-muted">
              Loading historical chart...
            </div>
          ) : history.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={history} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E5E1D3" vertical={false} />
                <XAxis
                  dataKey="date"
                  tick={{ fontSize: 11, fill: '#77796D' }}
                  tickLine={false}
                  stroke="#E5E1D3"
                  minTickGap={30}
                />
                <YAxis
                  yAxisId="price"
                  domain={['auto', 'auto']}
                  tick={{ fontSize: 11, fill: '#77796D' }}
                  tickLine={false}
                  stroke="#E5E1D3"
                  tickFormatter={(val) => `₹${val.toFixed(0)}`}
                />
                <YAxis
                  yAxisId="volume"
                  orientation="right"
                  domain={[0, 'dataMax * 3.5']}
                  hide={true}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#FFFDF7',
                    borderColor: '#E5E1D3',
                    borderRadius: '8px',
                    fontSize: '12px',
                    boxShadow: '0 4px 12px rgba(0,0,0,0.06)',
                  }}
                  formatter={(val: any, name: any) => {
                    if (name === 'Close Price') return [`₹${Number(val).toFixed(2)}`, name];
                    if (name === 'Fundamental Anchor') return [`₹${Number(val).toFixed(2)}`, name];
                    if (name === 'Volume') return [`${Number(val).toLocaleString()} shares`, name];
                    return [val, name];
                  }}
                />
                <Bar
                  yAxisId="volume"
                  dataKey="volume"
                  name="Volume"
                  fill="#CCD5B5"
                  opacity={0.6}
                  radius={[2, 2, 0, 0]}
                />
                <Line
                  yAxisId="price"
                  type="monotone"
                  dataKey="close"
                  name="Close Price"
                  stroke="#606B3F"
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  yAxisId="price"
                  type="monotone"
                  dataKey="fundamental_value"
                  name="Fundamental Anchor"
                  stroke="#A7AE8A"
                  strokeWidth={1.8}
                  strokeDasharray="4 4"
                  dot={false}
                />
              </ComposedChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-full flex items-center justify-center text-xs text-charcoal-muted">
              No historical price data loaded.
            </div>
          )}
        </div>

        {/* Transparency note */}
        <div className="p-3 rounded-lg bg-cream-50 border border-borderBase text-xs text-charcoal-muted flex items-start gap-2">
          <Info className="w-4 h-4 text-olive-600 flex-shrink-0 mt-0.5" />
          <span>
            <strong>Data Disclosure:</strong> This replay uses verified daily closing and volume records from the National Stock Exchange of India (NSE) from 2016 through 2017. Intraday price execution is simulated at the next day's open with execution friction (half-spread 1.0% + quadratic volume impact). Daily OHLCV figures do not represent live tick-by-tick order book depth.
          </span>
        </div>
      </div>
    </div>
  );
};
