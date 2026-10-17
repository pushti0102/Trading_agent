import React, { useEffect, useState } from 'react';
import {
  History,
  Download,
  Search,
  Filter,
  ChevronLeft,
  ChevronRight,
  ShieldAlert,
  ArrowUpDown,
} from 'lucide-react';
import { api, TradeRecord, TradesResponse } from '../services/api';
import { Badge } from '../components/Badge';
import { LoadingState } from '../components/LoadingState';
import { EmptyState } from '../components/EmptyState';

export const TradesPage: React.FC = () => {
  const [data, setData] = useState<TradesResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  // Filters state
  const [selectedAgent, setSelectedAgent] = useState<string>('all');
  const [selectedSymbol, setSelectedSymbol] = useState<string>('all');
  const [selectedSide, setSelectedSide] = useState<string>('all');
  const [selectedStatus, setSelectedStatus] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [page, setPage] = useState<number>(1);
  const pageSize = 50;

  useEffect(() => {
    loadTrades();
  }, [selectedAgent, selectedSymbol, selectedSide, selectedStatus, page]);

  const loadTrades = async () => {
    try {
      setLoading(true);
      const res = await api.getTrades({
        agent: selectedAgent,
        symbol: selectedSymbol,
        side: selectedSide,
        status: selectedStatus,
        search: searchQuery,
        page,
        page_size: pageSize,
      });
      setData(res);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setPage(1);
    loadTrades();
  };

  const handleResetFilters = () => {
    setSelectedAgent('all');
    setSelectedSymbol('all');
    setSelectedSide('all');
    setSelectedStatus('all');
    setSearchQuery('');
    setPage(1);
  };

  const stocks = [
    'ICICIBANK',
    'INFY',
    'ITC',
    'MARUTI',
    'RELIANCE',
    'SBIN',
    'SUNPHARMA',
    'TATAMOTORS',
    'TATASTEEL',
    'TCS',
  ];

  return (
    <div className="space-y-6">
      {/* Title & Export */}
      <div className="bg-ivory p-6 rounded-xl border border-borderBase shadow-card flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <h1 className="text-xl font-bold text-charcoal tracking-tight">Simulated Trade History</h1>
            <Badge variant="olive">Execution Audit Log</Badge>
          </div>
          <p className="text-xs text-charcoal-muted mt-1 leading-relaxed">
            Complete transaction ledger across all trading sessions. Each execution accounts for reference price slippage, half-spread adjustment, quadratic price impact, and transaction fees.
          </p>
        </div>

        <a
          href={api.getExportTradesUrl()}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center space-x-1.5 px-3 py-2 bg-cream-100 hover:bg-cream-200 text-charcoal border border-borderBase rounded-lg text-xs font-semibold transition-colors self-start sm:self-auto"
        >
          <Download className="w-3.5 h-3.5 text-olive-700" />
          <span>Export All Trades (CSV)</span>
        </a>
      </div>

      {/* Filter & Search Bar */}
      <div className="bg-ivory p-5 rounded-xl border border-borderBase shadow-card space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2 text-xs font-bold text-charcoal">
            <Filter className="w-4 h-4 text-olive-600" />
            <span>Filter Transactions</span>
          </div>

          <button
            onClick={handleResetFilters}
            className="text-xs text-olive-700 hover:text-olive-900 font-semibold"
          >
            Reset Filters
          </button>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-5 gap-3">
          {/* Agent Filter */}
          <div>
            <label className="block text-[11px] font-semibold text-charcoal-muted mb-1">Agent Policy</label>
            <select
              value={selectedAgent}
              onChange={(e) => { setSelectedAgent(e.target.value); setPage(1); }}
              className="w-full px-2.5 py-1.5 text-xs bg-cream-50 border border-borderBase rounded-lg text-charcoal focus:outline-none focus:border-olive-500"
            >
              <option value="all">All Agents</option>
              <option value="rl_trader">RL Trader (DQN)</option>
              <option value="leader">Stackelberg Leader</option>
              <option value="momentum">Momentum Follower</option>
              <option value="buy_hold">Buy & Hold</option>
            </select>
          </div>

          {/* Stock Filter */}
          <div>
            <label className="block text-[11px] font-semibold text-charcoal-muted mb-1">NSE Symbol</label>
            <select
              value={selectedSymbol}
              onChange={(e) => { setSelectedSymbol(e.target.value); setPage(1); }}
              className="w-full px-2.5 py-1.5 text-xs bg-cream-50 border border-borderBase rounded-lg text-charcoal focus:outline-none focus:border-olive-500 font-mono"
            >
              <option value="all">All Stocks (10)</option>
              {stocks.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </div>

          {/* Side Filter */}
          <div>
            <label className="block text-[11px] font-semibold text-charcoal-muted mb-1">Order Side</label>
            <select
              value={selectedSide}
              onChange={(e) => { setSelectedSide(e.target.value); setPage(1); }}
              className="w-full px-2.5 py-1.5 text-xs bg-cream-50 border border-borderBase rounded-lg text-charcoal focus:outline-none focus:border-olive-500"
            >
              <option value="all">All Sides</option>
              <option value="BUY">BUY Only</option>
              <option value="SELL">SELL Only</option>
            </select>
          </div>

          {/* Status Filter */}
          <div>
            <label className="block text-[11px] font-semibold text-charcoal-muted mb-1">Status</label>
            <select
              value={selectedStatus}
              onChange={(e) => { setSelectedStatus(e.target.value); setPage(1); }}
              className="w-full px-2.5 py-1.5 text-xs bg-cream-50 border border-borderBase rounded-lg text-charcoal focus:outline-none focus:border-olive-500"
            >
              <option value="all">All Statuses</option>
              <option value="EXECUTED">EXECUTED</option>
              <option value="REJECTED">REJECTED</option>
            </select>
          </div>

          {/* Search Form */}
          <div>
            <label className="block text-[11px] font-semibold text-charcoal-muted mb-1">Keyword Search</label>
            <form onSubmit={handleSearchSubmit} className="relative">
              <input
                type="text"
                placeholder="Search symbol, agent..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-8 pr-2.5 py-1.5 text-xs bg-cream-50 border border-borderBase rounded-lg text-charcoal placeholder-charcoal-muted/60 focus:outline-none focus:border-olive-500"
              />
              <Search className="w-3.5 h-3.5 text-charcoal-muted absolute left-2.5 top-2.5" />
            </form>
          </div>
        </div>
      </div>

      {/* Trades Table */}
      <div className="bg-ivory rounded-xl border border-borderBase shadow-card overflow-hidden">
        {loading ? (
          <LoadingState message="Loading trade records..." subMessage="Querying execution database" />
        ) : !data || data.items.length === 0 ? (
          <EmptyState
            title="No Matching Trades Found"
            description="No trade records match the active filters. Try adjusting filters or reset to view all."
            action={{ label: 'Reset Filters', onClick: handleResetFilters }}
          />
        ) : (
          <div>
            <div className="p-4 bg-cream-50 border-b border-borderBase flex items-center justify-between text-xs">
              <div className="font-semibold text-charcoal">
                Showing <span className="font-mono text-olive-800">{data.items.length}</span> of{' '}
                <span className="font-mono text-olive-800">{data.total_count}</span> recorded transactions
              </div>

              {/* Pagination controls */}
              <div className="flex items-center space-x-2">
                <span className="text-[11px] text-charcoal-muted font-mono">
                  Page {data.page} of {data.total_pages}
                </span>
                <button
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  disabled={data.page <= 1}
                  className="p-1 bg-ivory border border-borderBase rounded text-charcoal disabled:opacity-40"
                >
                  <ChevronLeft className="w-3.5 h-3.5" />
                </button>
                <button
                  onClick={() => setPage((p) => Math.min(data.total_pages, p + 1))}
                  disabled={data.page >= data.total_pages}
                  className="p-1 bg-ivory border border-borderBase rounded text-charcoal disabled:opacity-40"
                >
                  <ChevronRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-borderBase bg-cream-100 text-[11px] uppercase tracking-wider text-charcoal-muted font-semibold">
                    <th className="py-2.5 px-3">Date</th>
                    <th className="py-2.5 px-3">Agent</th>
                    <th className="py-2.5 px-3">Symbol</th>
                    <th className="py-2.5 px-3">Side</th>
                    <th className="py-2.5 px-3 text-right">Quantity</th>
                    <th className="py-2.5 px-3 text-right">Ref Price</th>
                    <th className="py-2.5 px-3 text-right">Exec Price</th>
                    <th className="py-2.5 px-3 text-right">Value (₹)</th>
                    <th className="py-2.5 px-3 text-right">Fee (₹)</th>
                    <th className="py-2.5 px-3 text-center">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-cream-200">
                  {data.items.map((trade, idx) => {
                    const isBuy = trade.side === 'BUY';
                    const isExecuted = trade.status === 'EXECUTED';

                    return (
                      <tr key={idx} className="hover:bg-cream-50 transition-colors font-mono">
                        <td className="py-2.5 px-3 text-[11px] text-charcoal-muted">{trade.date}</td>
                        <td className="py-2.5 px-3 font-sans font-medium text-charcoal capitalize">
                          {trade.agent.replace('_', ' ')}
                        </td>
                        <td className="py-2.5 px-3 font-bold text-charcoal">{trade.symbol}</td>
                        <td className="py-2.5 px-3">
                          <span
                            className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                              isBuy
                                ? 'bg-gain-bg text-gain border border-gain-border'
                                : 'bg-loss-bg text-loss border border-loss-border'
                            }`}
                          >
                            {trade.side}
                          </span>
                        </td>
                        <td className="py-2.5 px-3 text-right text-charcoal font-semibold">{trade.quantity}</td>
                        <td className="py-2.5 px-3 text-right text-charcoal-muted">
                          ₹{trade.reference_price ? trade.reference_price.toFixed(2) : '—'}
                        </td>
                        <td className="py-2.5 px-3 text-right font-bold text-charcoal">
                          ₹{trade.execution_price ? trade.execution_price.toFixed(2) : '—'}
                        </td>
                        <td className="py-2.5 px-3 text-right text-charcoal">
                          ₹{trade.trade_value ? trade.trade_value.toLocaleString('en-IN', { maximumFractionDigits: 1 }) : '—'}
                        </td>
                        <td className="py-2.5 px-3 text-right text-charcoal-muted">
                          ₹{trade.fee ? trade.fee.toFixed(2) : '0.00'}
                        </td>
                        <td className="py-2.5 px-3 text-center">
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-sans font-bold ${
                              isExecuted
                                ? 'bg-olive-100 text-olive-800 border border-olive-200'
                                : 'bg-amber-100 text-amber-800 border border-amber-200'
                            }`}
                          >
                            {trade.status}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Bottom Pagination */}
            <div className="p-4 bg-cream-50 border-t border-borderBase flex items-center justify-between text-xs">
              <span className="text-charcoal-muted font-mono text-[11px]">
                Showing {Math.min(data.total_count, (data.page - 1) * pageSize + 1)} –{' '}
                {Math.min(data.total_count, data.page * pageSize)} of {data.total_count} records
              </span>

              <div className="flex items-center space-x-2">
                <button
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  disabled={data.page <= 1}
                  className="px-2.5 py-1 bg-ivory border border-borderBase rounded text-xs font-semibold text-charcoal hover:bg-cream-200 disabled:opacity-40"
                >
                  Previous
                </button>
                <button
                  onClick={() => setPage((p) => Math.min(data.total_pages, p + 1))}
                  disabled={data.page >= data.total_pages}
                  className="px-2.5 py-1 bg-ivory border border-borderBase rounded text-xs font-semibold text-charcoal hover:bg-cream-200 disabled:opacity-40"
                >
                  Next
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
