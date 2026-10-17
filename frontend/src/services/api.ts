// OliveTrade API Service Layer

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface StockInfo {
  symbol: string;
  latest_close: number;
  open: number;
  high: number;
  low: number;
  previous_close: number;
  daily_return_pct: number;
  total_return_pct: number;
  median_volume: number;
  median_turnover: number;
  volatility_20d_pct: number;
  data_points: number;
}

export interface StockHistoryPoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  turnover: number;
  return_pct: number;
  volatility: number;
  fundamental_value: number;
}

export interface AgentMetric {
  agent: string;
  start_date: string;
  end_date: string;
  initial_value: number;
  final_value: number;
  total_return_pct: number;
  annualized_volatility_pct: number;
  sharpe_ratio: number;
  max_drawdown_pct: number;
  executed_orders: number;
  fees_paid: number;
}

export interface EvaluationSummaryResponse {
  split: string;
  test_dates_count: number;
  start_date: string;
  end_date: string;
  initial_cash: number;
  assumptions: {
    transaction_cost_pct: number;
    quoted_spread_pct: number;
    impact_coefficient: number;
    max_order_size: number;
    cash_reserve_pct: number;
    max_position_pct: number;
  };
  summary: AgentMetric[];
}

export interface EvaluationHistoryPoint {
  date: string;
  rl_trader?: number;
  leader?: number;
  momentum?: number;
  buy_hold?: number;
}

export interface TrainingEpisode {
  episode: number;
  final_portfolio_value: number;
  total_reward: number;
  mean_loss: number;
  executed_orders: number;
  rejected_orders: number;
  return_pct: number;
  epsilon: number;
}

export interface TrainingStatusResponse {
  status: 'idle' | 'running' | 'completed' | 'error' | 'stopped';
  progress: {
    current_episode: number;
    total_episodes: number;
    message: string;
  };
  logs: string[];
  model: {
    exists: boolean;
    path: string;
    size_bytes: number;
    last_modified: string | null;
  };
  episodes: TrainingEpisode[];
  train_dates_count: number;
  train_start_date: string;
  train_end_date: string;
}

export interface TradeRecord {
  status: 'EXECUTED' | 'REJECTED' | 'SKIPPED' | 'HOLD';
  agent: string;
  symbol: string;
  side: 'BUY' | 'SELL' | 'HOLD';
  quantity: number;
  reference_price: number;
  execution_price: number;
  trade_value: number;
  fee: number;
  spread_rate?: number;
  impact_rate?: number;
  date: string;
  reason?: string;
}

export interface TradesResponse {
  total_count: number;
  page: number;
  page_size: number;
  total_pages: number;
  items: TradeRecord[];
}

export interface StackelbergAnalysisResponse {
  game_type: string;
  leader_profile: {
    name: string;
    role: string;
    objective: string;
    rules: string;
  };
  follower_profile: {
    name: string;
    role: string;
    objective: string;
    rules: string;
  };
  action_matrix: Record<string, number>;
  sample_decisions: Array<{
    date: string;
    symbol: string;
    price: number;
    fundamental_value: number;
    valuation_gap_pct: number;
    market_return_pct: number;
    leader_action: string;
    leader_quantity: number;
    follower_action: string;
    follower_quantity: number;
    leader_rationale: string;
    follower_rationale: string;
  }>;
}

export interface SimulationAgentState {
  cash: number;
  portfolio_value: number;
  return_pct: number;
  inventory: Record<string, number>;
}

export interface SimulationState {
  active: boolean;
  current_step: number;
  total_steps: number;
  current_date: string;
  is_finished: boolean;
  agents: Record<string, SimulationAgentState>;
  market_prices: Record<string, number>;
  history: Array<{
    date: string;
    rl_trader_value?: number;
    leader_value?: number;
    momentum_value?: number;
    buy_hold_value?: number;
    rl_trader_cash?: number;
    leader_cash?: number;
    momentum_cash?: number;
    buy_hold_cash?: number;
  }>;
}

export interface OverviewResponse {
  dataset_period: {
    start: string;
    end: string;
    total_dates: number;
    holdout_start: string;
    holdout_end: string;
    holdout_dates: number;
  };
  best_strategy: AgentMetric | null;
  agents_summary: Record<string, AgentMetric | null>;
  leaderboard: AgentMetric[];
  recent_trades: TradeRecord[];
  market_snapshot: StockInfo[];
  assumptions: Record<string, any>;
}

export const api = {
  getHealth: async () => {
    const res = await fetch(`${API_BASE}/api/health`);
    if (!res.ok) throw new Error('Failed to fetch health');
    return res.json();
  },

  getOverview: async (): Promise<OverviewResponse> => {
    const res = await fetch(`${API_BASE}/api/overview`);
    if (!res.ok) throw new Error('Failed to fetch overview data');
    return res.json();
  },

  getMarketStocks: async (): Promise<StockInfo[]> => {
    const res = await fetch(`${API_BASE}/api/market/stocks`);
    if (!res.ok) throw new Error('Failed to fetch stock list');
    return res.json();
  },

  getMarketHistory: async (symbol: string, startDate?: string, endDate?: string): Promise<StockHistoryPoint[]> => {
    const params = new URLSearchParams({ symbol });
    if (startDate) params.append('start_date', startDate);
    if (endDate) params.append('end_date', endDate);
    const res = await fetch(`${API_BASE}/api/market/history?${params.toString()}`);
    if (!res.ok) throw new Error('Failed to fetch stock history');
    return res.json();
  },

  getAgents: async () => {
    const res = await fetch(`${API_BASE}/api/agents`);
    if (!res.ok) throw new Error('Failed to fetch agents data');
    return res.json();
  },

  getStackelbergAnalysis: async (): Promise<StackelbergAnalysisResponse> => {
    const res = await fetch(`${API_BASE}/api/stackelberg/analysis`);
    if (!res.ok) throw new Error('Failed to fetch Stackelberg game analysis');
    return res.json();
  },

  getTrades: async (params: {
    agent?: string;
    symbol?: string;
    side?: string;
    status?: string;
    search?: string;
    page?: number;
    page_size?: number;
  }): Promise<TradesResponse> => {
    const query = new URLSearchParams();
    if (params.agent && params.agent !== 'all') query.append('agent', params.agent);
    if (params.symbol && params.symbol !== 'all') query.append('symbol', params.symbol);
    if (params.side && params.side !== 'all') query.append('side', params.side);
    if (params.status && params.status !== 'all') query.append('status', params.status);
    if (params.search) query.append('search', params.search);
    if (params.page) query.append('page', params.page.toString());
    if (params.page_size) query.append('page_size', params.page_size.toString());

    const res = await fetch(`${API_BASE}/api/trades?${query.toString()}`);
    if (!res.ok) throw new Error('Failed to fetch trade history');
    return res.json();
  },

  getEvaluationSummary: async (): Promise<EvaluationSummaryResponse> => {
    const res = await fetch(`${API_BASE}/api/evaluation/summary`);
    if (!res.ok) throw new Error('Failed to fetch evaluation summary');
    return res.json();
  },

  getEvaluationHistory: async (): Promise<EvaluationHistoryPoint[]> => {
    const res = await fetch(`${API_BASE}/api/evaluation/history`);
    if (!res.ok) throw new Error('Failed to fetch evaluation history');
    return res.json();
  },

  runEvaluation: async (): Promise<EvaluationSummaryResponse> => {
    const res = await fetch(`${API_BASE}/api/evaluation/run`, { method: 'POST' });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Failed to run evaluation');
    }
    return res.json();
  },

  getTrainingStatus: async (): Promise<TrainingStatusResponse> => {
    const res = await fetch(`${API_BASE}/api/training/status`);
    if (!res.ok) throw new Error('Failed to fetch training status');
    return res.json();
  },

  startTraining: async (episodes: number = 10) => {
    const res = await fetch(`${API_BASE}/api/training/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ episodes }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Failed to start training');
    }
    return res.json();
  },

  stopTraining: async () => {
    const res = await fetch(`${API_BASE}/api/training/stop`, { method: 'POST' });
    if (!res.ok) throw new Error('Failed to stop training');
    return res.json();
  },

  getSimulationState: async (): Promise<SimulationState> => {
    const res = await fetch(`${API_BASE}/api/simulation/state`);
    if (!res.ok) throw new Error('Failed to fetch simulation state');
    return res.json();
  },

  resetSimulation: async (split: string = 'holdout'): Promise<SimulationState> => {
    const res = await fetch(`${API_BASE}/api/simulation/reset`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ split }),
    });
    if (!res.ok) throw new Error('Failed to reset simulation');
    return res.json();
  },

  stepSimulation: async () => {
    const res = await fetch(`${API_BASE}/api/simulation/step`, { method: 'POST' });
    if (!res.ok) throw new Error('Failed to step simulation');
    return res.json();
  },

  getExportTradesUrl: () => `${API_BASE}/api/export/trades.csv`,
  getExportEvaluationUrl: () => `${API_BASE}/api/export/evaluation.csv`,
};
