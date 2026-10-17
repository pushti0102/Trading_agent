import React, { useState } from 'react';
import {
  Sliders,
  RotateCcw,
  ShieldCheck,
  CheckCircle2,
  Database,
  Building2,
  Cpu,
  BookOpen,
  Info,
} from 'lucide-react';
import { api } from '../services/api';
import { Badge } from '../components/Badge';

export const SettingsPage: React.FC = () => {
  const [resetMsg, setResetMsg] = useState<string | null>(null);
  const [isResetting, setIsResetting] = useState<boolean>(false);

  const handleReset = async (split: string) => {
    try {
      setIsResetting(true);
      setResetMsg(null);
      await api.resetSimulation(split);
      setResetMsg(`Interactive session successfully reset to ${split} partition start.`);
    } catch (err: any) {
      setResetMsg(`Error: ${err.message || 'Failed to reset simulation'}`);
    } finally {
      setIsResetting(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="bg-ivory p-6 rounded-xl border border-borderBase shadow-card">
        <div className="flex items-center space-x-2">
          <h1 className="text-xl font-bold text-charcoal tracking-tight">System Configuration & Parameters</h1>
          <Badge variant="olive">Simulation Engine v1.0</Badge>
        </div>
        <p className="text-xs text-charcoal-muted mt-1 leading-relaxed max-w-3xl">
          Review the core financial parameters, execution friction models, dataset scope, and academic constraints governing the Stackelberg Trading Simulator.
        </p>
      </div>

      {resetMsg && (
        <div className="p-4 rounded-lg bg-gain-bg border border-gain-border text-xs text-gain flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 flex-shrink-0" />
          <span>{resetMsg}</span>
        </div>
      )}

      {/* Engine Parameters Review */}
      <div className="bg-ivory p-6 rounded-xl border border-borderBase shadow-card space-y-4">
        <div className="flex items-center justify-between border-b border-cream-200 pb-3">
          <div>
            <h2 className="text-sm font-bold text-charcoal flex items-center gap-2">
              <Sliders className="w-4 h-4 text-olive-600" />
              Simulated Execution Engine Parameters
            </h2>
            <p className="text-xs text-charcoal-muted mt-0.5">
              Hardcoded institutional friction applied uniformly across all trading agents
            </p>
          </div>
          <Badge variant="sage">Fixed Specifications</Badge>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 text-xs">
          <div className="p-4 rounded-lg bg-cream-50 border border-borderBase space-y-1">
            <span className="text-[10px] uppercase font-bold text-charcoal-muted">Initial Capital</span>
            <div className="text-base font-bold font-mono text-charcoal">₹100,000.00</div>
            <p className="text-[11px] text-charcoal-muted">Base cash endowment provided to each agent policy</p>
          </div>

          <div className="p-4 rounded-lg bg-cream-50 border border-borderBase space-y-1">
            <span className="text-[10px] uppercase font-bold text-charcoal-muted">Transaction Cost Rate</span>
            <div className="text-base font-bold font-mono text-charcoal">0.10% (0.001)</div>
            <p className="text-[11px] text-charcoal-muted">Percentage fee levied on gross trade value per order</p>
          </div>

          <div className="p-4 rounded-lg bg-cream-50 border border-borderBase space-y-1">
            <span className="text-[10px] uppercase font-bold text-charcoal-muted">Quoted Bid-Ask Spread</span>
            <div className="text-base font-bold font-mono text-charcoal">2.00% (0.02)</div>
            <p className="text-[11px] text-charcoal-muted">Half-spread (1.0%) charged against reference prices</p>
          </div>

          <div className="p-4 rounded-lg bg-cream-50 border border-borderBase space-y-1">
            <span className="text-[10px] uppercase font-bold text-charcoal-muted">Market Impact Coefficient</span>
            <div className="text-base font-bold font-mono text-charcoal">0.0005 / share</div>
            <p className="text-[11px] text-charcoal-muted">Temporary linear slippage penalty proportional to order size</p>
          </div>

          <div className="p-4 rounded-lg bg-cream-50 border border-borderBase space-y-1">
            <span className="text-[10px] uppercase font-bold text-charcoal-muted">Maximum Order Size</span>
            <div className="text-base font-bold font-mono text-charcoal">100 Shares</div>
            <p className="text-[11px] text-charcoal-muted">Maximum single-order batch limit per equity</p>
          </div>

          <div className="p-4 rounded-lg bg-cream-50 border border-borderBase space-y-1">
            <span className="text-[10px] uppercase font-bold text-charcoal-muted">Cash Reserve Floor</span>
            <div className="text-base font-bold font-mono text-charcoal">5.0% (₹5,000)</div>
            <p className="text-[11px] text-charcoal-muted">Mandatory liquidity buffer preventing total cash exhaustion</p>
          </div>

          <div className="p-4 rounded-lg bg-cream-50 border border-borderBase space-y-1">
            <span className="text-[10px] uppercase font-bold text-charcoal-muted">Max Position Exposure</span>
            <div className="text-base font-bold font-mono text-charcoal">20.0% of Portfolio</div>
            <p className="text-[11px] text-charcoal-muted">Diversification cap on single equity concentration</p>
          </div>

          <div className="p-4 rounded-lg bg-cream-50 border border-borderBase space-y-1">
            <span className="text-[10px] uppercase font-bold text-charcoal-muted">Volatility Rolling Window</span>
            <div className="text-base font-bold font-mono text-charcoal">20 Trading Days</div>
            <p className="text-[11px] text-charcoal-muted">Rolling standard deviation window for state normalization</p>
          </div>

          <div className="p-4 rounded-lg bg-cream-50 border border-borderBase space-y-1">
            <span className="text-[10px] uppercase font-bold text-charcoal-muted">Fundamental Anchor Window</span>
            <div className="text-base font-bold font-mono text-charcoal">20 Trading Days</div>
            <p className="text-[11px] text-charcoal-muted">Rolling close mean serving as Stackelberg Leader anchor</p>
          </div>
        </div>
      </div>

      {/* Dataset Scope & Interactive Reset */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Dataset Details */}
        <div className="bg-ivory p-6 rounded-xl border border-borderBase shadow-card space-y-3">
          <div className="flex items-center justify-between border-b border-cream-200 pb-3">
            <h3 className="text-sm font-bold text-charcoal flex items-center gap-2">
              <Database className="w-4 h-4 text-olive-600" />
              Dataset Specifications
            </h3>
            <Badge variant="olive">NSE Selected 10</Badge>
          </div>

          <div className="space-y-2 text-xs">
            <div className="flex items-center justify-between py-1.5 border-b border-cream-200">
              <span className="text-charcoal-muted">Market / Exchange:</span>
              <span className="font-semibold text-charcoal">National Stock Exchange of India (NSE)</span>
            </div>
            <div className="flex items-center justify-between py-1.5 border-b border-cream-200">
              <span className="text-charcoal-muted">Selected Equities:</span>
              <span className="font-mono font-semibold text-charcoal">10 High-Turnover Constituents</span>
            </div>
            <div className="flex items-center justify-between py-1.5 border-b border-cream-200">
              <span className="text-charcoal-muted">Total Trading Dates:</span>
              <span className="font-mono font-semibold text-charcoal">495 Daily Sessions</span>
            </div>
            <div className="flex items-center justify-between py-1.5 border-b border-cream-200">
              <span className="text-charcoal-muted">Historical Span:</span>
              <span className="font-mono font-semibold text-charcoal">2016-01-01 through 2017-12-29</span>
            </div>
            <div className="flex items-center justify-between py-1.5">
              <span className="text-charcoal-muted">Constituents:</span>
              <span className="font-mono text-[11px] text-olive-900 font-semibold">
                ICICIBANK, INFY, ITC, MARUTI, RELIANCE, SBIN, SUNPHARMA, TATAMOTORS, TATASTEEL, TCS
              </span>
            </div>
          </div>
        </div>

        {/* Interactive Simulation Session Management */}
        <div className="bg-ivory p-6 rounded-xl border border-borderBase shadow-card space-y-3 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between border-b border-cream-200 pb-3">
              <h3 className="text-sm font-bold text-charcoal flex items-center gap-2">
                <RotateCcw className="w-4 h-4 text-olive-600" />
                Session Controls
              </h3>
              <Badge variant="neutral">Environment</Badge>
            </div>

            <p className="text-xs text-charcoal-muted mt-2 leading-relaxed">
              Reset the interactive live stepper environment to either the start of the holdout test period or the beginning of the full historical dataset.
            </p>

            <div className="space-y-2.5 mt-4">
              <button
                onClick={() => handleReset('holdout')}
                disabled={isResetting}
                className="w-full flex items-center justify-center space-x-2 px-4 py-2.5 bg-olive-500 hover:bg-olive-600 text-ivory rounded-lg text-xs font-bold shadow-sm transition-colors disabled:opacity-50"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                <span>Reset to Holdout Period (2017-05-29)</span>
              </button>

              <button
                onClick={() => handleReset('all')}
                disabled={isResetting}
                className="w-full flex items-center justify-center space-x-2 px-4 py-2 bg-cream-200 hover:bg-cream-300 text-charcoal rounded-lg text-xs font-semibold border border-borderBase transition-colors disabled:opacity-50"
              >
                <span>Reset to Full Dataset Start (2016-01-01)</span>
              </button>
            </div>
          </div>

          <div className="pt-3 border-t border-cream-200 text-[11px] text-charcoal-muted">
            All interactive stepping occurs in-memory and does not alter evaluation CSVs.
          </div>
        </div>
      </div>

      {/* Project Runbook Reference */}
      <div className="bg-ivory p-6 rounded-xl border border-borderBase shadow-card space-y-3">
        <h3 className="text-sm font-bold text-charcoal flex items-center gap-2">
          <BookOpen className="w-4 h-4 text-olive-600" />
          Academic Runbook & Reproducibility Guidelines
        </h3>
        <p className="text-xs text-charcoal-muted leading-relaxed">
          This system was constructed for academic demonstration of sequential game theory coupled with reinforcement learning.
          To reproduce command-line results independently of this UI, execute:
        </p>

        <div className="bg-charcoal text-cream-100 p-4 rounded-lg font-mono text-xs space-y-1.5 overflow-x-auto">
          <div><span className="text-olive-400"># 1. Train DQN on 70% split</span></div>
          <div>python train_rl.py</div>
          <div className="pt-1"><span className="text-olive-400"># 2. Evaluate model on 30% holdout split</span></div>
          <div>python evaluate_test.py</div>
          <div className="pt-1"><span className="text-olive-400"># 3. Run full-history Stackelberg episode</span></div>
          <div>python run_episode.py</div>
        </div>
      </div>
    </div>
  );
};
