import React from 'react';
import { Calendar, AlertTriangle, RefreshCw, Activity, Terminal } from 'lucide-react';

interface HeaderProps {
  currentDate?: string;
  totalDates?: number;
  currentStep?: number;
  periodText?: string;
  simulationStatus?: string;
  onRefresh?: () => void;
  isRefreshing?: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  currentDate = '2017-05-29',
  totalDates = 495,
  currentStep,
  periodText = '2016-01-01 — 2017-12-29',
  simulationStatus = 'Active Replay',
  onRefresh,
  isRefreshing = false,
}) => {
  return (
    <header className="h-16 bg-ivory border-b border-borderBase px-6 flex items-center justify-between sticky top-0 z-20 shadow-sm">
      {/* Left: Simulation Period & Date Indicator */}
      <div className="flex items-center space-x-4">
        <div className="flex items-center space-x-2 text-xs text-charcoal">
          <Calendar className="w-4 h-4 text-olive-600" />
          <span className="font-semibold text-charcoal">NSE Historical Period:</span>
          <span className="font-mono text-olive-800 bg-cream-200 px-2 py-0.5 rounded border border-borderBase">
            {periodText}
          </span>
          <span className="text-charcoal-muted font-mono">({totalDates} sessions)</span>
        </div>

        <div className="h-4 w-px bg-borderBase hidden md:block"></div>

        {/* Current Replay Date */}
        <div className="hidden sm:flex items-center space-x-2 text-xs">
          <span className="text-charcoal-muted">Replay Session:</span>
          <span className="font-mono font-bold text-olive-900 bg-olive-100 px-2 py-0.5 rounded border border-olive-200">
            {currentDate}
          </span>
          {currentStep !== undefined && (
            <span className="text-[11px] text-charcoal-muted font-mono">
              [Step {currentStep}]
            </span>
          )}
        </div>
      </div>

      {/* Right: Institutional Simulation Only Badge & Status */}
      <div className="flex items-center space-x-3">
        {/* Status Pill */}
        <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-full bg-cream-100 border border-borderBase text-xs">
          <Activity className="w-3.5 h-3.5 text-olive-600 animate-pulse" />
          <span className="text-charcoal-muted text-[11px]">Mode:</span>
          <span className="font-medium text-olive-800 text-[11px]">{simulationStatus}</span>
        </div>

        {/* MANDATORY SIMULATION ONLY BADGE */}
        <div className="flex items-center space-x-1 px-2.5 py-1 rounded bg-amber-50 border border-amber-300 text-amber-900 text-xs font-semibold tracking-tight shadow-sm">
          <AlertTriangle className="w-3.5 h-3.5 text-amber-700" />
          <span>SIMULATION ONLY</span>
        </div>

        {/* Refresh button */}
        {onRefresh && (
          <button
            onClick={onRefresh}
            disabled={isRefreshing}
            title="Refresh simulation data"
            className="p-1.5 rounded-lg text-charcoal-muted hover:text-charcoal hover:bg-cream-200 border border-borderBase transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 ${isRefreshing ? 'animate-spin text-olive-600' : ''}`} />
          </button>
        )}

        {/* Researcher terminal tag */}
        <div className="hidden lg:flex items-center space-x-1.5 pl-2 border-l border-borderBase text-xs text-charcoal-muted">
          <Terminal className="w-3.5 h-3.5 text-olive-600" />
          <span className="font-mono text-[11px]">v1.0.0</span>
        </div>
      </div>
    </header>
  );
};
