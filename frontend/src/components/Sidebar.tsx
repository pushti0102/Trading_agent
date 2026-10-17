import React from 'react';
import {
  LayoutDashboard,
  CandlestickChart,
  Users,
  Network,
  Cpu,
  LineChart,
  History,
  Sliders,
  ShieldCheck,
  Building2,
} from 'lucide-react';

export type NavTab =
  | 'overview'
  | 'market'
  | 'agents'
  | 'stackelberg'
  | 'training'
  | 'evaluation'
  | 'trades'
  | 'settings';

interface SidebarProps {
  activeTab: NavTab;
  onTabChange: (tab: NavTab) => void;
  simulationActive?: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  onTabChange,
  simulationActive = true,
}) => {
  const navItems: Array<{ id: NavTab; label: string; icon: React.ReactNode; badge?: string }> = [
    { id: 'overview', label: 'Overview', icon: <LayoutDashboard className="w-4 h-4" /> },
    { id: 'market', label: 'Market Explorer', icon: <CandlestickChart className="w-4 h-4" /> },
    { id: 'agents', label: 'Trading Agents', icon: <Users className="w-4 h-4" /> },
    { id: 'stackelberg', label: 'Stackelberg Game', icon: <Network className="w-4 h-4" />, badge: 'Core' },
    { id: 'training', label: 'RL Training', icon: <Cpu className="w-4 h-4" /> },
    { id: 'evaluation', label: 'Backtesting & Evaluation', icon: <LineChart className="w-4 h-4" /> },
    { id: 'trades', label: 'Trade History', icon: <History className="w-4 h-4" /> },
    { id: 'settings', label: 'Settings', icon: <Sliders className="w-4 h-4" /> },
  ];

  return (
    <aside className="w-64 bg-cream-50 border-r border-borderBase flex flex-col justify-between h-screen sticky top-0 select-none">
      {/* Brand Header */}
      <div>
        <div className="p-6 border-b border-borderBase">
          <div className="flex items-center space-x-3">
            <div className="w-9 h-9 rounded-lg bg-olive-700 flex items-center justify-center text-ivory font-bold shadow-sm">
              <span className="font-mono text-base tracking-tighter">OT</span>
            </div>
            <div>
              <div className="flex items-center space-x-1.5">
                <span className="font-bold text-base text-charcoal tracking-tight">OliveTrade</span>
                <span className="text-[10px] font-semibold px-1.5 py-0.2 rounded bg-olive-100 text-olive-800 border border-olive-200 uppercase">
                  RL
                </span>
              </div>
              <p className="text-[11px] text-charcoal-muted leading-tight mt-0.5">
                Stackelberg Simulator • NSE
              </p>
            </div>
          </div>
        </div>

        {/* Navigation list */}
        <nav className="p-3 space-y-1">
          <div className="px-3 py-1.5 text-[10px] font-bold uppercase tracking-wider text-charcoal-muted">
            Platform Modules
          </div>
          {navItems.map((item) => {
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onTabChange(item.id)}
                className={`w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-xs font-medium transition-all ${
                  isActive
                    ? 'bg-olive-500 text-ivory shadow-sm font-semibold'
                    : 'text-charcoal hover:bg-cream-200 hover:text-charcoal'
                }`}
              >
                <div className="flex items-center space-x-3">
                  <span className={isActive ? 'text-ivory' : 'text-olive-700'}>
                    {item.icon}
                  </span>
                  <span>{item.label}</span>
                </div>
                {item.badge && (
                  <span
                    className={`text-[10px] px-1.5 py-0.5 rounded font-mono ${
                      isActive
                        ? 'bg-olive-600 text-ivory border border-olive-400'
                        : 'bg-olive-100 text-olive-800 border border-olive-200'
                    }`}
                  >
                    {item.badge}
                  </span>
                )}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Terminal Metadata Footer */}
      <div className="p-4 border-t border-borderBase space-y-3">
        <div className="bg-ivory rounded-lg p-3 border border-borderBase text-[11px] space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-charcoal-muted flex items-center gap-1.5">
              <Building2 className="w-3.5 h-3.5 text-olive-600" />
              Dataset
            </span>
            <span className="font-semibold text-charcoal">NSE Top 10</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-charcoal-muted flex items-center gap-1.5">
              <ShieldCheck className="w-3.5 h-3.5 text-olive-600" />
              Engine
            </span>
            <span className="font-semibold text-charcoal">Friction & Impact</span>
          </div>
          <div className="flex items-center justify-between pt-1 border-t border-cream-200">
            <span className="text-charcoal-muted">Sim Status</span>
            <span className="flex items-center gap-1 text-olive-700 font-medium">
              <span className="w-1.5 h-1.5 rounded-full bg-olive-500 animate-pulse"></span>
              {simulationActive ? 'Online / Ready' : 'Idle'}
            </span>
          </div>
        </div>

        <div className="text-[10px] text-charcoal-muted text-center leading-relaxed">
          B.Tech Final Project • Department of CSE
        </div>
      </div>
    </aside>
  );
};
