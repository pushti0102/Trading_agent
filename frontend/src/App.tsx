import React, { useState, useEffect } from 'react';
import { Sidebar, NavTab } from './components/Sidebar';
import { Header } from './components/Header';
import { OverviewPage } from './pages/OverviewPage';
import { MarketExplorerPage } from './pages/MarketExplorerPage';
import { AgentsPage } from './pages/AgentsPage';
import { StackelbergPage } from './pages/StackelbergPage';
import { TrainingPage } from './pages/TrainingPage';
import { EvaluationPage } from './pages/EvaluationPage';
import { TradesPage } from './pages/TradesPage';
import { SettingsPage } from './pages/SettingsPage';
import { api, SimulationState } from './services/api';
import { AlertCircle } from 'lucide-react';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<NavTab>('overview');
  const [simState, setSimState] = useState<SimulationState | null>(null);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealthSilently, 5000);
    return () => clearInterval(interval);
  }, []);

  const checkHealth = async () => {
    try {
      setIsRefreshing(true);
      await api.getHealth();
      const state = await api.getSimulationState();
      setSimState(state);
      setBackendOnline(true);
    } catch (err) {
      setBackendOnline(false);
    } finally {
      setIsRefreshing(false);
    }
  };

  const checkHealthSilently = async () => {
    try {
      await api.getHealth();
      const state = await api.getSimulationState();
      setSimState(state);
      setBackendOnline(true);
    } catch {
      setBackendOnline(false);
    }
  };

  return (
    <div className="flex min-h-screen bg-cream-100 text-charcoal">
      {/* Persistent Left Sidebar */}
      <Sidebar
        activeTab={activeTab}
        onTabChange={(tab) => setActiveTab(tab)}
        simulationActive={backendOnline === true}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top Header */}
        <Header
          currentDate={simState?.current_date || '2017-05-29'}
          totalDates={495}
          currentStep={simState?.current_step}
          periodText="2016-01-01 — 2017-12-29"
          simulationStatus={
            backendOnline === false
              ? 'Backend Disconnected'
              : simState?.is_finished
              ? 'Replay Complete'
              : 'Interactive Ready'
          }
          onRefresh={checkHealth}
          isRefreshing={isRefreshing}
        />

        {/* Backend Offline Warning Banner if API cannot be reached */}
        {backendOnline === false && (
          <div className="bg-amber-100 border-b border-amber-300 px-6 py-2.5 text-xs text-amber-900 flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <AlertCircle className="w-4 h-4 text-amber-700 flex-shrink-0" />
              <span>
                <strong>FastAPI Backend Unavailable:</strong> The web dashboard is attempting to connect to{' '}
                <code className="bg-amber-200 px-1 py-0.5 rounded font-mono text-[11px]">http://localhost:8000</code>.
                Make sure the backend server is running via <code className="bg-amber-200 px-1 py-0.5 rounded font-mono text-[11px]">uvicorn src.api.main:app --port 8000</code>.
              </span>
            </div>
            <button
              onClick={checkHealth}
              className="px-2.5 py-1 bg-amber-200 hover:bg-amber-300 rounded font-semibold text-[11px] transition-colors"
            >
              Retry
            </button>
          </div>
        )}

        {/* Dynamic Page View */}
        <main className="flex-1 p-6 max-w-7xl w-full mx-auto">
          {activeTab === 'overview' && <OverviewPage onNavigate={setActiveTab} />}
          {activeTab === 'market' && <MarketExplorerPage />}
          {activeTab === 'agents' && <AgentsPage />}
          {activeTab === 'stackelberg' && <StackelbergPage />}
          {activeTab === 'training' && <TrainingPage onNavigate={setActiveTab} />}
          {activeTab === 'evaluation' && <EvaluationPage />}
          {activeTab === 'trades' && <TradesPage />}
          {activeTab === 'settings' && <SettingsPage />}
        </main>
      </div>
    </div>
  );
};

export default App;
