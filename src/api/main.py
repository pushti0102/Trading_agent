from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path
import io
import pandas as pd

from src.api.service import simulator_service

app = FastAPI(
    title="OliveTrade API",
    description="Research-Grade Backend API for Multi-Agent RL Stackelberg Trading Simulator for NSE Stocks",
    version="1.0.0"
)

# Enable CORS for local Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TrainingStartRequest(BaseModel):
    episodes: int = Field(default=10, ge=1, le=50, description="Number of episodes to train")


class ResetSimulationRequest(BaseModel):
    split: str = Field(default="holdout", description="Date split: holdout, train, or all")


@app.get("/api/health")
def get_health():
    return {
        "status": "healthy",
        "service": "OliveTrade Simulator API",
        "version": "1.0.0",
        "dataset": {
            "stocks_count": len(simulator_service.symbols),
            "symbols": simulator_service.symbols,
            "total_trading_dates": len(simulator_service.all_dates),
            "date_range": f"{simulator_service.all_dates[0]} to {simulator_service.all_dates[-1]}",
            "train_dates_count": len(simulator_service.train_dates),
            "holdout_dates_count": len(simulator_service.holdout_dates),
        },
        "model_loaded": simulator_service.get_training_results()["model"]["exists"],
        "training_status": simulator_service.training_status,
        "interactive_simulation_active": simulator_service.interactive_env is not None
    }


@app.get("/api/overview")
def get_overview():
    eval_summary = simulator_service.get_evaluation_summary()
    stocks = simulator_service.get_market_stocks()
    recent_trades = simulator_service.get_trades(page=1, page_size=10)["items"]
    inst_benchmark = simulator_service.get_v2_institutional_benchmark()
    master_benchmark = simulator_service.get_v2_master_benchmark()

    # Calculate overall best return and leaderboard
    summary_list = eval_summary.get("summary", [])
    best_strategy = summary_list[0] if summary_list else None

    # Key metric cards based on the holdout evaluation
    rl_agent = next((s for s in summary_list if s["agent"] == "rl_trader"), None)
    leader_agent = next((s for s in summary_list if s["agent"] == "leader"), None)
    momentum_agent = next((s for s in summary_list if s["agent"] == "momentum"), None)
    buy_hold_agent = next((s for s in summary_list if s["agent"] == "buy_hold"), None)

    ppo_inst = next((s for s in inst_benchmark if "PPO" in s.get("strategy", "") and "0.2%" in s.get("regime", "")), None)
    ensemble_inst = next((s for s in inst_benchmark if "Ensemble" in s.get("strategy", "") and "0.2%" in s.get("regime", "")), None)

    return {
        "dataset_period": {
            "start": simulator_service.all_dates[0],
            "end": simulator_service.all_dates[-1],
            "total_dates": len(simulator_service.all_dates),
            "holdout_start": simulator_service.holdout_dates[0],
            "holdout_end": simulator_service.holdout_dates[-1],
            "holdout_dates": len(simulator_service.holdout_dates),
        },
        "best_strategy": best_strategy,
        "agents_summary": {
            "rl_trader": rl_agent,
            "leader": leader_agent,
            "momentum": momentum_agent,
            "buy_hold": buy_hold_agent,
            "ppo_trader": ppo_inst,
            "ensemble": ensemble_inst,
        },
        "institutional_benchmark": inst_benchmark,
        "master_benchmark": master_benchmark,
        "leaderboard": summary_list,
        "recent_trades": recent_trades,
        "market_snapshot": stocks[:6],
        "assumptions": eval_summary.get("assumptions", {})
    }


@app.get("/api/market/stocks")
def get_market_stocks():
    return simulator_service.get_market_stocks()


@app.get("/api/market/history")
def get_market_history(
    symbol: str = Query(..., description="Stock symbol (e.g., RELIANCE)"),
    start_date: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="End date YYYY-MM-DD")
):
    try:
        return simulator_service.get_market_history(symbol=symbol.upper(), start_date=start_date, end_date=end_date)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/agents")
def get_agents():
    eval_summary = simulator_service.get_evaluation_summary()
    summary_map = {s["agent"]: s for s in eval_summary.get("summary", [])}

    agents_info = [
        {
            "id": "rl_trader",
            "name": "RL Trader (Deep Q-Network)",
            "category": "Reinforcement Learning",
            "description": "Deep Q-Network agent trained on 10 normalized market features including valuation divergence, price return, volatility, turnover liquidity, and inventory exposure. Operates with strict cash reserve & position caps.",
            "policy": "Epsilon-greedy with experience replay (Adam optimizer, SmoothL1Loss, target network sync every 100 steps)",
            "metrics": summary_map.get("rl_trader"),
            "risk_controls": {
                "cash_reserve": "5% (₹5,000)",
                "max_position": "20% of portfolio",
                "max_order_size": "100 shares",
                "action_space": "BUY (5 shares), HOLD, SELL (5 shares)"
            }
        },
        {
            "id": "leader",
            "name": "Stackelberg Leader",
            "category": "Game Theory / First-Mover",
            "description": "Informed institutional mean-reversion leader. Observes 20-day rolling fundamental anchor and commits orders first. Buys undervalued assets (< -1%) and sells overvalued assets (> +1%).",
            "policy": "Sequential commitment: Price < 0.99*FV -> BUY 5; Price > 1.01*FV -> SELL 5; else HOLD",
            "metrics": summary_map.get("leader"),
            "risk_controls": {
                "cash_reserve": "5% (₹5,000)",
                "max_position": "20% of portfolio",
                "max_order_size": "100 shares"
            }
        },
        {
            "id": "momentum",
            "name": "Momentum Follower",
            "category": "Game Theory / Second-Mover",
            "description": "Reactive trend-following follower. Observes the leader's committed action and recent daily return before choosing orders, capturing trend continuation or signaling effects.",
            "policy": "Follow leader BUY if Return >= 0; follow leader SELL if inventory available; otherwise trade ±1% price return breakouts",
            "metrics": summary_map.get("momentum"),
            "risk_controls": {
                "cash_reserve": "5% (₹5,000)",
                "max_position": "20% of portfolio",
                "max_order_size": "100 shares"
            }
        },
        {
            "id": "buy_hold",
            "name": "Buy-and-Hold Benchmark",
            "category": "Passive Baseline",
            "description": "Equal-budget passive portfolio basket allocated on the first evaluation day across all selected stocks, constrained by max order size and cash reserves, held statically without rebalancing.",
            "policy": "Equal-weight basket entry on Day 1, zero rebalancing trades thereafter",
            "metrics": summary_map.get("buy_hold"),
            "risk_controls": {
                "initial_budget_split": "10% per stock across 10 stocks",
                "cash_reserve": "5% (₹5,000)"
            }
        }
    ]

    return {
        "agents": agents_info,
        "assumptions": eval_summary.get("assumptions", {})
    }


@app.get("/api/stackelberg/analysis")
def get_stackelberg_analysis():
    return simulator_service.get_stackelberg_game_analysis()


@app.get("/api/trades")
def get_trades(
    agent: Optional[str] = Query(None, description="Filter by agent id"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    side: Optional[str] = Query(None, description="Filter by side: BUY or SELL"),
    status: Optional[str] = Query(None, description="Filter by status: EXECUTED or REJECTED"),
    search: Optional[str] = Query(None, description="Search keyword"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=200, description="Page size")
):
    return simulator_service.get_trades(
        agent=agent,
        symbol=symbol,
        side=side,
        status=status,
        search=search,
        page=page,
        page_size=page_size
    )


@app.get("/api/evaluation/summary")
def get_evaluation_summary():
    return simulator_service.get_evaluation_summary()


@app.get("/api/evaluation/history")
def get_evaluation_history():
    return simulator_service.get_evaluation_history()


@app.post("/api/evaluation/run")
def run_evaluation():
    try:
        return simulator_service.run_evaluation()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/training/status")
def get_training_status():
    return simulator_service.get_training_results()


@app.post("/api/training/start")
def start_training(req: TrainingStartRequest):
    try:
        simulator_service.start_training(episodes=req.episodes)
        return {"status": "started", "episodes": req.episodes, "message": f"Training initiated for {req.episodes} episodes."}
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/api/training/stop")
def stop_training():
    simulator_service.stop_training()
    return {"status": "stop_requested", "message": "Signal sent to stop training."}


@app.get("/api/simulation/state")
def get_simulation_state():
    return simulator_service.get_interactive_state()


@app.post("/api/simulation/reset")
def reset_simulation(req: ResetSimulationRequest):
    return simulator_service.reset_interactive(split=req.split)


@app.post("/api/simulation/step")
def step_simulation():
    return simulator_service.step_interactive()


@app.get("/api/export/trades.csv")
def export_trades_csv():
    trades = simulator_service.cached_eval_trades
    if not trades:
        raise HTTPException(status_code=404, detail="No trades available for export")
    df = pd.DataFrame(trades)
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    return Response(
        content=stream.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=olivetrade_simulated_trades.csv"}
    )


@app.get("/api/export/evaluation.csv")
def export_evaluation_csv():
    summary = simulator_service.get_evaluation_summary()
    items = summary.get("summary", [])
    if not items:
        raise HTTPException(status_code=404, detail="No evaluation summary available")
    df = pd.DataFrame(items)
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    return Response(
        content=stream.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=olivetrade_evaluation_summary.csv"}
    )


# =============================================================================
# OliveTrade 2.0 Research API Endpoints
# =============================================================================

@app.get("/api/v2/agents")
def get_v2_agents():
    """Return all 15 discrete and continuous agents in OliveTrade 2.0."""
    return {"agents": simulator_service.get_v2_agents()}


@app.get("/api/v2/walk-forward/folds")
def get_v2_walk_forward_folds():
    """Return walk-forward expanding folds, training boundaries, and per-fold performance."""
    return simulator_service.get_v2_walk_forward_folds()


@app.get("/api/v2/walk-forward/summary")
def get_v2_walk_forward_summary():
    """Return aggregate multi-fold walk-forward metrics across all seeds."""
    return simulator_service.get_v2_walk_forward_summary()


@app.get("/api/v2/robustness")
def get_v2_robustness():
    """Return cost sensitivity grids, regime stress tests, and bootstrap confidence intervals."""
    return simulator_service.get_v2_robustness_data()


@app.get("/api/v2/reports/manifest")
def get_v2_reports_manifest():
    """Return status and artifacts manifest for all research phases (Phase 1-8)."""
    return simulator_service.get_v2_reports_manifest()


@app.get("/api/v2/export/robustness.csv")
def export_robustness_csv():
    """Export transaction-cost sensitivity grid results as CSV."""
    r_dir = simulator_service.feed_data
    csv_path = Path("results/olive_trade_v2/phase_07_robustness/cost_sensitivity.csv")
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Robustness cost sensitivity results not found")
    df = pd.read_csv(csv_path)
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    return Response(
        content=stream.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=olivetrade_cost_sensitivity.csv"}
    )


@app.get("/api/v2/export/walk-forward.csv")
def export_walk_forward_csv():
    """Export walk-forward aggregate benchmark metrics as CSV."""
    csv_path = Path("results/olive_trade_v2/phase_06_walk_forward/walk_forward_aggregate_summary.csv")
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Walk-forward summary results not found")
    df = pd.read_csv(csv_path)
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    return Response(
        content=stream.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=olivetrade_walk_forward_benchmark.csv"}
    )


@app.get("/api/v2/evaluation/history")
def get_v2_evaluation_history():
    """Return daily portfolio values for all continuous RL, ensemble, and baseline agents."""
    return simulator_service.get_v2_evaluation_history()


@app.get("/api/v2/benchmark")
def get_v2_benchmark():
    """Return 15-model master benchmark dataset."""
    return {"benchmark": simulator_service.get_v2_master_benchmark()}


@app.get("/api/v2/institutional-benchmark")
def get_v2_institutional_benchmark():
    """Return realistic institutional execution regime benchmark."""
    return {"benchmark": simulator_service.get_v2_institutional_benchmark()}


@app.get("/api/v2/export/institutional-benchmark.csv")
def export_institutional_benchmark_csv():
    """Export realistic institutional execution benchmark results as CSV."""
    csv_path = Path("results/olive_trade_v2/phase_08_final_report/institutional_execution_benchmark.csv")
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Institutional benchmark results not found")
    df = pd.read_csv(csv_path)
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    return Response(
        content=stream.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=olivetrade_institutional_benchmark.csv"}
    )


@app.get("/api/v2/export/master-benchmark.csv")
def export_master_benchmark_csv():
    """Export 15-strategy master benchmark results as CSV."""
    csv_path = Path("results/olive_trade_v2/phase_08_final_report/master_strategy_benchmark.csv")
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Master benchmark results not found")
    df = pd.read_csv(csv_path)
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    return Response(
        content=stream.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=olivetrade_master_benchmark.csv"}
    )


@app.get("/api/v2/export/evaluation-history.csv")
def export_v2_evaluation_history_csv():
    """Export 149-day holdout equity curves for all continuous RL and ensemble agents as CSV."""
    csv_path = Path("results/olive_trade_v2/phase_08_final_report/v2_evaluation_history.csv")
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="V2 evaluation history not found")
    df = pd.read_csv(csv_path)
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    return Response(
        content=stream.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=olivetrade_v2_evaluation_history.csv"}
    )



