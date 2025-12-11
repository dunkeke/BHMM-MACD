import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings

# 忽略部分 sklearn 警告
warnings.filterwarnings("ignore")

# ==========================================
# PART 1: 策略封装 (Strategy Zoo)
# ==========================================

class StrategyBase:
    """策略基类，统一接口"""
    def generate_signals(self, df):
        raise NotImplementedError

class HMMAdaptiveStrategy(StrategyBase):
    """
    [调优版] 自适应贝叶斯策略
    参数: min_covar=0.01 (防过拟合), threshold=0.0003 (3bps)
    """
    def __init__(self, n_components=3, iter_num=1000, window_size=21, threshold=0.0003):
        self.n_components = n_components
        self.iter_num = iter_num
        self.window_size = window_size
        self.threshold = threshold

    def generate_signals(self, df):
        df = df.copy()
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility'] = df['Log_Ret'].rolling(window=self.window_size).std()
        df.dropna(inplace=True)
        
        if len(df) < 100: return df
        
        X = df[['Log_Ret', 'Volatility']].values * 100.0
        
        try:
            model = GaussianHMM(n_components=self.n_components, covariance_type="full", n_iter=self.iter_num, random_state=42, tol=0.01, min_covar=0.01)
            model.fit(X)
        except: return df
        
        hidden_states = model.predict(X)
        
        # 状态排序
        state_vol_means = [X[hidden_states == i, 1].mean() for i in range(self.n_components)]
        sorted_stats = sorted(list(enumerate(state_vol_means)), key=lambda x: x[1])
        mapping = {old: new for new, (old, _) in enumerate(sorted_stats)}
        
        # 后验概率
        posterior_probs = model.predict_proba(X)
        sorted_probs = np.zeros_like(posterior_probs)
        for old_i, new_i in mapping.items():
            sorted_probs[:, new_i] = posterior_probs[:, old_i]
            
        df['Regime'] = np.array([mapping[s] for s in hidden_states])
        
        for i in range(self.n_components):
            df[f'Prob_S{i}'] = sorted_probs[:, i]
        
        # 贝叶斯期望
        state_means = []
        for i in range(self.n_components):
            mean_ret = df[df['Regime'] == i]['Log_Ret'].mean()
            state_means.append(mean_ret)
        
        new_transmat = np.zeros_like(model.transmat_)
        for i in range(self.n_components):
            for j in range(self.n_components):
                new_transmat[mapping[i], mapping[j]] = model.transmat_[i, j]
                
        next_probs = np.dot(sorted_probs, new_transmat)
        df['Bayes_Exp_Ret'] = np.dot(next_probs, state_means)
        
        # 信号生成 (使用传入的动态阈值)
        df['Signal'] = 0
        df.loc[df['Bayes_Exp_Ret'] > self.threshold, 'Signal'] = 1
        df.loc[df['Bayes_Exp_Ret'] < -self.threshold, 'Signal'] = -1
        
        return df

class HMM_MACD_Strategy(StrategyBase):
    """
    [增强版] HMM + 4H MACD 结构共振
    """
    def __init__(self, n_components=3, iter_num=1000, window_size=21):
        self.n_components = n_components
        self.iter_num = iter_num
        self.window_size = window_size

    def calculate_macd_structure(self, df_prices):
        exp1 = df_prices.ewm(span=12, adjust=False).mean()
        exp2 = df_prices.ewm(span=26, adjust=False).mean()
        dif = exp1 - exp2
        dea = dif.ewm(span=9, adjust=False).mean()
        hist = dif - dea
        dif_mean = dif.rolling(window=60).mean()
        dif_std = dif.rolling(window=60).std()
        dif_z = (dif - dif_mean) / (dif_std + 1e-8)
        return dif, dea, hist, dif_z

    def get_4h_macd_data(self, ticker):
        try:
            df_1h = yf.download(ticker, period="60d", interval="1h", progress=False, auto_adjust=True)
            if isinstance(df_1h.columns, pd.MultiIndex): df_1h.columns = df_1h.columns.get_level_values(0)
            if len(df_1h) < 60: return None
            if df_1h.index.tz is not None: df_1h.index = df_1h.index.tz_localize(None)

            df_4h = df_1h['Close'].resample('4h').ohlc()
            df_4h.dropna(inplace=True)
            dif, dea, hist, dif_z = self.calculate_macd_structure(df_4h['close'])
            
            macd_df = pd.DataFrame({'MACD_Hist_4H': hist, 'MACD_Z_4H': dif_z})
            return macd_df.resample('1D').last().fillna(method='ffill')
        except: return None

    def generate_signals(self, df, ticker_symbol=None):
        if df.index.tz is not None: df.index = df.index.tz_localize(None)

        # 复用基础逻辑
        base_strat = HMMAdaptiveStrategy(self.n_components, self.iter_num, self.window_size)
        df = base_strat.generate_signals(df)
        
        df['Prob_S0_Prior'] = df.get('Prob_S0', 0.33)
        df['Prob_S2_Prior'] = df.get('Prob_S2', 0.33)
        
        if ticker_symbol:
            macd_data = self.get_4h_macd_data(ticker_symbol)
            if macd_data is not None:
                df = df.join(macd_data, how='left').fillna(method='ffill').fillna(0)
            else:
                df['MACD_Hist_4H'] = 0; df['MACD_Z_4H'] = 0
        else:
            df['MACD_Hist_4H'] = 0; df['MACD_Z_4H'] = 0

        # 贝叶斯更新
        macd_norm = np.clip(df['MACD_Hist_4H'] / (df['Close'] * 0.01 + 1e-6), -1, 1) * 2.0
        likelihood_0 = np.exp(macd_norm)
        likelihood_2 = np.exp(-macd_norm)
        
        df['Prob_S0_Post'] = df['Prob_S0_Prior'] * likelihood_0
        df['Prob_S1_Post'] = df.get('Prob_S1', 0.33)
        df['Prob_S2_Post'] = df['Prob_S2_Prior'] * likelihood_2
        
        total = df['Prob_S0_Post'] + df['Prob_S1_Post'] + df['Prob_S2_Post']
        df['Prob_S0_Post'] /= total
        df['Prob_S2_Post'] /= total
        
        # 信号
        df['Signal'] = 0
        df.loc[(df['Prob_S0_Post'] > 0.45) & (df['MACD_Hist_4H'] > 0), 'Signal'] = 1
        df.loc[(df['Prob_S2_Post'] > 0.45) & (df['MACD_Hist_4H'] < 0), 'Signal'] = -1
        
        return df

# ==========================================
# PART 2: 回测引擎 & 鲁棒性测试 (Robustness)
# ==========================================

class BacktestEngine:
    def __init__(self, initial_capital=100000, transaction_cost=0.0002):
        self.initial_capital = initial_capital
        self.cost = transaction_cost

    def run(self, df, ret_col='Log_Ret'):
        df = df.copy()
        df['Position'] = df['Signal'].shift(1).fillna(0)
        trades = df['Position'].diff().abs().fillna(0)
        fees = trades * self.cost
        
        df[ret_col] = df[ret_col].fillna(0)
        df['Strategy_Ret'] = (df['Position'] * df[ret_col]) - fees
        df['Equity_Curve'] = self.initial_capital * (1 + df['Strategy_Ret']).cumprod()
        df['Benchmark_Curve'] = self.initial_capital * (1 + df[ret_col]).cumprod()
        return df

    def get_sharpe(self, df):
        if df.empty or 'Strategy_Ret' not in df.columns: return 0
        ret = df['Strategy_Ret']
        vol = ret.std() * np.sqrt(252)
        return (ret.mean() * 252) / (vol + 1e-8) if vol > 0 else 0

    def calculate_metrics(self, df):
        # ... (保持原有的详细指标计算逻辑)
        if df.empty or 'Equity_Curve' not in df.columns or len(df) < 2: return self._empty_metrics()
        equity = df['Equity_Curve']
        ret = df['Strategy_Ret']
        start_val = equity.iloc[0] if equity.iloc[0] > 0 else self.initial_capital
        total_ret = (equity.iloc[-1] / start_val) - 1
        time_span = df.index[-1] - df.index[0]
        days = time_span.days + (time_span.seconds / 86400)
        cagr = (1 + total_ret) ** (365 / days) - 1 if days > 0.5 else 0
        vol = ret.std() * np.sqrt(252)
        sharpe = (ret.mean() * 252) / (vol + 1e-8) if vol > 0 else 0
        roll_max = equity.cummax()
        dd = (equity - roll_max) / (roll_max + 1e-8)
        max_dd = dd.min()
        active_days = df[df['Position'] != 0]
        win_rate = len(active_days[active_days['Strategy_Ret'] > 0]) / len(active_days) if len(active_days) > 0 else 0
        return {"Total Return": f"{total_ret*100:.2f}%", "CAGR": f"{cagr*100:.2f}%", "Sharpe Ratio": f"{sharpe:.2f}", "Max Drawdown": f"{max_dd*100:.2f}%", "Win Rate": f"{win_rate*100:.1f}%"}
    
    def _empty_metrics(self):
        return {k: "N/A" for k in ["Total Return", "CAGR", "Sharpe Ratio", "Max Drawdown", "Win Rate"]}

# ==========================================
# PART 3: UI 与 通知系统
# ==========================================

def display_alert_system(df):
    """交易信号通知中心"""
    last = df.iloc[-1]
    sig = last['Signal']
    date_str = last.name.strftime('%Y-%m-%d')
    
    # 动态通知
    if sig == 1:
        st.toast(f"🟢 [LONG SIGNAL] {date_str}: 建议做多!", icon="🚀")
        st.success(f"### 🚀 强力做多信号触发 ({date_str})")
        st.markdown("**核心理由**: 宏观低波动体制确认 + 贝叶斯正向期望显著。")
    elif sig == -1:
        st.toast(f"🔴 [SHORT SIGNAL] {date_str}: 建议做空/清仓!", icon="💎")
        st.error(f"### 💎 强力卖出信号触发 ({date_str})")
        st.markdown("**核心理由**: 市场进入高波动崩盘模式，或技术面死叉共振。")
    else:
        st.info(f"### ⏳ 当前建议: 空仓观望 ({date_str})")
        st.markdown("**核心理由**: 市场方向不明或预期收益不足以覆盖成本。")

def run_robustness_test(df_raw, ticker_name):
    """
    参数敏感性压力测试
    扫描: Window Size (15-30) x Threshold (2bps-5bps)
    """
    st.markdown("### 🧪 策略鲁棒性实验室 (Robustness Lab)")
    st.markdown("正在对 **自适应策略** 进行多参数压力测试，寻找夏普比率的 '高原区'...")
    
    windows = range(15, 31, 3) # 15, 18, 21, 24, 27, 30
    thresholds = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]
    
    results = []
    
    progress_bar = st.progress(0)
    total_steps = len(windows) * len(thresholds)
    step = 0
    
    engine = BacktestEngine()
    
    for w in windows:
        for t in thresholds:
            # 动态训练策略
            strat = HMMAdaptiveStrategy(window_size=w, threshold=t)
            df_sig = strat.generate_signals(df_raw)
            if 'Signal' in df_sig.columns:
                df_bt = engine.run(df_sig)
                sharpe = engine.get_sharpe(df_bt)
                results.append({'Window': w, 'Threshold (bps)': t*10000, 'Sharpe': sharpe})
            
            step += 1
            progress_bar.progress(step / total_steps)
            
    # 结果可视化
    res_df = pd.DataFrame(results)
    
    # 绘制热力图
    heatmap_data = res_df.pivot(index='Window', columns='Threshold (bps)', values='Sharpe')
    
    fig = px.imshow(heatmap_data, 
                    labels=dict(x="开仓阈值 (bps)", y="波动率窗口 (Days)", color="Sharpe Ratio"),
                    x=heatmap_data.columns,
                    y=heatmap_data.index,
                    color_continuous_scale='Viridis',
                    text_auto='.2f',
                    title=f"{ticker_name} 参数敏感性热力图")
    
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
    
    avg_sharpe = res_df['Sharpe'].mean()
    st.caption(f"**解读**: 颜色越亮越好。如果亮色区域连成一片(高原)，说明策略稳健；如果是孤立的亮点，说明过拟合。\n**平均夏普**: {avg_sharpe:.2f}")

# ==========================================
# PART 4: Main UI
# ==========================================

st.set_page_config(page_title="Energy Quant Pro+", layout="wide", page_icon="⚡")
st.title("⚡ Energy Quant System: Pro Max")

# 侧边栏
st.sidebar.header("⚙️ 控制台")
mode = st.sidebar.radio("工作模式", ["策略回测与信号", "鲁棒性压力测试"])
strategy_type = st.sidebar.selectbox("策略内核", ["HMM + 4H MACD 共振", "HMM 自适应贝叶斯"])
asset = st.sidebar.selectbox("交易标的", ["Brent Crude", "WTI Crude", "Natural Gas", "Dutch TTF"])

tickers = {"Brent Crude": "BZ=F", "WTI Crude": "CL=F", "Natural Gas": "NG=F", "Dutch TTF": "TTF=F"}
ticker = tickers[asset]

start_date = st.sidebar.date_input("开始日期", datetime.now() - timedelta(days=365*2))
end_date = st.sidebar.date_input("结束日期", datetime.now())

if st.sidebar.button("🚀 启动引擎", type="primary"):
    with st.spinner("正在进行量化计算..."):
        try:
            # 1. 获取数据
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            if df.empty:
                st.error("数据获取失败。")
            else:
                # --- 分流逻辑 ---
                if mode == "鲁棒性压力测试":
                    run_robustness_test(df, asset)
                    
                else: # 策略回测模式
                    # 2. 策略执行
                    if "MACD" in strategy_type:
                        strat = HMM_MACD_Strategy()
                        # MACD 需要 ticker symbol 来获取额外的高频数据
                        df_res = strat.generate_signals(df, ticker_symbol=ticker)
                    else:
                        strat = HMMAdaptiveStrategy()
                        df_res = strat.generate_signals(df)
                    
                    if 'Signal' in df_res.columns:
                        # 3. 信号通知 (置顶)
                        display_alert_system(df_res)
                        st.divider()
                        
                        # 4. 回测计算
                        engine = BacktestEngine(initial_capital=100000)
                        df_bt = engine.run(df_res)
                        metrics = engine.calculate_metrics(df_bt)
                        
                        # KPI
                        k1, k2, k3, k4, k5 = st.columns(5)
                        k1.metric("总回报", metrics['Total Return'])
                        k2.metric("年化收益", metrics['CAGR'])
                        k3.metric("夏普比率", metrics['Sharpe Ratio'])
                        k4.metric("最大回撤", metrics['Max Drawdown'])
                        k5.metric("胜率", metrics['Win Rate'])
                        
                        # 图表
                        tab1, tab2 = st.tabs(["资金曲线", "详细数据"])
                        with tab1:
                            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4])
                            fig.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Close'], name="Price", line=dict(color='white', width=1)), row=1, col=1)
                            
                            # 标记买卖
                            buy = df_bt[df_bt['Signal']==1]
                            sell = df_bt[df_bt['Signal']==-1]
                            fig.add_trace(go.Scatter(x=buy.index, y=buy['Close'], mode='markers', marker=dict(symbol='triangle-up', color='lime', size=8), name='Buy'), row=1, col=1)
                            fig.add_trace(go.Scatter(x=sell.index, y=sell['Close'], mode='markers', marker=dict(symbol='triangle-down', color='red', size=8), name='Sell'), row=1, col=1)
                            
                            fig.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Equity_Curve'], name="Equity", line=dict(color='cyan', width=2)), row=2, col=1)
                            fig.update_layout(height=600, template="plotly_dark")
                            st.plotly_chart(fig, use_container_width=True)
                            
                        with tab2:
                            st.dataframe(df_bt.tail(100).sort_index(ascending=False))
                    else:
                        st.warning("策略未能生成有效信号。")
                        
        except Exception as e:
            st.error(f"系统运行出错: {e}")
