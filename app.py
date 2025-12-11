import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta  # [关键修复] 补全时间库导入
import warnings

# 忽略部分 sklearn 警告
warnings.filterwarnings("ignore")

# ==========================================
# PART 1: 核心策略类 (Strategies)
# ==========================================

class HMMStandardStrategy:
    """经典 HMM 策略: 低波(0)做多，高波(2)做空"""
    def __init__(self, n_components=3, iter_num=1000, window_size=21):
        self.n_components = n_components
        self.iter_num = iter_num
        self.window_size = window_size

    def generate_signals(self, df):
        df = df.copy()
        if df.index.tz is not None: df.index = df.index.tz_localize(None) # 时区修复
        
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
        
        # 状态排序 (按波动率)
        state_vol_means = [X[hidden_states == i, 1].mean() for i in range(self.n_components)]
        sorted_stats = sorted(list(enumerate(state_vol_means)), key=lambda x: x[1])
        mapping = {old: new for new, (old, _) in enumerate(sorted_stats)}
        
        df['Regime'] = np.array([mapping[s] for s in hidden_states])
        
        df['Signal'] = 0
        df.loc[df['Regime'] == 0, 'Signal'] = 1   
        df.loc[df['Regime'] == self.n_components-1, 'Signal'] = -1 
        
        return df

class HMMAdaptiveStrategy:
    """自适应贝叶斯策略"""
    def __init__(self, n_components=3, iter_num=1000, window_size=21):
        self.n_components = n_components
        self.iter_num = iter_num
        self.window_size = window_size

    def generate_signals(self, df):
        df = df.copy()
        if df.index.tz is not None: df.index = df.index.tz_localize(None) # 时区修复

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
        state_vol_means = [X[hidden_states == i, 1].mean() for i in range(self.n_components)]
        sorted_stats = sorted(list(enumerate(state_vol_means)), key=lambda x: x[1])
        mapping = {old: new for new, (old, _) in enumerate(sorted_stats)}
        
        posterior_probs = model.predict_proba(X)
        sorted_probs = np.zeros_like(posterior_probs)
        for old_i, new_i in mapping.items():
            sorted_probs[:, new_i] = posterior_probs[:, old_i]
            
        df['Regime'] = np.array([mapping[s] for s in hidden_states])
        for i in range(self.n_components):
            df[f'Prob_S{i}'] = sorted_probs[:, i]
        
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
        
        threshold = 0.0003
        df['Signal'] = 0
        df.loc[df['Bayes_Exp_Ret'] > threshold, 'Signal'] = 1
        df.loc[df['Bayes_Exp_Ret'] < -threshold, 'Signal'] = -1
        
        return df

class HMM_MACD_Strategy:
    """
    HMM + 4H MACD 贝叶斯共振策略
    """
    def __init__(self, n_components=3, iter_num=1000, window_size=21):
        self.n_components = n_components
        self.iter_num = iter_num
        self.window_size = window_size

    def calculate_macd(self, df_prices):
        exp1 = df_prices.ewm(span=12, adjust=False).mean()
        exp2 = df_prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        return macd, signal, hist

    def get_4h_macd_signal(self, ticker):
        try:
            # 获取最近60天的高频数据
            df_1h = yf.download(ticker, period="60d", interval="1h", progress=False, auto_adjust=True)
            if isinstance(df_1h.columns, pd.MultiIndex): df_1h.columns = df_1h.columns.get_level_values(0)
            
            if len(df_1h) < 24: return None

            # [关键] 强制去除时区，解决 Cannot join tz-naive with tz-aware 错误
            if df_1h.index.tz is not None:
                df_1h.index = df_1h.index.tz_localize(None)

            df_4h = df_1h['Close'].resample('4h').ohlc() 
            df_4h.dropna(inplace=True)
            close_4h = df_4h['close']

            macd, signal, hist = self.calculate_macd(close_4h)
            
            # 对齐到日线
            macd_daily = hist.resample('1D').last().fillna(method='ffill')
            
            return macd_daily
        except Exception as e:
            # st.error(f"MACD 计算警告: {e}")
            return None

    def generate_signals(self, df, ticker_symbol=None):
        if df.index.tz is not None: df.index = df.index.tz_localize(None)

        # 1. 基础 HMM
        df = HMMAdaptiveStrategy(self.n_components, self.iter_num, self.window_size).generate_signals(df)
        
        df['Signal_HMM_Only'] = df['Signal']
        df['Prob_S0_Prior'] = df.get('Prob_S0', 0.33)
        df['Prob_S2_Prior'] = df.get('Prob_S2', 0.33)
        
        # 2. 获取 MACD 证据
        if ticker_symbol:
            macd_series = self.get_4h_macd_signal(ticker_symbol)
            if macd_series is not None:
                df = df.join(macd_series.rename('MACD_Hist_4H'), how='left')
                df['MACD_Hist_4H'] = df['MACD_Hist_4H'].fillna(method='ffill').fillna(0)
            else:
                df['MACD_Hist_4H'] = 0
        else:
            df['MACD_Hist_4H'] = 0

        # 3. 贝叶斯更新
        # MACD 归一化后作为似然函数的指数
        macd_norm = np.clip(df['MACD_Hist_4H'] / (df['Close'] * 0.01 + 1e-6), -1, 1) * 2.0
        
        likelihood_0 = np.exp(macd_norm)
        likelihood_1 = np.ones_like(macd_norm)
        likelihood_2 = np.exp(-macd_norm)
        
        df['Prob_S0_Post'] = df['Prob_S0_Prior'] * likelihood_0
        df['Prob_S1_Post'] = df.get('Prob_S1', 0.33) * likelihood_1
        df['Prob_S2_Post'] = df['Prob_S2_Prior'] * likelihood_2
        
        total_prob = df['Prob_S0_Post'] + df['Prob_S1_Post'] + df['Prob_S2_Post']
        df['Prob_S0_Post'] /= total_prob
        df['Prob_S2_Post'] /= total_prob
        
        # 4. 融合信号
        df['Signal'] = 0 
        df.loc[(df['Prob_S0_Post'] > 0.4) & (df['MACD_Hist_4H'] > 0), 'Signal'] = 1
        df.loc[(df['Prob_S2_Post'] > 0.4) & (df['MACD_Hist_4H'] < 0), 'Signal'] = -1
        
        return df

class SpreadArbStrategy:
    """统计套利策略"""
    def __init__(self, window_size=20, z_threshold=1.5):
        self.window_size = window_size
        self.z_threshold = z_threshold

    def generate_signals(self, df_a, df_b):
        if df_a.index.tz is not None: df_a.index = df_a.index.tz_localize(None)
        if df_b.index.tz is not None: df_b.index = df_b.index.tz_localize(None)

        data = pd.DataFrame(index=df_a.index)
        data['Price_A'] = df_a['Close']
        data['Price_B'] = df_b['Close']
        data.dropna(inplace=True)
        if len(data) < 50: return data

        data['Spread'] = data['Price_A'] - data['Price_B']
        data['Spread_Mean'] = data['Spread'].rolling(self.window_size).mean()
        data['Spread_Std'] = data['Spread'].rolling(self.window_size).std()
        
        data['Z_Score'] = (data['Spread'] - data['Spread_Mean']) / (data['Spread_Std'] + 1e-8)
        
        data['Signal'] = 0
        data.loc[data['Z_Score'] > self.z_threshold, 'Signal'] = -1
        data.loc[data['Z_Score'] < -self.z_threshold, 'Signal'] = 1
        
        ret_a = np.log(data['Price_A'] / data['Price_A'].shift(1)).fillna(0)
        ret_b = np.log(data['Price_B'] / data['Price_B'].shift(1)).fillna(0)
        data['Spread_Ret_Raw'] = ret_a - ret_b
        return data

# ==========================================
# PART 2: 回测引擎 (Backtest Engine)
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

    def calculate_metrics(self, df):
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
            
        return {
            "Total Return": f"{total_ret*100:.2f}%",
            "CAGR": f"{cagr*100:.2f}%",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Max Drawdown": f"{max_dd*100:.2f}%",
            "Win Rate": f"{win_rate*100:.1f}%"
        }
        
    def _empty_metrics(self):
        return {k: "N/A" for k in ["Total Return", "CAGR", "Sharpe Ratio", "Max Drawdown", "Win Rate"]}

# ==========================================
# PART 3: 信号展示
# ==========================================

def display_signal_panel(df, strategy_type):
    last = df.iloc[-1]
    sig = last['Signal']
    
    st.markdown("### 🚦 实时交易信号驾驶舱")
    col_sig, col_reason = st.columns([1, 2])
    
    with col_sig:
        if sig == 1: st.success("## 🟢 强力做多\n**LONG SIGNAL**")
        elif sig == -1: st.error("## 🔴 强力卖出\n**SHORT SIGNAL**")
        else: st.warning("## ⚪ 空仓观望\n**WAIT / CASH**")
            
    with col_reason:
        st.markdown("#### 🤖 策略逻辑分析")
        if "MACD" in strategy_type:
            prob_0_post = last.get('Prob_S0_Post', 0) * 100
            prob_2_post = last.get('Prob_S2_Post', 0) * 100
            macd_val = last.get('MACD_Hist_4H', 0)
            macd_status = "🟢 金叉/向上" if macd_val > 0 else "🔴 死叉/向下"
            
            msg = f"""
            - **HMM 宏观概率**: 牛(S0): **{prob_0_post:.1f}%** | 熊(S2): **{prob_2_post:.1f}%**
            - **4H 技术面共振**: MACD Histogram = {macd_val:.3f} ({macd_status})
            """
            st.info(msg)
        elif "自适应" in strategy_type:
            exp_ret = last.get('Bayes_Exp_Ret', 0) 
            st.info(f"- **当前体制**: State {int(last['Regime'])}\n- **贝叶斯期望**: {exp_ret*100:.4f}%")
        elif "套利" in strategy_type:
            st.info(f"- **Z-Score**: {last.get('Z_Score', 0):.2f} σ")
        else:
            st.info(f"- **当前体制**: State {int(last['Regime'])}")

# ==========================================
# PART 4: Streamlit UI 主程序
# ==========================================

st.set_page_config(page_title="能源量化终端 Pro Max", layout="wide", page_icon="⚡")
st.title("⚡ Energy Quant Lab: HMM + MACD Resonance System")
st.markdown("### 贝叶斯后验增强版：日线 HMM 叠加 4H MACD 信号")

st.sidebar.header("⚙️ 策略控制台")
strategy_type = st.sidebar.selectbox(
    "选择策略类型",
    ["HMM + 4H MACD 贝叶斯共振 (New!)", "HMM 自适应贝叶斯 (Adaptive)", "HMM 经典模型 (Standard)", "统计套利 (Pairs Trading)"]
)

tickers = {"Brent Crude": "BZ=F", "WTI Crude": "CL=F", "Natural Gas (HH)": "NG=F", "Dutch TTF": "TTF=F"}

if "套利" in strategy_type:
    col1, col2 = st.sidebar.columns(2)
    asset_a = col1.selectbox("资产 A (Long)", list(tickers.keys()), index=0)
    asset_b = col2.selectbox("资产 B (Short)", list(tickers.keys()), index=1)
    ticker = f"{asset_a} vs {asset_b}"
else:
    asset = st.sidebar.selectbox("选择交易标的", list(tickers.keys()))
    ticker = tickers[asset]

start_date = st.sidebar.date_input("回测开始", datetime.now() - timedelta(days=365*2))
end_date = st.sidebar.date_input("回测结束", datetime.now())

if st.sidebar.button("🚀 运行分析", type="primary"):
    engine = BacktestEngine(initial_capital=100000)
    
    with st.spinner(f"正在计算 {ticker} 的量化信号..."):
        try:
            if "套利" in strategy_type:
                df_a = yf.download(tickers[asset_a], start=start_date, end=end_date, progress=False, auto_adjust=True)
                df_b = yf.download(tickers[asset_b], start=start_date, end=end_date, progress=False, auto_adjust=True)
                
                if isinstance(df_a.columns, pd.MultiIndex): df_a.columns = df_a.columns.get_level_values(0)
                if isinstance(df_b.columns, pd.MultiIndex): df_b.columns = df_b.columns.get_level_values(0)

                if df_a.empty or df_b.empty:
                    st.error("数据不足。")
                else:
                    strat = SpreadArbStrategy()
                    df_res = strat.generate_signals(df_a, df_b)
                    if len(df_res) > 0:
                        display_signal_panel(df_res, strategy_type)
                        st.divider()
                        df_bt = engine.run(df_res, ret_col='Spread_Ret_Raw')
                        metrics = engine.calculate_metrics(df_bt)
                        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
                        kpi1.metric("总回报", metrics['Total Return'])
                        kpi2.metric("年化收益", metrics['CAGR'])
                        kpi3.metric("夏普比率", metrics['Sharpe Ratio'])
                        kpi4.metric("最大回撤", metrics['Max Drawdown'])
                        kpi5.metric("胜率", metrics['Win Rate'])
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Equity_Curve'], name="Strategy Equity", line=dict(color='cyan', width=2)))
                        st.plotly_chart(fig, use_container_width=True)
            else:
                df = yf.download(tickers[asset], start=start_date, end=end_date, progress=False, auto_adjust=True)
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                
                if df.empty:
                    st.error("数据获取失败。")
                else:
                    if "MACD" in strategy_type:
                        strat = HMM_MACD_Strategy()
                        df_res = strat.generate_signals(df, ticker_symbol=tickers[asset])
                    elif "自适应" in strategy_type:
                        strat = HMMAdaptiveStrategy()
                        df_res = strat.generate_signals(df)
                    else:
                        strat = HMMStandardStrategy()
                        df_res = strat.generate_signals(df)
                    
                    if 'Signal' in df_res.columns:
                        display_signal_panel(df_res, strategy_type)
                        st.divider()
                        df_bt = engine.run(df_res, ret_col='Log_Ret')
                        metrics = engine.calculate_metrics(df_bt)
                        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
                        kpi1.metric("总回报", metrics['Total Return'])
                        kpi2.metric("年化收益", metrics['CAGR'])
                        kpi3.metric("夏普比率", metrics['Sharpe Ratio'])
                        kpi4.metric("最大回撤", metrics['Max Drawdown'])
                        kpi5.metric("胜率", metrics['Win Rate'])
                        
                        tab1, tab2 = st.tabs(["📈 信号与净值", "🔬 数据细节"])
                        with tab1:
                            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.05)
                            fig.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Close'], name="Price", line=dict(color='white', width=1)), row=1, col=1)
                            buy_sig = df_bt[df_bt['Signal'] == 1]
                            sell_sig = df_bt[df_bt['Signal'] == -1]
                            fig.add_trace(go.Scatter(x=buy_sig.index, y=buy_sig['Close'], mode='markers', marker=dict(symbol='triangle-up', color='lime', size=10), name='Buy'), row=1, col=1)
                            fig.add_trace(go.Scatter(x=sell_sig.index, y=sell_sig['Close'], mode='markers', marker=dict(symbol='triangle-down', color='red', size=10), name='Sell'), row=1, col=1)
                            
                            if "MACD" in strategy_type and 'MACD_Hist_4H' in df_bt.columns:
                                colors = df_bt['MACD_Hist_4H'].apply(lambda x: '#00ff00' if x>0 else '#ff0000')
                                fig.add_trace(go.Bar(x=df_bt.index, y=df_bt['MACD_Hist_4H'], name="4H MACD Hist", marker_color=colors), row=2, col=1)
                                fig.update_yaxes(title_text="MACD (4H)", row=2, col=1)
                            else:
                                fig.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Equity_Curve'], name="Strategy Equity", line=dict(color='cyan', width=2)), row=2, col=1)
                                fig.update_yaxes(title_text="Equity", row=2, col=1)
                            
                            fig.update_layout(height=600, template="plotly_dark", title="Resonance Chart")
                            st.plotly_chart(fig, use_container_width=True)
                        with tab2:
                            st.dataframe(df_bt.tail(100).sort_index(ascending=False))
                    else:
                        st.warning("信号生成失败。")
        except Exception as e:
            st.error(f"运行出错: {e}")
