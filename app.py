import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings

# 忽略部分 sklearn 警告
warnings.filterwarnings("ignore")

# ==========================================
# PART 1: 核心策略类 (Strategies)
# ==========================================

class HMMStandardStrategy:
    """经典 HMM 策略"""
    def __init__(self, n_components=3, iter_num=1000, window_size=21):
        self.n_components = n_components
        self.iter_num = iter_num
        self.window_size = window_size

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
            state_means.append(df[df['Regime'] == i]['Log_Ret'].mean())
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
    [增强版] HMM + 4H MACD 结构共振策略
    核心升级：引入 MACD 水位 (Water Level) 概念，区分水上水下信号质量。
    """
    def __init__(self, n_components=3, iter_num=1000, window_size=21):
        self.n_components = n_components
        self.iter_num = iter_num
        self.window_size = window_size

    def calculate_macd_structure(self, df_prices):
        """计算 MACD 及其水位 Z-Score"""
        # 1. 标准 MACD
        exp1 = df_prices.ewm(span=12, adjust=False).mean()
        exp2 = df_prices.ewm(span=26, adjust=False).mean()
        dif = exp1 - exp2
        dea = dif.ewm(span=9, adjust=False).mean()
        hist = dif - dea
        
        # 2. 计算 DIF 的 Z-Score (相对水位)
        # 使用 60 周期滚动窗口来定义"当前相对位置"
        dif_mean = dif.rolling(window=60).mean()
        dif_std = dif.rolling(window=60).std()
        dif_z = (dif - dif_mean) / (dif_std + 1e-8)
        
        return dif, dea, hist, dif_z

    def get_4h_macd_data(self, ticker):
        try:
            # 获取数据
            df_1h = yf.download(ticker, period="60d", interval="1h", progress=False, auto_adjust=True)
            if isinstance(df_1h.columns, pd.MultiIndex): df_1h.columns = df_1h.columns.get_level_values(0)
            if len(df_1h) < 60: return None
            if df_1h.index.tz is not None: df_1h.index = df_1h.index.tz_localize(None)

            # 重采样 4H
            df_4h = df_1h['Close'].resample('4h').ohlc()
            df_4h.dropna(inplace=True)
            
            # 计算结构化 MACD
            dif, dea, hist, dif_z = self.calculate_macd_structure(df_4h['close'])
            
            # 组合数据
            macd_df = pd.DataFrame({
                'MACD_DIF_4H': dif,
                'MACD_DEA_4H': dea,
                'MACD_Hist_4H': hist,
                'MACD_Z_4H': dif_z
            })
            
            # 降采样对齐到日线 (ffill)
            macd_daily = macd_df.resample('1D').last().fillna(method='ffill')
            return macd_daily
        except Exception as e:
            return None

    def generate_signals(self, df, ticker_symbol=None):
        if df.index.tz is not None: df.index = df.index.tz_localize(None)

        # 1. HMM 先验
        df = HMMAdaptiveStrategy(self.n_components, self.iter_num, self.window_size).generate_signals(df)
        
        df['Signal_HMM_Only'] = df['Signal']
        df['Prob_S0_Prior'] = df.get('Prob_S0', 0.33) # 牛
        df['Prob_S2_Prior'] = df.get('Prob_S2', 0.33) # 熊
        
        # 2. MACD 证据获取
        has_macd = False
        if ticker_symbol:
            macd_data = self.get_4h_macd_data(ticker_symbol)
            if macd_data is not None:
                df = df.join(macd_data, how='left')
                df = df.fillna(method='ffill').fillna(0)
                has_macd = True
        
        if not has_macd:
            df['MACD_Hist_4H'] = 0
            df['MACD_Z_4H'] = 0

        # 3. [核心] 结构化贝叶斯似然函数
        # 逻辑：
        # - 如果 MACD Hist > 0 (金叉态) 且 Z < -1 (深水区) -> 极强看多信号
        # - 如果 MACD Hist < 0 (死叉态) 且 Z > 1 (高空区) -> 极强看空信号
        # - 如果 MACD Hist > 0 但 Z > 1.5 -> 动能衰竭 (弱看多)
        
        # 基础分：直方图方向
        base_score = np.sign(df['MACD_Hist_4H']) 
        
        # 水位修正系数 (Water Level Multiplier)
        # Z越小(水下)，做多权重越大；Z越大(水上)，做空权重越大
        # 公式设计：Buy_Power = (1 - Z), Sell_Power = (1 + Z)
        # 例如 Z=-2 (深水), Buy_Power = 3 (强力); Z=2 (高空), Sell_Power = 3 (强力)
        
        water_factor = df['MACD_Z_4H'].clip(-2, 2)
        
        # 构建似然指数
        # 对 State 0 (牛) 的支持度
        score_bull = np.where(df['MACD_Hist_4H'] > 0, 
                              1.0 * (1 - 0.5 * water_factor), # 金叉在水下得分高，水上得分低
                              -1.0)                           # 死叉不支持牛
        
        # 对 State 2 (熊) 的支持度
        score_bear = np.where(df['MACD_Hist_4H'] < 0,
                              1.0 * (1 + 0.5 * water_factor), # 死叉在水上得分高，水下得分低
                              -1.0)                           # 金叉不支持熊
        
        # 计算似然
        likelihood_0 = np.exp(score_bull * 1.5) # 放大系数
        likelihood_2 = np.exp(score_bear * 1.5)
        
        # 贝叶斯更新
        df['Prob_S0_Post'] = df['Prob_S0_Prior'] * likelihood_0
        df['Prob_S2_Post'] = df['Prob_S2_Prior'] * likelihood_2
        
        # 归一化 (加上S1)
        prob_sum = df['Prob_S0_Post'] + df.get('Prob_S1', 0.33) + df['Prob_S2_Post']
        df['Prob_S0_Post'] /= prob_sum
        df['Prob_S2_Post'] /= prob_sum
        
        # 4. 生成最终信号
        df['Signal'] = 0
        # 只有在概率优势明显时才开仓
        df.loc[df['Prob_S0_Post'] > 0.45, 'Signal'] = 1
        df.loc[df['Prob_S2_Post'] > 0.45, 'Signal'] = -1
        
        # 5. 辅助列：用于 UI 展示水位状态
        df['MACD_Status'] = "Neutral"
        df.loc[df['MACD_Z_4H'] > 1.0, 'MACD_Status'] = "High (Overbought)"
        df.loc[df['MACD_Z_4H'] < -1.0, 'MACD_Status'] = "Deep Water (Oversold)"
        
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
        return {"Total Return": f"{total_ret*100:.2f}%", "CAGR": f"{cagr*100:.2f}%", "Sharpe Ratio": f"{sharpe:.2f}", "Max Drawdown": f"{max_dd*100:.2f}%", "Win Rate": f"{win_rate*100:.1f}%"}
        
    def _empty_metrics(self):
        return {k: "N/A" for k in ["Total Return", "CAGR", "Sharpe Ratio", "Max Drawdown", "Win Rate"]}

# ==========================================
# PART 3: 信号解读与展示
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
            
            # MACD 结构化数据
            macd_val = last.get('MACD_Hist_4H', 0)
            macd_z = last.get('MACD_Z_4H', 0)
            macd_status = last.get('MACD_Status', 'N/A')
            
            # 图标逻辑
            trend_icon = "🟢" if macd_val > 0 else "🔴"
            water_icon = "🌊" if macd_z < -1 else ("🏔️" if macd_z > 1 else "⚖️")
            
            msg = f"""
            - **HMM 宏观概率**: 牛(S0): **{prob_0_post:.1f}%** | 熊(S2): **{prob_2_post:.1f}%**
            - **MACD 结构雷达**: 
                - 动能: {trend_icon} Hist = {macd_val:.3f}
                - 水位: {water_icon} Z-Score = **{macd_z:.2f}** ({macd_status})
            """
            
            # 智能文案生成
            if sig == 1:
                if macd_z < -1: msg += "\n\n💡 **结论**: 黄金机会！**深水区金叉 (Low Suction)**，HMM 确认反转，建议**重仓做多**。"
                else: msg += "\n\n💡 **结论**: 趋势向好，动能配合，建议**顺势做多**。"
            elif sig == -1:
                if macd_z > 1: msg += "\n\n💡 **结论**: 风险极大！**高位死叉 (Top Divergence)**，HMM 确认崩盘，建议**清仓/做空**。"
                else: msg += "\n\n💡 **结论**: 趋势向下，建议**做空**。"
            else:
                if macd_z < -1 and macd_val < 0: msg += "\n\n💡 **结论**: 虽然深水区，但**尚未金叉** (左侧交易风险)，建议**紧盯反转信号**。"
                else: msg += "\n\n💡 **结论**: 信号背离或动能不足，建议**观望**。"
            
            st.info(msg)
            
        elif "自适应" in strategy_type:
            prob_0 = last.get('Prob_S0', 0) * 100
            exp_ret = last.get('Bayes_Exp_Ret', 0) 
            msg = f"- **当前体制**: State {int(last['Regime'])}\n- **贝叶斯期望**: {exp_ret*100:.4f}%"
            st.info(msg)
        elif "套利" in strategy_type:
            z_score = last.get('Z_Score', 0)
            msg = f"- **Z-Score**: {z_score:.2f} σ"
            st.info(msg)
        else:
            st.info(f"- **当前体制**: State {int(last['Regime'])}")

# ==========================================
# PART 4: Streamlit UI 主程序
# ==========================================

st.set_page_config(page_title="能源量化终端 Pro Max", layout="wide", page_icon="⚡")
st.title("⚡ Energy Quant Lab: HMM + MACD Resonance System")
st.markdown("### 贝叶斯后验增强版：日线 HMM 叠加 4H MACD 结构分析")

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
    engine = BacktestEngine(initial_capital=100000, transaction_cost=0.0002)
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
                            # 增加行高以容纳MACD
                            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.05)
                            
                            # Row 1: 价格 + 信号
                            fig.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Close'], name="Price", line=dict(color='white', width=1)), row=1, col=1)
                            buy_sig = df_bt[df_bt['Signal'] == 1]
                            sell_sig = df_bt[df_bt['Signal'] == -1]
                            fig.add_trace(go.Scatter(x=buy_sig.index, y=buy_sig['Close'], mode='markers', marker=dict(symbol='triangle-up', color='lime', size=10), name='Buy'), row=1, col=1)
                            fig.add_trace(go.Scatter(x=sell_sig.index, y=sell_sig['Close'], mode='markers', marker=dict(symbol='triangle-down', color='red', size=10), name='Sell'), row=1, col=1)
                            
                            # Row 2: 净值
                            fig.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Equity_Curve'], name="Equity", line=dict(color='cyan', width=1.5)), row=2, col=1)
                            
                            # Row 3: MACD (如果是该策略)
                            if "MACD" in strategy_type and 'MACD_Hist_4H' in df_bt.columns:
                                # Hist 颜色
                                colors = df_bt['MACD_Hist_4H'].apply(lambda x: '#00ff00' if x>0 else '#ff0000')
                                fig.add_trace(go.Bar(x=df_bt.index, y=df_bt['MACD_Hist_4H'], name="4H MACD", marker_color=colors), row=3, col=1)
                                # Z-Score 线 (水位)
                                fig.add_trace(go.Scatter(x=df_bt.index, y=df_bt['MACD_Z_4H'], name="Z-Score (Water)", line=dict(color='yellow', width=1, dash='dot'), yaxis='y4'), row=3, col=1)
                                
                                # 标记 Z-Score 阈值
                                fig.add_hline(y=1.5, line_dash="dot", line_color="red", row=3, col=1, annotation_text="High Water")
                                fig.add_hline(y=-1.5, line_dash="dot", line_color="green", row=3, col=1, annotation_text="Deep Water")
                            else:
                                # 其他策略显示波动率
                                fig.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Volatility'], name="Volatility", line=dict(color='orange')), row=3, col=1)

                            fig.update_layout(height=800, template="plotly_dark", title="Market Structure Resonance")
                            st.plotly_chart(fig, use_container_width=True)
                        with tab2:
                            st.dataframe(df_bt.tail(100).sort_index(ascending=False))
                    else:
                        st.warning("信号生成失败。")
        except Exception as e:
            st.error(f"运行出错: {e}")
