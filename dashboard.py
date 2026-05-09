import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from dotenv import load_dotenv
from collector import DataCollector
from analyzer import MarketAnalyzer
from sheets_manager import SheetsManager

# ページの設定
st.set_page_config(page_title="S&P 500 AI予測ダッシュボード", layout="wide")

st.title("📈 S&P 500 AI予測ダッシュボード")
st.markdown("世界中のニュースを分析し、S&P 500の価格動向を予測します。")

# 設定の読み込み（クラウド優先、なければローカル）
load_dotenv()
api_key = st.secrets.get("ALPHA_VANTAGE_API_KEY", os.getenv("ALPHA_VANTAGE_API_KEY"))
spreadsheet_id = st.secrets.get("SPREADSHEET_ID", os.getenv("SPREADSHEET_ID"))

if not api_key or "ここに" in str(api_key):
    st.error("APIキーが設定されていません。")
    st.stop()

# サイドバー設定
st.sidebar.header("分析設定")
period = st.sidebar.selectbox("分析期間", ["1mo", "3mo", "6mo", "1y"], index=0)

# データ取得
@st.cache_data(ttl=3600)
def get_all_data(api_key, period):
    collector = DataCollector(api_key)
    stock_df = collector.get_stock_data(period=period)
    sentiment_df = collector.get_news_sentiment()
    return stock_df, sentiment_df

if st.button("最新データで分析開始"):
    with st.spinner("データを取得・分析中..."):
        stock_data, sentiment_data = get_all_data(api_key, period)

        if sentiment_data.empty:
            st.warning("ニュースデータが取得できませんでした。時間をおいて再度お試しください。")
        else:
            analyzer = MarketAnalyzer()
            df = analyzer.prepare_data(stock_data, sentiment_data)
            train_msg = analyzer.train_model(df)

            col1, col2 = st.columns([1, 2])
            latest_sentiment = sentiment_data.sort_values("date", ascending=False).iloc[0]

            with col1:
                st.subheader("🤖 AI予測結果")
                if analyzer.is_trained:
                    prediction, confidence = analyzer.predict_next_move(latest_sentiment["sentiment_score"])
                    color = "red" if prediction == "上昇" else "blue"
                    st.markdown(f"""
                        <div style="padding:20px; border-radius:10px; border:2px solid {color}; text-align:center;">
                            <h2 style="color:{color};">{prediction}</h2>
                            <p>確信度: {confidence:.1f}%</p>
                            <p><small>分析日: {latest_sentiment['date']}</small></p>
                        </div>
                    """, unsafe_allow_html=True)

                    try:
                        sheets = SheetsManager(spreadsheet_id)
                        save_msg = sheets.save_prediction(
                            latest_sentiment['date'],
                            latest_sentiment['sentiment_score'],
                            prediction,
                            confidence
                        )
                        st.success(save_msg)
                    except Exception as e:
                        st.error(f"スプレッドシート保存エラー: {e}")
                else:
                    st.info("データ不足のため予測をスキップしました。")
                    st.write(f"最新の感情スコア ({latest_sentiment['date']}): {latest_sentiment['sentiment_score']:.2f}")
                
                st.write(f"ステータス: {train_msg}")

            with col2:
                st.subheader("📊 価格とニュース感情スコアの推移")
                if not df.empty:
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    fig.add_trace(go.Scatter(x=df['date'], y=df['Close'], name="S&P 500 価格"), secondary_y=False)
                    fig.add_trace(go.Bar(x=df['date'], y=df['sentiment_score'], name="感情スコア", opacity=0.3), secondary_y=True)
                    fig.update_layout(title_text="価格とニュースの相関")
                    st.plotly_chart(fig, use_container_width=True)

            st.subheader("📰 最新の感情スコア（日次平均）")
            st.dataframe(sentiment_data.sort_values("date", ascending=False), use_container_width=True)
else:
    st.info("サイドバーで期間を選び、「分析開始」ボタンを押してください。")
