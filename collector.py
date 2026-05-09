import yfinance as yf
import pandas as pd
import requests
import datetime

class DataCollector:
    def __init__(self, api_key):
        self.api_key = api_key
        self.ticker = "^GSPC"  # S&P 500

    def get_stock_data(self, period="1mo", interval="1d"):
        """S&P 500の株価データを取得"""
        print(f"S&P 500の価格データを取得中 ({period})...")
        data = yf.download(self.ticker, period=period, interval=interval)
        return data

    def get_news_sentiment(self):
        """Alpha Vantageからニュースと感情分析スコアを取得"""
        print("ニュースと感情分析データを取得中...")
        # 取得件数を増やして（最大1000件）、過去数日分のデータが含まれるようにします
        url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={self.api_key}&limit=1000'
        
        try:
            r = requests.get(url)
            data = r.json()
        except Exception as e:
            print(f"ネットワークエラーが発生しました: {e}")
            return pd.DataFrame()

        # 回数制限やエラーのチェック
        if "Information" in data:
            print(f"\n[!] Alpha Vantageからの通知: {data['Information']}")
            return pd.DataFrame()
        
        if "Note" in data:
            print(f"\n[!] API制限: {data['Note']}")
            return pd.DataFrame()

        if "Error Message" in data:
            print(f"\n[!] エラー: APIキーが無効な可能性があります。 (.envファイルを確認してください)")
            return pd.DataFrame()

        if "feed" not in data or not data["feed"]:
            print("\n[!] ニュースデータが取得できませんでした。しばらく時間を置いてから再度お試しください。")
            return pd.DataFrame()

        news_list = []
        for item in data["feed"]:
            dt_str = item["time_published"]
            date = datetime.datetime.strptime(dt_str, "%Y%m%dT%H%M%S").date()
            
            news_list.append({
                "date": date,
                "sentiment_score": float(item["overall_sentiment_score"])
            })

        df = pd.DataFrame(news_list)
        
        if df.empty or "date" not in df.columns:
            print("\n[!] 有効なニュースデータがありません。")
            return pd.DataFrame()

        # 日次で集計（1日の平均感情スコアを算出）
        daily_sentiment = df.groupby("date")["sentiment_score"].mean().reset_index()
        return daily_sentiment
