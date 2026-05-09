import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class MarketAnalyzer:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False

    def prepare_data(self, stock_df, sentiment_df):
        """価格データと感情スコアを結合し、予測用のデータセットを作成"""
        # yfinanceの新仕様による多層カラム（MultiIndex）を解除
        stock_df_flat = stock_df.copy()
        if isinstance(stock_df_flat.columns, pd.MultiIndex):
            stock_df_flat.columns = stock_df_flat.columns.get_level_values(0)
        
        # インデックスを日付型に揃える
        stock_df_flat.index = pd.to_datetime(stock_df_flat.index).date
        
        sentiment_df_copy = sentiment_df.copy()
        sentiment_df_copy["date"] = pd.to_datetime(sentiment_df_copy["date"]).dt.date
        
        # 結合
        df = pd.merge(stock_df_flat, sentiment_df_copy, left_index=True, right_on="date", how="inner")
        
        # 目的変数: 翌日の株価が上がったか(1)下がったか(0)
        df['Price_Up'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # 欠損値（最新日）を削除
        df = df.dropna()
        
        return df

    def train_model(self, df):
        """モデルの学習"""
        if len(df) < 3:
            return f"データが少なすぎます（{len(df)}件）。分析にはもう少し日数が必要です。"

        X = df[['sentiment_score']]  # 特徴量: 感情スコア
        y = df['Price_Up']           # ターゲット: 翌日の騰落

        # データが少ない場合は分割せずに全データで学習（簡易版）
        if len(df) < 10:
            self.model.fit(X, y)
            self.is_trained = True
            return f"限定的なデータ（{len(df)}件）で学習しました。"
        
        # 学習データとテストデータに分割
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        return f"モデルの学習完了。テスト精度: {acc:.2f}"

    def predict_next_move(self, latest_sentiment_score):
        """最新の感情スコアから明日の動きを予測"""
        prediction = self.model.predict([[latest_sentiment_score]])
        prob = self.model.predict_proba([[latest_sentiment_score]])
        
        result = "上昇" if prediction[0] == 1 else "下落"
        confidence = prob[0][prediction[0]] * 100
        
        return result, confidence
