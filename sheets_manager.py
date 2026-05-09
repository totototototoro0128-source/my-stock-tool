import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import datetime
import os
import streamlit as st

class SheetsManager:
    def __init__(self, spreadsheet_id):
        self.scope = ['https://www.googleapis.com/auth/spreadsheets',
                      'https://www.googleapis.com/auth/drive']
        
        # クラウド設定（st.secrets）があるか確認
        if "gcp_service_account" in st.secrets:
            # クラウド環境用
            creds_dict = dict(st.secrets["gcp_service_account"])
            self.credentials = Credentials.from_service_account_info(creds_dict, scopes=self.scope)
        else:
            # ローカル環境用（ファイルから読み込み）
            creds_path = 'google_credentials.json'
            if not os.path.exists(creds_path):
                raise FileNotFoundError("google_credentials.json が見つかりません。")
            self.credentials = Credentials.from_service_account_file(creds_path, scopes=self.scope)

        self.client = gspread.authorize(self.credentials)
        self.spreadsheet = self.client.open_by_key(spreadsheet_id)
        self.sheet = self.spreadsheet.get_worksheet(0)

        # ヘッダー作成
        try:
            headers = self.sheet.row_values(1)
            if not headers:
                self.sheet.append_row(["分析日", "感情スコア", "予測結果", "確信度", "実行日時"])
        except:
            pass

    def save_prediction(self, date, sentiment_score, prediction, confidence):
        data = [
            str(date),
            round(float(sentiment_score), 4),
            prediction,
            f"{confidence:.1f}%",
            str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        ]
        self.sheet.append_row(data)
        return "スプレッドシートに結果を保存しました。"
