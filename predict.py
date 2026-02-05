import yfinance as yf
import numpy as np
import pandas as pd
import joblib
import requests
import tensorflow as tf
from datetime import datetime

# --- KONFIGURASI ---
BOT_TOKEN = "8599866641:AAHA7GxblUZ6jVedQ2UOniFWKqxBy6HMn3M"
CHAT_IDS = ["977432672", "864486458"] # Sesuaikan ID

# Target Saham (Sama seperti daftar Anda)
target_tickers = [
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "BRIS.JK", "BBTN.JK", "BDMN.JK",
    "ADRO.JK", "PTBA.JK", "ITMG.JK", "UNTR.JK", "PGAS.JK", "MEDC.JK", "AKRA.JK",
    "ANTM.JK", "MDKA.JK", "INCO.JK", "TINS.JK", "MBMA.JK", "NCKL.JK", "HRUM.JK",
    "TLKM.JK", "ISAT.JK", "EXCL.JK", "JSMR.JK", "ICBP.JK", "INDF.JK", "MYOR.JK",
    "UNVR.JK", "AMRT.JK", "MIDI.JK", "MAPI.JK", "ACES.JK", "CPIN.JK", "JPFA.JK",
    "BSDE.JK", "CTRA.JK", "SMRA.JK", "PWON.JK", "PANI.JK", "GOTO.JK", "BUKA.JK",
    "EMTK.JK", "ASII.JK", "AUTO.JK", "DRMA.JK", "INKP.JK", "TKIM.JK", "BRMS.JK",
    "BUMI.JK", "DEWA.JK", "PSAB.JK", "DOID.JK", "MAPA.JK", "SRTG.JK", "ESSA.JK"
]

SEQ_LEN = 45 # Harus sama dengan Train

def send_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    for uid in CHAT_IDS:
        try:
            requests.post(url, json={"chat_id": uid, "text": message, "parse_mode": "Markdown"})
        except: pass

def get_technical_features_live(df):
    # Logika HARUS sama persis dengan train.py
    df = df.copy()
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    ma20 = df['Close'].rolling(window=20).mean()
    df['Dist_MA20'] = (df['Close'] - ma20) / ma20
    
    std20 = df['Close'].rolling(window=20).std()
    upper = ma20 + (std20 * 2)
    lower = ma20 - (std20 * 2)
    df['BB_Width'] = (upper - lower) / ma20
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI_Norm'] = df['RSI'] / 100.0
    
    df['Vol_Force'] = df['Log_Ret'] * (df['Volume'] / df['Volume'].rolling(20).mean())
    
    return df[['Log_Ret', 'Dist_MA20', 'BB_Width', 'RSI_Norm', 'Vol_Force']]

def predict_daily():
    print("🔍 Memulai AI Scanning...")
    try:
        model = tf.keras.models.load_model('model_advanced.keras')
        scaler = joblib.load('scaler_v2.pkl')
    except:
        print("❌ Model/Scaler missing.")
        return

    report_lines = []
    
    # Ambil data lebih panjang untuk safety calculation indicator
    data = yf.download(target_tickers, period="6mo", auto_adjust=True, group_by='ticker', threads=True)

    for t in target_tickers:
        try:
            df = data[t].copy()
            if len(df) < 60: continue
            
            # Handle NaN diawal
            df = df.dropna() 
            
            # Hitung Features
            feat_df = get_technical_features_live(df)
            feat_df = feat_df.dropna()
            
            # Ambil sequence terakhir (SEQ_LEN)
            last_seq = feat_df.values[-SEQ_LEN:]
            
            if len(last_seq) != SEQ_LEN: continue
            
            # Scaling
            last_seq_scaled = scaler.transform(last_seq)
            
            # Reshape (1, 45, 5)
            input_data = np.expand_dims(last_seq_scaled, axis=0)
            
            # Prediksi
            prob = model.predict(input_data, verbose=0)[0][0]
            price = df.iloc[-1]['Close']
            
            # Logic Signal
            # > 75% : Strong Buy (🔥)
            # > 60% : Potential Buy (🟢)
            # < 40% : Avoid/Sell (🔴)
            
            ticker_clean = t.replace('.JK', '')
            
            if prob >= 0.60:
                icon = "🔥" if prob >= 0.75 else "🟢"
                line = f"{icon} {ticker_clean:<5}| {int(price):<6}| {int(prob*100)}%"
                report_lines.append(line)
            # Debugging (Optional): Uncomment bawah ini kalau mau lihat yg jelek juga
            # elif prob < 0.40:
            #     line = f"🔴 {ticker_clean:<5}| {int(price):<6}| {int(prob*100)}%"
            #     report_lines.append(line)

        except Exception as e:
            continue

    # Sorting based on Probability
    report_lines.sort(key=lambda x: int(x.split('|')[2].replace('%','')), reverse=True)
    
    # Kirim Laporan
    date_now = datetime.now().strftime("%d-%m-%Y")
    msg = f"🧠 *EAGLE EYE AI v2.0*\n🗓 {date_now}\n"
    msg += f"_Architecture: Bi-LSTM + Attention_\n"
    msg += "—" * 15 + "\n"
    msg += "`Sts  Saham  Harga   Score`\n```\n"
    
    if report_lines:
        msg += "\n".join(report_lines[:15]) # Top 15 saja biar gak spam
    else:
        msg += "Pasar Konsolidasi. Tidak ada sinyal kuat."
        
    msg += "\n```\n_Score > 60% indicates Uptrend Potential_\n💡 *Disclaimer On*"
    
    send_telegram(msg)
    print("Selesai.")

if __name__ == "__main__":
    predict_daily()