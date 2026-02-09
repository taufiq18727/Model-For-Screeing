import yfinance as yf
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler

# --- KONFIGURASI CANGGIH ---
tickers = [
# SAHAM-SAHAM GORENGAN & SECOND LINER (Sering Spike)
    "BUMI", "DEWA", "BRMS", "PSAB", "ELSA", "ENRG", "MDKA", "TINS", "ANTM", "INCO",
    "ADMR", "DOID", "HRUM", "INDY", "ITMG", "PTBA", "ADRO", "MEDC", "PGAS", "AKRA",
    "GOTO", "BUKA", "EMTK", "SCMA", "ARTO", "BBHI", "AGRO", "BRIS", "PNBS", "BNBR",
    "ASII", "AUTO", "DRMA", "GJTL", "IMAS", "MPPA", "LPPF", "RALS", "ACES", "MAPI",
    "MNCN", "BMTR", "KPIG", "BHIT", "BCAP", "SAME", "SILO", "HEAL", "MIKA", "PRIM",
    "BBCA", "BBRI", "BMRI", "BBNI", "TLKM", "ISAT", "EXCL", "TOWR", "TBIG", "JSMR",
    "BSDE", "CTRA", "SMRA", "PWON", "ASRI", "PANI", "PTPP", "WIKA", "ADHI", "WEGE",
    "SRTG", "KIJA", "BEST", "APLN", "META", "LCGP", "HOKI", "CLEO", "WOOD", "MYOR"
]

SEQ_LEN = 45       # Dikurangi ke 45 hari (2 bulan trading) agar lebih responsif
PREDICT_DAYS = 3   # Target swing 3 hari ke depan
TARGET_GAIN = 0.015 # Target: Harga harus naik minimal 1.5% (Filter Noise)

def technical_features(df):
    df = df.copy()
    
    # 1. Log Returns (Penting untuk stasioneritas)
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # 2. Distance to MA20 (Mean Reversion)
    ma20 = df['Close'].rolling(window=20).mean()
    df['Dist_MA20'] = (df['Close'] - ma20) / ma20
    
    # 3. Bollinger Bands Width (Volatility Squeeze)
    std20 = df['Close'].rolling(window=20).std()
    upper = ma20 + (std20 * 2)
    lower = ma20 - (std20 * 2)
    df['BB_Width'] = (upper - lower) / ma20
    
    # 4. RSI (Normalized 0-1)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI_Norm'] = df['RSI'] / 100.0
    
    # 5. Volume Force (Volume * Price Change)
    df['Vol_Force'] = df['Log_Ret'] * (df['Volume'] / df['Volume'].rolling(20).mean())
    
    return df[['Log_Ret', 'Dist_MA20', 'BB_Width', 'RSI_Norm', 'Vol_Force']].dropna()

def create_dataset(data, seq_len):
    X, y = [], []
    # Convert dataframe to numpy array
    vals = data.values
    
    for i in range(len(vals) - seq_len - PREDICT_DAYS):
        # Input: Sequence 45 hari
        X.append(vals[i : i+seq_len])
        
        # Target Logic: 
        # Apakah Max High dalam 3 hari ke depan > Close hari ini + 1.5%?
        # Kita pakai Close hari ini vs Max High masa depan (Swing Potential)
        current_close = data.iloc[i+seq_len-1]['Close'] # Ambil dari DF asli jika memungkinkan, tapi disini kita pakai logika return
        
        # Karena kita sudah pakai Log Return, kita harus hitung akumulasi return
        # Simplifikasi: Jika Cumulative Return 3 hari ke depan > 1.5%
        future_ret = np.sum(vals[i+seq_len : i+seq_len+PREDICT_DAYS, 0]) # Index 0 adalah Log_Ret
        
        # Label 1 jika naik > 1.5%, else 0
        y.append(1 if future_ret > TARGET_GAIN else 0)
        
    return np.array(X), np.array(y)

def build_advanced_model(input_shape):
    # Input
    inputs = Input(shape=input_shape)
    
    # Bidirectional LSTM (Melihat masa lalu & konteks)
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Attention Mechanism (Fokus pada hari-hari penting dalam sequence)
    # Query, Value, Key approach
    attn_output = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = x + attn_output # Residual connection
    x = LayerNormalization()(x)
    
    # Global Pooling (Meringkas sequence menjadi satu vektor)
    x = GlobalAveragePooling1D()(x)
    
    # Dense Layers
    x = Dense(64, activation='swish')(x) # Swish lebih bagus dari Relu untuk deep learning
    x = Dropout(0.2)(x)
    x = Dense(32, activation='swish')(x)
    
    # Output
    outputs = Dense(1, activation='sigmoid')(inputs)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.AUC()])
    return model

def train():
    print("🚀 Memulai Training Deep Learning V2...")
    
    # Download Data
    raw_data = yf.download(tickers, period="5y", auto_adjust=True, group_by='ticker', threads=True)
    
    processed_X = []
    processed_y = []
    
    scaler = RobustScaler() # Lebih tahan outlier daripada MinMaxScaler
    
    # Fase 1: Feature Engineering per Ticker
    print("🛠️ Feature Engineering...")
    temp_dfs = []
    
    for t in tickers:
        try:
            df = raw_data[t].copy()
            if len(df) < 200: continue
            
            # Hitung technical features
            feat_df = technical_features(df)
            
            # Simpan data asli (Log_Ret) untuk penentuan Target Label
            # Namun untuk input X, kita butuh scaling.
            # Trik: Kita scaling belakangan setelah gabung semua agar distribusinya rata
            temp_dfs.append(feat_df)
            
        except Exception as e:
            continue

    if not temp_dfs:
        print("❌ Data kosong.")
        return

    # Gabung semua untuk fitting scaler (Global Generalization)
    all_data = pd.concat(temp_dfs)
    scaler.fit(all_data)
    joblib.dump(scaler, 'scaler_v2.pkl')
    
    # Fase 2: Sequence Creation
    print("✂️ Creating Sequences...")
    for df in temp_dfs:
        # Scale dulu
        scaled_vals = scaler.transform(df)
        
        # Perlu Log_Ret asli untuk target? 
        # Di function create_dataset di atas, kita pakai index 0 (Log_Ret) dari scaled data.
        # Masalah: Scaled Log_Ret tidak mencerminkan % asli.
        # Solusi: Kita modifikasi create_dataset untuk pakai threshold scaled atau logic lain.
        # AGAR LEBIH AKURAT: Kita buat y secara manual sebelum scaling.
        
        # Re-logic:
        vals = df.values # Data belum di scale
        X_list, y_list = [], []
        
        for i in range(len(vals) - SEQ_LEN - PREDICT_DAYS):
            # Target calculation pakai raw data
            cumulative_return = np.sum(vals[i+SEQ_LEN : i+SEQ_LEN+PREDICT_DAYS, 0]) # Col 0 is Log_Ret
            label = 1 if cumulative_return > TARGET_GAIN else 0
            
            # Input data pakai data yg akan di scale nanti
            X_list.append(vals[i : i+SEQ_LEN])
            y_list.append(label)
            
        if len(X_list) > 0:
            # Scale X sekarang
            X_arr = np.array(X_list) # (N, 45, 5)
            N, T, F = X_arr.shape
            X_reshaped = X_arr.reshape(N * T, F)
            X_scaled = scaler.transform(X_reshaped).reshape(N, T, F)
            
            processed_X.append(X_scaled)
            processed_y.append(np.array(y_list))

    final_X = np.vstack(processed_X)
    final_y = np.concatenate(processed_y)
    
    # Shuffle data agar tidak bias urutan saham
    indices = np.arange(final_X.shape[0])
    np.random.shuffle(indices)
    final_X = final_X[indices]
    final_y = final_y[indices]

    print(f"🧠 Training Data Shape: {final_X.shape}")
    print(f"⚖️ Class Balance: {np.mean(final_y)*100:.2f}% Positive Samples")

    # Build Model
    model = build_advanced_model((SEQ_LEN, 5))
    
    # Callbacks
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

    # Train
    model.fit(
        final_X, final_y,
        epochs=30, # Bisa sampai 30, stop otomatis jika tidak improve
        batch_size=64,
        validation_split=0.15,
        callbacks=[es, lr],
        verbose=1
    )
    
    model.save('model_advanced.keras')
    print("✅ Model Advanced Tersimpan.")

if __name__ == "__main__":
    train()
