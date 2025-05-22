import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split



st.title("🔌 Power Consumption Prediction")

# 1. Upload dataset
st.sidebar.header("📂 Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV (must contain 'Timestamp', 'BlockNo', 'Power Consumption')", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Using default dataset since no file uploaded.")
    df = pd.read_csv(r"C:\Users\DEV\Downloads\Processed_Power_Data.csv")

# 2. Preprocessing
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Date'] = df['Timestamp'].dt.date
df = df[df['Timestamp'].dt.year == 2020]

df['Hour'] = df['Timestamp'].dt.hour
df['Minute'] = df['Timestamp'].dt.minute
df['DayOfWeek'] = df['Timestamp'].dt.weekday
df['Month'] = df['Timestamp'].dt.month
df['DayOfYear'] = df['Timestamp'].dt.dayofyear
df['WeekOfYear'] = df['Timestamp'].dt.isocalendar().week.astype(int)

def assign_season(month):
    if month in [3, 4, 5]:
        return 0
    elif month in [6, 7, 8, 9]:
        return 1
    else:
        return 2
df['Season'] = df['Month'].apply(assign_season)

festival_dates = {
    '2020-01-15', '2020-02-21', '2020-03-9',
    '2020-03-10', '2020-03-25', '2020-04-2','2020-04-8',
    '2020-07-5','2020-07-25','2020-08-3','2020-08-22',
    '2020-10-17','2020-10-25','2020-11-16'
}
df['Festival'] = df['Date'].astype(str).isin(festival_dates).astype(int)

# Lag and rolling features
df['Lag1'] = df['Power Consumption'].shift(1)
df['Lag96'] = df['Power Consumption'].shift(96)
df['RollingMean_4'] = df['Power Consumption'].rolling(window=4).mean()
df['RollingStd_4'] = df['Power Consumption'].rolling(window=4).std()
df.dropna(inplace=True)

# Features
features = ['BlockNo', 'Hour', 'Minute', 'DayOfWeek', 'Month',
            'DayOfYear', 'WeekOfYear', 'Season', 'Festival',
            'Lag1', 'Lag96', 'RollingMean_4', 'RollingStd_4']
X = df[features]
y = np.log1p(df['Power Consumption'])

# Train model
model = lgb.LGBMRegressor(
    n_estimators=500, learning_rate=0.05,
    max_depth=7, num_leaves=31,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Prediction helper functions
def get_lag_features(date, block, hist_df, pred_cache):
    prev_block = block - 1
    prev_date = date
    if prev_block < 1:
        prev_block = 96
        prev_date = date - timedelta(days=1)

    lag1 = pred_cache.get((prev_date, prev_block))
    if lag1 is None:
        row = hist_df[(hist_df['Date'] == prev_date) & (hist_df['BlockNo'] == prev_block)]
        lag1 = row['Power Consumption'].values[0] if not row.empty else np.nan

    lag96_date = date - timedelta(days=1)
    lag96 = pred_cache.get((lag96_date, block))
    if lag96 is None:
        row = hist_df[(hist_df['Date'] == lag96_date) & (hist_df['BlockNo'] == block)]
        lag96 = row['Power Consumption'].values[0] if not row.empty else np.nan

    rolling = []
    for i in range(1, 5):
        b = block - i
        d = date
        if b < 1:
            b += 96
            d -= timedelta(days=1)
        val = pred_cache.get((d, b))
        if val is None:
            row = hist_df[(hist_df['Date'] == d) & (hist_df['BlockNo'] == b)]
            val = row['Power Consumption'].values[0] if not row.empty else np.nan
        rolling.append(val)

    return lag1, lag96, np.nanmean(rolling), np.nanstd(rolling)

def create_input(date, block, hist_df, pred_cache):
    hour = (block - 1) // 4
    minute = ((block - 1) % 4) * 15
    lag1, lag96, rmean, rstd = get_lag_features(date, block, hist_df, pred_cache)

    features = {
        'BlockNo': block,
        'Hour': hour,
        'Minute': minute,
        'DayOfWeek': date.weekday(),
        'Month': date.month,
        'DayOfYear': date.timetuple().tm_yday,
        'WeekOfYear': date.isocalendar()[1],
        'Season': assign_season(date.month),
        'Festival': int(str(date) in festival_dates),
        'Lag1': lag1,
        'Lag96': lag96,
        'RollingMean_4': rmean,
        'RollingStd_4': rstd
    }

    for col in ['Lag1', 'Lag96', 'RollingMean_4', 'RollingStd_4']:
        if pd.isna(features[col]):
            features[col] = df[df['BlockNo'] == block]['Power Consumption'].mean()

    return pd.DataFrame([features])

def predict_block(date, block, hist_df, model, pred_cache):
    X_input = create_input(date, block, hist_df, pred_cache)
    pred = np.expm1(model.predict(X_input)[0])
    pred_cache[(date, block)] = pred
    return pred

# 3. User input
st.subheader("🔍 Make a Prediction")

col1, col2 = st.columns([1, 1])

with col1:
    user_date = st.date_input("Select Date", datetime(2020, 1, 1), format="YYYY-MM-DD")

with col2:
    mode = st.radio("Prediction Mode", ["Single Block", "All 96 Blocks"], horizontal=True)


if mode == "Single Block":
    block = st.number_input("Block Number (1–96)", min_value=1, max_value=96, value=1)
    pred_cache = {(row['Date'], row['BlockNo']): row['Power Consumption'] for _, row in df.iterrows()}

    for b in range(1, block):
        predict_block(user_date, b, df, model, pred_cache)

    pred = predict_block(user_date, block, df, model, pred_cache)
    st.success(f"Predicted Power on {user_date}, Block {block}: **{pred:.2f}** units")

    actual = df[(df['Date'] == user_date) & (df['BlockNo'] == block)]
    if not actual.empty:
        actual_val = actual['Power Consumption'].values[0]
        error = pred - actual_val
        percent = (error / actual_val) * 100
        st.write(f"📊 Actual: **{actual_val:.2f}** | Error: **{error:.2f}** | % Error: **{percent:.2f}%**")
    else:
        st.info("Actual value not found for comparison.")

else:
    pred_cache = {(row['Date'], row['BlockNo']): row['Power Consumption'] for _, row in df.iterrows()}
    preds = []
    for b in range(1, 97):
        pred = predict_block(user_date, b, df, model, pred_cache)
        preds.append((b, pred))

    pred_df = pd.DataFrame(preds, columns=["Block", "Predicted Power"])

    # Try to fetch actuals for error comparison
    actual_df = df[df['Date'] == user_date][['BlockNo', 'Power Consumption']].rename(columns={'BlockNo': 'Block', 'Power Consumption': 'Actual Power'})
    final_df = pd.merge(pred_df, actual_df, on="Block", how="left")
    final_df['Absolute Error'] = final_df['Predicted Power'] - final_df['Actual Power']
    final_df['% Error'] = (final_df['Absolute Error'] / final_df['Actual Power']) * 100

    st.dataframe(final_df.style.format({"Predicted Power": "{:.2f}", "Actual Power": "{:.2f}", "Absolute Error": "{:.2f}", "% Error": "{:.2f}"}))


if not actual.empty:
    import matplotlib.pyplot as plt

    # Prepare data for plotting
    fig, ax = plt.subplots()
    ax.bar(['Actual', 'Predicted'], [actual_val, pred], color=['blue', 'orange'])
    ax.set_ylabel('Power Consumption')
    ax.set_title(f'Actual vs Predicted Power (Block {block} on {user_date})')
    st.pyplot(fig)



















































































































