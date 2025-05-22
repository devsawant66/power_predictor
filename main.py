import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load and preprocess data ---
df = pd.read_csv(r"C:\Users\DEV\Downloads\Processed_Power_Data.csv")
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df[df['Timestamp'].dt.year == 2020]

# Time-based features
df['Hour'] = df['Timestamp'].dt.hour
df['Minute'] = df['Timestamp'].dt.minute
df['DayOfWeek'] = df['Timestamp'].dt.weekday
df['Month'] = df['Timestamp'].dt.month
df['DayOfYear'] = df['Timestamp'].dt.dayofyear
df['WeekOfYear'] = df['Timestamp'].dt.isocalendar().week.astype(int)

# Season feature
def assign_season(month):
    if month in [3, 4, 5]:
        return 0  # Summer
    elif month in [6, 7, 8, 9]:
        return 1  # Monsoon
    else:
        return 2  # Winter
df['Season'] = df['Month'].apply(assign_season)

# Festival feature
festival_dates = {
    '2020-01-15', '2020-02-21', '2020-03-9',
    '2020-03-10', '2020-03-25', '2020-04-2','2020-04-8','2020-07-5','2020-07-25','2020-08-3','2020-08-22','2020-10-17','2020-10-25','2020-11-16'
}
df['Date'] = df['Timestamp'].dt.date
df['Festival'] = df['Date'].astype(str).isin(festival_dates).astype(int)

# Sort dataframe
df.sort_values(['Timestamp'], inplace=True)

# Remove negatives and cap outliers
df = df[df['Power Consumption'] >= 0]
upper_limit = df['Power Consumption'].quantile(0.995)
df['Power Consumption'] = df['Power Consumption'].clip(upper=upper_limit)
print(f"✅ Cleaned data: Removed negatives and capped outliers above {upper_limit:,.2f}")

# Features and target (log-transformed)
features = ['BlockNo', 'Hour', 'Minute', 'DayOfWeek', 'Month',
            'DayOfYear', 'WeekOfYear', 'Season', 'Festival',
            'Lag1', 'Lag96', 'RollingMean_4', 'RollingStd_4']

# For training, create lag features statically (to train model)
df['Lag1'] = df['Power Consumption'].shift(1)
df['Lag96'] = df['Power Consumption'].shift(96)
df['RollingMean_4'] = df['Power Consumption'].rolling(window=4).mean()
df['RollingStd_4'] = df['Power Consumption'].rolling(window=4).std()
df.dropna(inplace=True)

X = df[features]
y = np.log1p(df['Power Consumption'])  # log transform target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train LightGBM model
model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='rmse',
    callbacks=[
        early_stopping(stopping_rounds=50),
        log_evaluation(period=50)
    ]
)

# Predict and evaluate on test set
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test_actual = np.expm1(y_test)

mae = mean_absolute_error(y_test_actual, y_pred)
rmse = sqrt(mean_squared_error(y_test_actual, y_pred))
r2 = r2_score(y_test_actual, y_pred)

print(f"\n✅ LightGBM model trained with log-transformed target")
print(f"📉 MAE: {mae:.2f}")
print(f"📉 RMSE: {rmse:.2f}")
print(f"📈 R² Score: {r2:.4f}")

# --- Recursive lag feature helper ---
def get_lag_features(date, block, hist_df, pred_cache):
    """
    Get lag features for a given date and block.
    pred_cache: dict with keys (date, block) and predicted power values.
    hist_df: historical dataframe.
    """
    prev_block = block - 1
    prev_date = date
    if prev_block < 1:
        prev_block = 96
        prev_date = date - timedelta(days=1)

    # Lag1
    lag1 = pred_cache.get((prev_date, prev_block))
    if lag1 is None:
        row = hist_df[(hist_df['Date'] == prev_date) & (hist_df['BlockNo'] == prev_block)]
        lag1 = row['Power Consumption'].values[0] if not row.empty else np.nan

    # Lag96 (same block previous day)
    lag96_date = date - timedelta(days=1)
    lag96 = pred_cache.get((lag96_date, block))
    if lag96 is None:
        row = hist_df[(hist_df['Date'] == lag96_date) & (hist_df['BlockNo'] == block)]
        lag96 = row['Power Consumption'].values[0] if not row.empty else np.nan

    # Rolling stats over last 4 blocks (handle day wrap)
    power_vals = []
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
        power_vals.append(val)

    rolling_vals = np.array(power_vals)
    rolling_mean_4 = np.nanmean(rolling_vals) if not np.all(np.isnan(rolling_vals)) else np.nan
    rolling_std_4 = np.nanstd(rolling_vals) if not np.all(np.isnan(rolling_vals)) else np.nan

    return lag1, lag96, rolling_mean_4, rolling_std_4

# --- Feature creation for user input with recursive lags ---
def create_features(date, block, hist_df, pred_cache, festival_dates):
    hour = (block - 1) // 4
    minute = ((block - 1) % 4) * 15
    lag1, lag96, rolling_mean_4, rolling_std_4 = get_lag_features(date, block, hist_df, pred_cache)

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
        'RollingMean_4': rolling_mean_4,
        'RollingStd_4': rolling_std_4
    }
    return pd.DataFrame([features])

# --- Recursive prediction for one block ---
def recursive_predict(date, block, hist_df, model, festival_dates, pred_cache):
    X_input = create_features(date, block, hist_df, pred_cache, festival_dates)
    block_val = int(X_input.loc[0, 'BlockNo'])  # Ensure it's an integer

    for col in ['Lag1', 'Lag96', 'RollingMean_4', 'RollingStd_4']:
     if pd.isna(X_input.loc[0, col]):
            
        if col in ['Lag1', 'Lag96']:
            fallback_series = df[df['BlockNo'] == block_val]['Power Consumption']
            fallback = fallback_series.mean() if not fallback_series.empty else df['Power Consumption'].mean()
        elif col == 'RollingMean_4':
            fallback = df['RollingMean_4'].mean(skipna=True)
        elif col == 'RollingStd_4':
            fallback = df['RollingStd_4'].mean(skipna=True)
        else:
            fallback = df['Power Consumption'].mean()

        X_input.loc[0, col] = fallback



    pred_log = model.predict(X_input)[0]
    pred = np.expm1(pred_log)
    pred_cache[(date, block)] = pred
    return pred

# --- Initialize prediction cache with historical known data ---
pred_cache = {}
for idx, row in df.iterrows():
    pred_cache[(row['Date'], row['BlockNo'])] = row['Power Consumption']

# --- User input ---
user_date_str = input("\n📅 Enter a date (YYYY-MM-DD): ")
user_date = datetime.strptime(user_date_str, "%Y-%m-%d").date()

while True:
    try:
        user_block = int(input("⏱ Enter Block Number (1–96): "))
        if 1 <= user_block <= 96:
            break
        else:
            print("❌ Invalid block number! Please enter 1-96.")
    except ValueError:
        print("⚠ Please enter a valid integer.")

all_festival_dates = festival_dates.union({
    '2026-01-15', '2026-03-10', '2026-04-02', '2026-08-11',
    '2026-10-25', '2026-11-14', '2026-12-25'
})



# Predict all previous blocks to build proper context (Lag1, RollingMean, etc.)
for b in range(1, user_block):
    recursive_predict(user_date, b, df, model, all_festival_dates, pred_cache)

# Now predict the user-specified block using updated cache
prediction = recursive_predict(user_date, user_block, df, model, all_festival_dates, pred_cache)

print(f"\n🔮 Predicted Power Consumption on {user_date} Block {user_block}: {prediction:.2f} units")

# --- Actual vs predicted if available ---
actual_row = df[(df['Date'] == user_date) & (df['BlockNo'] == user_block)]
if not actual_row.empty:
    actual_value = actual_row['Power Consumption'].values[0]
    error = prediction - actual_value
    percent_error = (error / actual_value) * 100
    print(f"✅ Actual Value Found: {actual_value:.2f} units")
    print(f"📉 Absolute Error: {error:.2f}")
    print(f"📊 Percent Error: {percent_error:.2f}%")
else:
    print("⚠ Actual value not found in dataset.")

# --- Visualization ---
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test_actual, y=y_pred, alpha=0.5)
plt.xlabel("Actual Power Consumption")
plt.ylabel("Predicted Power Consumption")
plt.title("🔍 Actual vs Predicted - Test Set (log-transformed target)")
plt.plot([y_test_actual.min(), y_test_actual.max()], [y_test_actual.min(), y_test_actual.max()], 'r--')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Full 2020 Dataset Predictions ---
y_full_log = model.predict(X)
y_full_pred = np.expm1(y_full_log)
y_actual_full = np.expm1(y)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_actual_full, y=y_full_pred, alpha=0.4, color='purple')
plt.xlabel("Actual Power Consumption (2020)")
plt.ylabel("Predicted Power Consumption (2020)")
plt.title("📊 Actual vs Predicted Power Consumption - Full 2020 Dataset")
plt.plot([y_actual_full.min(), y_actual_full.max()], [y_actual_full.min(), y_actual_full.max()], 'r--')
plt.grid(True)
plt.tight_layout()
plt.show()
