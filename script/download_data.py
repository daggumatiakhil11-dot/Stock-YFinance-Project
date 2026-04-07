import pandas as pd
import yfinance as yf
import os
import time
import matplotlib.pyplot as plt

print("STEP 1: Starting monthly OHLCV data download...")


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_file = os.path.join(BASE_DIR, "input", "Fund and Index Returns.xlsx")
output_folder = os.path.join(BASE_DIR, "output")
charts_folder = os.path.join(BASE_DIR, "charts")

os.makedirs(output_folder, exist_ok=True)
os.makedirs(charts_folder, exist_ok=True)


df = pd.read_excel(input_file, sheet_name=0, header=None)
tickers = df.iloc[1, 1:].dropna().astype(str).str.strip().tolist()

print(f"Found {len(tickers)} tickers")


valid_tickers = []
invalid_tickers = []

for t in tickers:
    try:
        data = yf.Ticker(t).history(period="1mo")
        if not data.empty:
            valid_tickers.append(t)
            print(f"Valid: {t}")
        else:
            invalid_tickers.append(t)
            print(f"Invalid (empty): {t}")
    except Exception:
        invalid_tickers.append(t)
        print(f"Error: {t}")

    time.sleep(1)

print(f"\nValid: {len(valid_tickers)} | Invalid: {len(invalid_tickers)}")


final_df = pd.DataFrame()   # <-- important initialization

if valid_tickers:

    data = yf.download(
        valid_tickers,
        start="2018-10-31",
        end="2026-02-28",
        interval="1mo",
        auto_adjust=False,
        group_by="ticker",
        progress=True
    )

    all_data = []

    for ticker in valid_tickers:
        try:
            ticker_df = data[ticker].copy()

            if ticker_df.empty:
                continue

            ticker_df = ticker_df.reset_index()

            ticker_df = ticker_df[["Date", "Open", "High", "Low", "Close", "Volume"]]

            ticker_df["Ticker"] = ticker

            ticker_df = ticker_df[
                ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]
            ]

            ticker_df["Date"] = pd.to_datetime(ticker_df["Date"]) \
                .dt.to_period("M").dt.to_timestamp("M")

            all_data.append(ticker_df)

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df = final_df.sort_values(by=["Ticker", "Date"])

        
        print("Calculating analytics...")

        final_df["Date"] = pd.to_datetime(final_df["Date"])

        final_df["Monthly_Return"] = final_df.groupby("Ticker")["Close"].pct_change()

        final_df["MA_3"] = final_df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(3).mean())
        final_df["MA_6"] = final_df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(6).mean())
        final_df["MA_12"] = final_df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(12).mean())

        final_df["Volatility"] = final_df.groupby("Ticker")["Monthly_Return"].transform(lambda x: x.rolling(6).std())

        def calculate_cagr(group):
            group = group.dropna(subset=["Close"])
            if len(group) < 2:
                return None

            start_price = group.iloc[0]["Close"]
            end_price = group.iloc[-1]["Close"]

            years = (group.iloc[-1]["Date"] - group.iloc[0]["Date"]).days / 365.25

            if years <= 0:
                return None

            return (end_price / start_price) ** (1 / years) - 1

        cagr_dict = final_df.groupby("Ticker").apply(calculate_cagr).to_dict()
        final_df["CAGR"] = final_df["Ticker"].map(cagr_dict)

        # Convert to %
        final_df["Monthly_Return"] *= 100
        final_df["Volatility"] *= 100
        final_df["CAGR"] *= 100

        
        print("\nGenerating charts...")

        print("Total rows:", len(final_df))
        print("Tickers:", final_df["Ticker"].unique())

        for ticker in final_df["Ticker"].unique():
            try:
                df_ticker = final_df[final_df["Ticker"] == ticker].copy()

                if df_ticker.empty:
                    print(f"Skipping {ticker} (no data)")
                    continue

                print(f"Creating chart for {ticker} ({len(df_ticker)} rows)")

                df_ticker = df_ticker.dropna(subset=["Close"])
                df_ticker["Date"] = pd.to_datetime(df_ticker["Date"])

                plt.figure(figsize=(10, 5))

                plt.plot(df_ticker["Date"], df_ticker["Close"], label="Close")
                plt.plot(df_ticker["Date"], df_ticker["MA_3"], label="MA 3")
                plt.plot(df_ticker["Date"], df_ticker["MA_6"], label="MA 6")
                plt.plot(df_ticker["Date"], df_ticker["MA_12"], label="MA 12")

                plt.title(f"{ticker} Price Trend")
                plt.legend()

                chart_path = os.path.join(charts_folder, f"{ticker}.png")
                plt.savefig(chart_path)
                plt.close()

                print(f"Saved: {chart_path}")

            except Exception as e:
                print(f"Chart error for {ticker}: {e}")

        
        output_file = os.path.join(output_folder, "monthly_ohlcv_data.xlsx")
        final_df.to_excel(output_file, index=False)

        print(f"\nSUCCESS: File saved at {output_file}")

    else:
        print("No data to save")


if invalid_tickers:
    invalid_file = os.path.join(output_folder, "invalid_tickers.xlsx")
    pd.DataFrame({"Invalid Tickers": invalid_tickers}).to_excel(invalid_file, index=False)

    print(f"Invalid tickers saved at {invalid_file}")