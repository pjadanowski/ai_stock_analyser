import time
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from html_report_generator import HtmlReportGenerator

class StockAnalyzer:
    def __init__(self):
        self.tickers = ["META", "GOOGL", "AMZN", "PYPL", "NVDA", "MSFT", "AAPL", "BKNG", "SHOP"]
        self.indicators = {
            'SMA': [20, 50, 200],
            'RSI': 14,
            'Stochastic': 14,
            'MACD': (12, 26, 9),
            'Bollinger': 20
        }
        self.timeout = 15  # Zwiększony timeout
        self.retries = 3  # Liczba prób pobrania danych

    def fetch_stock_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Pobiera dane giełdowe z wieloma zabezpieczeniami"""
        for attempt in range(1, self.retries + 1):
            try:
                print(f"Pobieranie danych dla {ticker}, próba {attempt}/{self.retries}")

                # Pobierz dane z yfinance z większym timeoutem
                data = yf.download(
                    tickers=ticker,
                    period="1y",
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    timeout=15,
                    group_by='ticker'
                )

                # Sprawdź czy otrzymaliśmy dane
                if data is None or data.empty:
                    print(f"Otrzymano pusty DataFrame dla {ticker}")
                    continue

                # Sprawdź czy mamy kolumnę Close (czasem yfinance zwraca MultiIndex)
                if isinstance(data.columns, pd.MultiIndex):
                    data = data.droplevel(0, axis=1)

                if 'Close' not in data.columns:
                    print(f"Brak kolumny 'Close' w danych dla {ticker}. Dostępne kolumny: {data.columns.tolist()}")
                    continue

                # Usuń wiersze z brakującymi wartościami Close
                data = data.dropna(subset=['Close'])

                # Sprawdź czy mamy wystarczająco danych
                if len(data) < 20:
                    print(f"Zbyt mało danych dla {ticker} ({len(data)} rekordów)")
                    continue

                # Wypełnij brakujące wartości dla innych kolumn
                cols_to_fill = ['Open', 'High', 'Low', 'Volume']
                for col in cols_to_fill:
                    if col in data.columns:
                        data[col] = data[col].ffill()

                print(f"Pomyślnie pobrano {len(data)} rekordów dla {ticker}")
                return data

            except Exception as e:
                print(f"Błąd podczas próby {attempt} dla {ticker}: {str(e)}")
                if attempt < self.retries:
                    print(f"Ponowna próba za 2 sekundy...")
                    time.sleep(2)

        print(f"Nie udało się pobrać danych dla {ticker} po {self.retries} próbach")

        # Ostateczna próba pobrania tylko aktualnej ceny
        try:
            print(f"Próba pobrania tylko aktualnej ceny dla {ticker}")
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(period='1d')
            if not hist.empty and 'Close' in hist.columns:
                current_price = hist['Close'].iloc[-1]
                print(f"Udało się pobrać aktualną cenę: {current_price}")
                return pd.DataFrame({'Close': [current_price]}, index=[datetime.now()])
        except Exception as e:
            print(f"Błąd podczas pobierania aktualnej ceny: {str(e)}")

        return None

    def calculate_technical_indicators(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Oblicza wskaźniki techniczne z dodatkowymi zabezpieczeniami"""
        if data is None or len(data) < 20:
            return None

        try:
            df = data.copy()

            # Obliczanie SMA
            for period in self.indicators['SMA']:
                if len(df) >= period:
                    df[f'SMA_{period}'] = df['Close'].rolling(window=period, min_periods=1).mean()

            # Obliczanie RSI tylko jeśli mamy wystarczająco danych
            if len(df) >= self.indicators['RSI']:
                delta = df['Close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)

                avg_gain = gain.rolling(window=self.indicators['RSI'], min_periods=1).mean()
                avg_loss = loss.rolling(window=self.indicators['RSI'], min_periods=1).mean()

                rs = np.where(avg_loss != 0, avg_gain / avg_loss, np.nan)
                df['RSI'] = 100 - (100 / (1 + rs))

            # Obliczanie Stochastic
            if len(df) >= self.indicators['Stochastic']:
                low = df['Low'].rolling(window=self.indicators['Stochastic'], min_periods=1).min()
                high = df['High'].rolling(window=self.indicators['Stochastic'], min_periods=1).max()

                df['%K'] = np.where((high - low) != 0, 100 * ((df['Close'] - low) / (high - low)), np.nan)
                df['%D'] = df['%K'].rolling(window=3, min_periods=1).mean()

            # Obliczanie MACD
            exp1 = df['Close'].ewm(span=self.indicators['MACD'][0], adjust=False).mean()
            exp2 = df['Close'].ewm(span=self.indicators['MACD'][1], adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal'] = df['MACD'].ewm(span=self.indicators['MACD'][2], adjust=False).mean()

            # Obliczanie Bollinger Bands
            if len(df) >= self.indicators['Bollinger']:
                df['BB_Middle'] = df['Close'].rolling(window=self.indicators['Bollinger'], min_periods=1).mean()
                std = df['Close'].rolling(window=self.indicators['Bollinger'], min_periods=1).std()
                df['BB_Upper'] = df['BB_Middle'] + 2 * std
                df['BB_Lower'] = df['BB_Middle'] - 2 * std

            return df.dropna()

        except Exception as e:
            print(f"Błąd podczas obliczania wskaźników: {str(e)}")
            return None

    def evaluate_signals(self, df: pd.DataFrame) -> Dict[str, Optional[bool]]:
        """Ocenia sygnały techniczne z zabezpieczeniami"""
        if df is None or df.empty:
            return {}

        try:
            last_row = df.iloc[-1]
            signals = {}

            # SMA Crossovers
            signals['SMA_20_50'] = last_row.get('SMA_20', np.nan) > last_row.get('SMA_50', np.nan) if pd.notna(
                last_row.get('SMA_20')) and pd.notna(last_row.get('SMA_50')) else None
            signals['SMA_50_200'] = last_row.get('SMA_50', np.nan) > last_row.get('SMA_200', np.nan) if pd.notna(
                last_row.get('SMA_50')) and pd.notna(last_row.get('SMA_200')) else None

            # RSI
            rsi = last_row.get('RSI', np.nan)
            signals['RSI'] = 30 < rsi < 70 if pd.notna(rsi) else None

            # Stochastic
            k = last_row.get('%K', np.nan)
            d = last_row.get('%D', np.nan)
            signals['Stochastic'] = (k > d) and (k < 80) if pd.notna(k) and pd.notna(d) else None

            # MACD
            macd = last_row.get('MACD', np.nan)
            signal = last_row.get('Signal', np.nan)
            signals['MACD'] = macd > signal if pd.notna(macd) and pd.notna(signal) else None

            # Bollinger Bands
            close = last_row.get('Close', np.nan)
            bb_lower = last_row.get('BB_Lower', np.nan)
            signals['Bollinger'] = close < bb_lower * 1.05 if pd.notna(close) and pd.notna(bb_lower) else None

            return signals

        except Exception as e:
            print(f"Błąd podczas oceny sygnałów: {str(e)}")
            return {}

    def get_growth_rate(self, ticker: str) -> Tuple[float, bool]:
        """Pobiera historyczną stopę wzrostu EPS"""
        try:
            stock = yf.Ticker(ticker)

            # Próba pobrania danych finansowych
            financials = stock.financials
            if financials is not None and not financials.empty and 'Net Income' in financials.index:
                # Pobierz historyczne wartości Net Income
                net_income = financials.loc['Net Income']
                if len(net_income) >= 2:
                    # Oblicz CAGR (Compound Annual Growth Rate)
                    oldest = net_income.iloc[-1]
                    newest = net_income.iloc[0]
                    years = len(net_income) - 1
                    if oldest > 0 and newest > 0:
                        growth_rate = (newest / oldest) ** (1 / years) - 1
                        return min(0.15, max(0.03, growth_rate)), True  # Ograniczenie do 3-15%

            # Jeśli nie udało się obliczyć na podstawie danych finansowych
            earnings_growth = stock.info.get('earningsGrowth')
            if earnings_growth and not np.isnan(earnings_growth):
                return min(0.15, max(0.03, earnings_growth)), True

            # Sprawdź przychody jako alternatywę
            revenue_growth = stock.info.get('revenueGrowth')
            if revenue_growth and not np.isnan(revenue_growth):
                return min(0.15, max(0.03, revenue_growth)), True

            # Domyślna wartość jeśli nie można obliczyć
            return 0.05, False
        except Exception as e:
            print(f"Błąd podczas pobierania stopy wzrostu dla {ticker}: {str(e)}")
            return 0.05, False

    def get_discount_rate(self, ticker: str) -> float:
        """Określa stopę dyskontową na podstawie dostępnych danych"""
        try:
            stock = yf.Ticker(ticker)

            # Próba pobrania beta
            beta = stock.info.get('beta')
            if beta and not np.isnan(beta):
                # Model CAPM: Rf + Beta * (Rm - Rf)
                risk_free_rate = 0.04  # 4% jako stopa wolna od ryzyka
                market_premium = 0.06  # 6% jako premia rynkowa
                discount_rate = risk_free_rate + beta * market_premium
                return min(0.15, max(0.08, discount_rate))  # Ograniczenie do 8-15%

            return 0.10  # Domyślna stopa dyskontowa 10%
        except Exception as e:
            print(f"Błąd podczas określania stopy dyskontowej dla {ticker}: {str(e)}")
            return 0.10

    def calculate_intrinsic_value(self, ticker: str) -> Optional[float]:
        """Oblicza wartość wewnętrzną z ulepszonymi metodami i DCF"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            methods = []

            # Metoda 1: DCF na podstawie Free Cash Flow
            if 'freeCashflow' in info and 'sharesOutstanding' in info:
                fcf = info.get('freeCashflow')
                shares = info.get('sharesOutstanding')

                if fcf and shares and shares > 0 and not np.isnan(fcf) and not np.isnan(shares):
                    fcf_per_share = fcf / shares

                    # Pobierz stopę wzrostu
                    growth_rate, is_reliable = self.get_growth_rate(ticker)
                    discount_rate = self.get_discount_rate(ticker)

                    # Model DCF na 5 lat + wartość terminalna
                    terminal_multiple = 15  # Mnożnik wartości końcowej
                    future_fcf = 0

                    # Oblicz zdyskontowane przepływy pieniężne na 5 lat
                    for year in range(1, 6):
                        future_value = fcf_per_share * (1 + growth_rate) ** year
                        present_value = future_value / (1 + discount_rate) ** year
                        future_fcf += present_value

                    # Wartość terminalna (zdyskontowana)
                    terminal_value = (fcf_per_share * (1 + growth_rate) ** 5 * terminal_multiple) / (
                                1 + discount_rate) ** 5

                    # Całkowita wartość DCF
                    dcf_value = future_fcf + terminal_value
                    methods.append(dcf_value)

            # Metoda 2: Zdyskontowane zyski (EPS)
            if 'trailingEps' in info and info['trailingEps'] and not np.isnan(info['trailingEps']):
                eps = info['trailingEps']
                growth_rate, _ = self.get_growth_rate(ticker)
                discount_rate = self.get_discount_rate(ticker)

                future_eps = 0
                for year in range(1, 6):
                    future_value = eps * (1 + growth_rate) ** year
                    present_value = future_value / (1 + discount_rate) ** year
                    future_eps += present_value

                # Dodaj wartość terminalną (mnożnik P/E)
                pe_multiple = 15
                terminal_value = (eps * (1 + growth_rate) ** 5 * pe_multiple) / (1 + discount_rate) ** 5
                eps_value = future_eps + terminal_value

                methods.append(eps_value)

            # Metoda 3: Wartość księgowa z premią za wzrost
            if 'bookValue' in info and info['bookValue'] and not np.isnan(info['bookValue']):
                book_value = info['bookValue']
                growth_rate, is_reliable = self.get_growth_rate(ticker)

                # Premia za wzrost - dla wyższych stóp wzrostu większa premia
                growth_premium = 1 + (growth_rate * 20)  # Mnożnik 1-4 w zależności od wzrostu
                book_value_method = book_value * growth_premium

                methods.append(book_value_method)

            # Metoda 4: Oparta na przychodach
            if 'revenue' in info and 'sharesOutstanding' in info:
                revenue = info.get('revenue')
                shares = info.get('sharesOutstanding')

                if revenue and shares and shares > 0 and not np.isnan(revenue) and not np.isnan(shares):
                    revenue_per_share = revenue / shares

                    # Szacowany mnożnik przychodów (P/S) dla sektora
                    sector = info.get('sector', '')

                    # Przykładowe mnożniki P/S dla różnych sektorów
                    ps_multiples = {
                        'Technology': 5,
                        'Consumer Cyclical': 2,
                        'Healthcare': 4,
                        'Communication Services': 3,
                        'Financial Services': 2,
                        'Industrials': 2,
                        'Consumer Defensive': 1.5,
                        'Energy': 1.5,
                        'Basic Materials': 1.5,
                        'Real Estate': 6,
                        'Utilities': 3
                    }

                    ps_ratio = ps_multiples.get(sector, 2.5)  # Domyślny mnożnik jeśli sektor nieznany
                    revenue_method = revenue_per_share * ps_ratio

                    methods.append(revenue_method)

            # Średnia ważona wszystkich metod
            if methods:
                # Jeśli mamy więcej niż jedną metodę, używamy średniej ważonej
                if len(methods) > 1:
                    # Przypisujemy wagi: DCF i EPS mają większą wagę niż inne metody
                    weights = []
                    for i, _ in enumerate(methods):
                        if i == 0 and 'freeCashflow' in info:  # DCF
                            weights.append(0.4)
                        elif i == 1 and 'trailingEps' in info:  # EPS
                            weights.append(0.3)
                        else:
                            weights.append(0.3 / (len(methods) - 2)) if len(methods) > 2 else weights.append(0.3)

                    # Normalizacja wag, aby sumowały się do 1
                    weights = [w / sum(weights) for w in weights]

                    # Obliczenie średniej ważonej
                    intrinsic_value = sum(m * w for m, w in zip(methods, weights))
                else:
                    intrinsic_value = methods[0]

                # Zwróć wynik z odpowiednią dokładnością
                return round(intrinsic_value, 2)

            return None

        except Exception as e:
            print(f"Błąd podczas obliczania wartości wewnętrznej dla {ticker}: {str(e)}")
            return None

    def analyze(self) -> None:
        """Główna metoda wykonująca analizę"""
        results = []

        for ticker in self.tickers:
            try:
                print(f"\nAnaliza dla {ticker}...")

                # Pobierz dane
                print("Pobieranie danych...")
                data = self.fetch_stock_data(ticker)

                if data is None:
                    print(f"Brak danych dla {ticker}")
                    continue

                print(f"Pobrano {len(data)} rekordów")

                # Oblicz wskaźniki
                print("Obliczanie wskaźników...")
                indicators = self.calculate_technical_indicators(data)

                if indicators is None:
                    print(f"Nie udało się obliczyć wskaźników dla {ticker}")
                    # Dodaj podstawowe informacje jeśli mamy tylko cenę
                    if len(data) >= 1:
                        results.append({
                            'ticker': ticker,
                            'price': data['Close'].iloc[-1],
                            'intrinsic': None,
                            'signals': {},
                            'indicator_values': {},
                            'score': 0
                        })
                    continue

                # Oceń sygnały
                signals = self.evaluate_signals(indicators)
                valid_signals = [s for s in signals.values() if s is not None]
                score = (sum(valid_signals) / len(valid_signals)) * 100 if valid_signals else 0

                # Pobierz wartości wskaźników
                last_row = indicators.iloc[-1]
                indicator_values = {
                    'SMA_20': last_row.get('SMA_20'),
                    'SMA_50': last_row.get('SMA_50'),
                    'SMA_200': last_row.get('SMA_200'),
                    'RSI': last_row.get('RSI'),
                    '%K': last_row.get('%K'),
                    '%D': last_row.get('%D'),
                    'MACD': last_row.get('MACD'),
                    'Signal': last_row.get('Signal'),
                    'BB_Upper': last_row.get('BB_Upper'),
                    'BB_Middle': last_row.get('BB_Middle'),
                    'BB_Lower': last_row.get('BB_Lower')
                }

                # Oblicz wartość wewnętrzną
                intrinsic = self.calculate_intrinsic_value(ticker)

                # Dodaj wynik
                results.append({
                    'ticker': ticker,
                    'price': last_row['Close'],
                    'intrinsic': intrinsic,
                    'signals': signals,
                    'indicator_values': indicator_values,
                    'score': score
                })

                print(f"Zakończono analizę dla {ticker}")

            except Exception as e:
                print(f"Błąd podczas analizy {ticker}: {str(e)}")
                continue

        # Wygeneruj raport
        if results:
            report = HtmlReportGenerator.generate_html_report(results)
            current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(f'stock_analysis_report_{current_date}.html', 'w', encoding='utf-8') as f:
                f.write(report)
            print("\nRaport wygenerowany: stock_analysis_report.html")
        else:
            print("\nBrak danych do wygenerowania raportu")
