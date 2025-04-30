import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import time

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

    def calculate_intrinsic_value(self, ticker: str) -> Optional[float]:
        """Oblicza wartość wewnętrzną z zabezpieczeniami"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            methods = []

            # Free Cash Flow method
            if all(k in info for k in ['freeCashflow', 'sharesOutstanding']):
                fcf = info['freeCashflow']
                shares = info['sharesOutstanding']
                if fcf and shares and shares > 0:
                    methods.append((fcf / shares) * 15)

            # Earnings method
            if 'trailingEps' in info and info['trailingEps']:
                methods.append(info['trailingEps'] * 18)

            # Book Value method
            if 'bookValue' in info and info['bookValue']:
                methods.append(info['bookValue'] * 3)

            return np.mean(methods) if methods else None

        except Exception as e:
            print(f"Błąd podczas obliczania wartości wewnętrznej dla {ticker}: {str(e)}")
            return None

    def generate_html_report(self, results: List[Dict]) -> str:
        """Generuje raport HTML z wynikami"""
        # Sortuj wyniki według score
        sorted_results = sorted(
            [r for r in results if r is not None and isinstance(r, dict)],
            key=lambda x: x.get('score', 0),
            reverse=True
        )

        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Stock Analysis Report</title>
            <script src="https://cdn.tailwindcss.com"></script>
            <style>
                .positive {{ color: #10B981; font-weight: bold; }}
                .negative {{ color: #EF4444; font-weight: bold; }}
                .neutral {{ color: #6B7280; }}
                .score-bar {{
                    background: #E5E7EB;
                    height: 8px;
                    border-radius: 4px;
                    overflow: hidden;
                }}
                .score-fill {{
                    height: 100%;
                    background: #10B981;
                    width: var(--width);
                }}
            </style>
        </head>
        <body class="bg-gray-50 p-8">
            <div class="max-w-7xl mx-auto">
                <h1 class="text-3xl font-bold text-gray-800 mb-6 text-center">📈 Stock Analysis Report</h1>

                <div class="bg-white shadow-md rounded-lg overflow-hidden mb-8">
                    <table class="min-w-full divide-y divide-gray-200">
                        <thead class="bg-gray-100">
                            <tr>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider">Ticker</th>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider">Price</th>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider">Intrinsic</th>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider">Valuation</th>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider">Score</th>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider">Signal</th>
                                <th class="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider">Indicators</th>
                            </tr>
                        </thead>
                        <tbody class="bg-white divide-y divide-gray-200">
        """

        for result in sorted_results:
            try:
                # Formatowanie wartości
                price = result.get('price')
                price_str = f"${price:.2f}" if price is not None and not np.isnan(price) else "N/A"

                intrinsic = result.get('intrinsic')
                intrinsic_str = f"${intrinsic:.2f}" if intrinsic is not None and not np.isnan(intrinsic) else "N/A"

                # Wycena
                if price is not None and intrinsic is not None and not np.isnan(price) and not np.isnan(intrinsic):
                    ratio = price / intrinsic
                    if ratio < 0.9:
                        valuation = "Undervalued"
                        val_class = "positive"
                    elif ratio > 1.1:
                        valuation = "Overvalued"
                        val_class = "negative"
                    else:
                        valuation = "Fair"
                        val_class = "neutral"
                else:
                    valuation = "N/A"
                    val_class = "neutral"

                # Score i sygnał
                score = result.get('score', 0)
                score = max(0, min(100, score))  # Ogranicz score do 0-100
                signal = "✅" if score >= 60 else "❌"
                signal_class = "positive" if score >= 60 else "negative"

                # Wskaźniki
                indicators_html = "<div class='text-xs space-y-1'>"
                signals = result.get('signals', {})
                indicator_values = result.get('indicator_values', {})

                for name in signals.keys():
                    value = signals.get(name)
                    val = indicator_values.get(name, "N/A")

                    if isinstance(val, (int, float)):
                        if np.isnan(val):
                            val = "N/A"
                        else:
                            val = f"{val:.2f}"
                    else:
                        val = str(val)

                    if value is None:
                        icon = "∅"
                        cls = "neutral"
                    elif value:
                        icon = "✓"
                        cls = "positive"
                    else:
                        icon = "✗"
                        cls = "negative"

                    indicators_html += f"""
                    <div>
                        <span class="font-medium">{name}:</span>
                        <span class="{cls}">{val} {icon}</span>
                    </div>
                    """

                indicators_html += "</div>"

                # Dodaj wiersz do tabeli
                html += f"""
                            <tr>
                                <td class="px-6 py-4 whitespace-nowrap font-medium">{result.get('ticker', 'N/A')}</td>
                                <td class="px-6 py-4 whitespace-nowrap">{price_str}</td>
                                <td class="px-6 py-4 whitespace-nowrap">{intrinsic_str}</td>
                                <td class="px-6 py-4 whitespace-nowrap {val_class}">{valuation}</td>
                                <td class="px-6 py-4 whitespace-nowrap">
                                    <div class="flex items-center gap-2">
                                        <span>{score:.0f}%</span>
                                        <div class="score-bar"><div class="score-fill" style="--width: {score}%"></div></div>
                                    </div>
                                </td>
                                <td class="px-6 py-4 whitespace-nowrap font-bold {signal_class}">{signal}</td>
                                <td class="px-6 py-4">{indicators_html}</td>
                            </tr>
                """
            except Exception as e:
                print(f"Błąd podczas generowania wiersza raportu: {str(e)}")
                continue

        html += """
                        </tbody>
                    </table>
                </div>

                <div class="bg-white shadow-md rounded-lg p-6 mb-8">
                    <h2 class="text-xl font-semibold text-gray-800 mb-4">Indicator Legend</h2>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                            <h3 class="font-medium text-gray-700 mb-2">Moving Averages</h3>
                            <ul class="list-disc pl-5 space-y-1 text-sm text-gray-600">
                                <li><span class="font-medium">SMA 20 > SMA 50</span> - Short-term bullish</li>
                                <li><span class="font-medium">SMA 50 > SMA 200</span> - Long-term bullish</li>
                            </ul>
                        </div>
                        <div>
                            <h3 class="font-medium text-gray-700 mb-2">RSI</h3>
                            <ul class="list-disc pl-5 space-y-1 text-sm text-gray-600">
                                <li>Optimal range: <span class="font-medium">30-70</span></li>
                                <li><span class="positive">Below 30</span> - Oversold</li>
                                <li><span class="negative">Above 70</span> - Overbought</li>
                            </ul>
                        </div>
                    </div>
                </div>

                <div class="text-sm text-gray-500 text-center">
                    <p>Report generated: {datetime}</p>
                </div>
            </div>
        </body>
        </html>
        """.format(datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        return html

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
            report = self.generate_html_report(results)
            current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(f'stock_analysis_report_{current_date}.html', 'w', encoding='utf-8') as f:
                f.write(report)
            print("\nRaport wygenerowany: stock_analysis_report.html")
        else:
            print("\nBrak danych do wygenerowania raportu")


if __name__ == "__main__":
    print("=== Rozpoczęcie analizy giełdowej ===")
    analyzer = StockAnalyzer()
    analyzer.analyze()
    print("=== Analiza zakończona ===")