from datetime import datetime
from typing import Dict, List, Any

import numpy as np


class HtmlReportGenerator:
    """Klasa odpowiedzialna za generowanie raportów HTML"""

    @staticmethod
    def create_html_header() -> str:
        """Generuje nagłówek HTML z stylami"""
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Stock Analysis Report</title>
            <script src="https://cdn.tailwindcss.com"></script>
            <style>
                .positive { color: #10B981; font-weight: bold; }
                .negative { color: #EF4444; font-weight: bold; }
                .neutral { color: #6B7280; }
                .score-bar {
                    background: #E5E7EB;
                    height: 8px;
                    border-radius: 4px;
                    overflow: hidden;
                }
                .score-fill {
                    height: 100%;
                    background: #10B981;
                    width: var(--width);
                }
            </style>
        </head>
        <body class="bg-gray-50 p-8">
            <div class="max-w-7xl mx-auto">
                <h1 class="text-3xl font-bold text-gray-800 mb-6 text-center">📈 Stock Analysis Report</h1>
        """

    @staticmethod
    def create_table_header() -> str:
        """Generuje nagłówek tabeli wyników"""
        return """
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

    @staticmethod
    def create_table_row(result: Dict[str, Any]) -> str:
        """Generuje pojedynczy wiersz tabeli z wynikami dla akcji"""
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

            # Wiersz tabeli
            return f"""
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
            return ""

    @staticmethod
    def create_legend_and_footer() -> str:
        """Generuje legendę i stopkę raportu"""
        return """
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

    @classmethod
    def generate_html_report(cls, results: List[Dict]) -> str:
        """Generuje kompletny raport HTML z wynikami"""
        # Sortuj wyniki według score
        sorted_results = sorted(
            [r for r in results if r is not None and isinstance(r, dict)],
            key=lambda x: x.get('score', 0),
            reverse=True
        )

        html = cls.create_html_header()
        html += cls.create_table_header()

        for result in sorted_results:
            html += cls.create_table_row(result)

        html += cls.create_legend_and_footer()
        return html
