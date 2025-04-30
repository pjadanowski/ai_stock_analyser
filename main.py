from stock_analyser import StockAnalyzer

if __name__ == "__main__":
    print("=== Rozpoczęcie analizy giełdowej ===")
    analyzer = StockAnalyzer()
    analyzer.analyze()
    print("=== Analiza zakończona ===")
