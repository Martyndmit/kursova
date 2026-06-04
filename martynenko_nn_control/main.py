from pathlib import Path
from experiments import run_all
from visualization import build_all_charts

def main():
    results_dir = Path('results')
    X_train, X_test, y_train, y_test, models, history_df, final_df = run_all(results_dir)
    build_all_charts(X_train, X_test, y_train, y_test, models, history_df, final_df, results_dir)
    print('Neural network control experiment completed.')
    print('Summary:', results_dir / 'summary_results.csv')
    print(final_df.to_string(index=False))

if __name__ == '__main__':
    main()
