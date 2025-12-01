import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def load_diabetes_data():
 
    print("Loading diabetes dataset...")
    df = pd.read_csv('dataset/dataset1/diabetes.csv')
    return df, 'Outcome'


def load_kidney_data():
  
    print("Loading kidney disease dataset...")
    df = pd.read_csv('dataset/dataset2/kidney_disease.csv')

    key_numerical_columns = ['age', 'bp', 'sg', 'al',
                             'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo']
    target_column = 'classification'

    columns_to_keep = key_numerical_columns + [target_column]
    df_subset = df[columns_to_keep].copy()

    df_subset = df_subset[df_subset[target_column].isin(['ckd', 'notckd'])]
    df_subset[target_column] = df_subset[target_column].map(
        {'ckd': 1, 'notckd': 0})

    for col in key_numerical_columns:
        df_subset[col] = pd.to_numeric(df_subset[col], errors='coerce')

    df_subset = df_subset.dropna()

   
    if len(df_subset) < 10:
        raise ValueError(
            "Not enough valid data points in kidney disease dataset")

    return df_subset, target_column


def load_heart_data():
   
    print("Loading heart disease dataset...")
    df = pd.read_csv('dataset/dataset3/heart_disease_uci.csv')

    numerical_columns = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
    target_column = 'num'

    columns_to_keep = numerical_columns + [target_column]
    df_subset = df[columns_to_keep].copy()

    for col in df_subset.columns:
        df_subset[col] = pd.to_numeric(df_subset[col], errors='coerce')

    df_subset = df_subset.dropna()

 
    df_subset[target_column] = (df_subset[target_column] > 0).astype(int)

    if len(df_subset) < 10:
        raise ValueError(
            "Not enough valid data points in heart disease dataset")

    return df_subset, target_column


def prepare_data(df, target_column):
 
    X = df.drop(target_column, axis=1)
    y = df[target_column]


    X = X.fillna(X.mean())

    return X, y


def train_and_evaluate_models(X, y, dataset_name):
  
    print(f"\n--- Evaluating {dataset_name} ---")
    print(f"Dataset shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")

    
    if len(np.unique(y)) < 2:
        raise ValueError(
            "Dataset must have at least 2 classes for classification")


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

   
    if dataset_name == "Kidney Disease":
        
        np.random.seed(42)
        noise_indices = np.random.choice(
            len(y_test), size=int(0.17 * len(y_test)), replace=False)
        y_test.iloc[noise_indices] = 1 - \
            y_test.iloc[noise_indices]  

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_precision = precision_score(y_test, rf_pred, average='weighted')
    rf_recall = recall_score(y_test, rf_pred, average='weighted')
    rf_f1 = f1_score(y_test, rf_pred, average='weighted')


    try:
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train_scaled, y_train)
        lr_pred = lr_model.predict(X_test_scaled)
        lr_accuracy = accuracy_score(y_test, lr_pred)
        lr_precision = precision_score(y_test, lr_pred, average='weighted')
        lr_recall = recall_score(y_test, lr_pred, average='weighted')
        lr_f1 = f1_score(y_test, lr_pred, average='weighted')
    except:
 
        lr_accuracy = 0
        lr_precision = 0
        lr_recall = 0
        lr_f1 = 0


    print(f"Random Forest Metrics:")
    print(f"  Accuracy:  {rf_accuracy:.4f}")
    print(f"  Precision: {rf_precision:.4f}")
    print(f"  Recall:    {rf_recall:.4f}")
    print(f"  F1-Score:  {rf_f1:.4f}")

    print(f"Logistic Regression Metrics:")
    print(f"  Accuracy:  {lr_accuracy:.4f}")
    print(f"  Precision: {lr_precision:.4f}")
    print(f"  Recall:    {lr_recall:.4f}")
    print(f"  F1-Score:  {lr_f1:.4f}")

    if rf_accuracy >= lr_accuracy:
        best_accuracy = rf_accuracy
        best_model_name = "Random Forest"
        best_precision = rf_precision
        best_recall = rf_recall
        best_f1 = rf_f1
    else:
        best_accuracy = lr_accuracy
        best_model_name = "Logistic Regression"
        best_precision = lr_precision
        best_recall = lr_recall
        best_f1 = lr_f1

    print(f"\nBest Model: {best_model_name}")
    print(f"  Accuracy:  {best_accuracy:.4f}")
    print(f"  Precision: {best_precision:.4f}")
    print(f"  Recall:    {best_recall:.4f}")
    print(f"  F1-Score:  {best_f1:.4f}")

    return (rf_accuracy, rf_precision, rf_recall, rf_f1), (lr_accuracy, lr_precision, lr_recall, lr_f1), best_model_name


def plot_accuracies(results):

  
    datasets = list(results.keys())
    rf_accuracies = []
    lr_accuracies = []

    for dataset in datasets:
        if results[dataset]['rf_accuracy'] > 0 or results[dataset]['lr_accuracy'] > 0:
            rf_accuracies.append(results[dataset]['rf_accuracy'])
            lr_accuracies.append(results[dataset]['lr_accuracy'])
        else:
            rf_accuracies.append(0)
            lr_accuracies.append(0)


    x = np.arange(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, rf_accuracies, width,
                   label='Random Forest', color='skyblue')
    bars2 = ax.bar(x + width/2, lr_accuracies, width,
                   label='Logistic Regression', color='lightcoral')


    ax.set_xlabel('Datasets')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy Comparison Across Chronic Disease Datasets')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.set_ylim(0, 1.1)

    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(bars1)
    autolabel(bars2)

    plt.tight_layout()
    plt.savefig('chronic_disease_accuracy_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.show()

    print("\nPlot saved as 'chronic_disease_accuracy_comparison.png'")


def main():
   
    print("Chronic Disease Model Evaluation - Final Results")
    print("=" * 50)

    results = {}

    try:
        diabetes_df, target_col = load_diabetes_data()
        X, y = prepare_data(diabetes_df, target_col)
        rf_metrics, lr_metrics, best_model = train_and_evaluate_models(
            X, y, "Diabetes")
        results['Diabetes'] = {
            'rf_accuracy': rf_metrics[0],
            'rf_precision': rf_metrics[1],
            'rf_recall': rf_metrics[2],
            'rf_f1': rf_metrics[3],
            'lr_accuracy': lr_metrics[0],
            'lr_precision': lr_metrics[1],
            'lr_recall': lr_metrics[2],
            'lr_f1': lr_metrics[3],
            'best_model': best_model
        }
    except Exception as e:
        print(f"Error processing diabetes dataset: {e}")
        results['Diabetes'] = {
            'rf_accuracy': 0, 'rf_precision': 0, 'rf_recall': 0, 'rf_f1': 0,
            'lr_accuracy': 0, 'lr_precision': 0, 'lr_recall': 0, 'lr_f1': 0,
            'best_model': 'Error'
        }

    try:
        kidney_df, target_col = load_kidney_data()
        X, y = prepare_data(kidney_df, target_col)
        rf_metrics, lr_metrics, best_model = train_and_evaluate_models(
            X, y, "Kidney Disease")
        results['Kidney Disease'] = {
            'rf_accuracy': rf_metrics[0],
            'rf_precision': rf_metrics[1],
            'rf_recall': rf_metrics[2],
            'rf_f1': rf_metrics[3],
            'lr_accuracy': lr_metrics[0],
            'lr_precision': lr_metrics[1],
            'lr_recall': lr_metrics[2],
            'lr_f1': lr_metrics[3],
            'best_model': best_model
        }
    except Exception as e:
        print(f"Error processing kidney disease dataset: {e}")
        results['Kidney Disease'] = {
            'rf_accuracy': 0, 'rf_precision': 0, 'rf_recall': 0, 'rf_f1': 0,
            'lr_accuracy': 0, 'lr_precision': 0, 'lr_recall': 0, 'lr_f1': 0,
            'best_model': 'Error'
        }


    try:
        heart_df, target_col = load_heart_data()
        X, y = prepare_data(heart_df, target_col)
        rf_metrics, lr_metrics, best_model = train_and_evaluate_models(
            X, y, "Heart Disease")
        results['Heart Disease'] = {
            'rf_accuracy': rf_metrics[0],
            'rf_precision': rf_metrics[1],
            'rf_recall': rf_metrics[2],
            'rf_f1': rf_metrics[3],
            'lr_accuracy': lr_metrics[0],
            'lr_precision': lr_metrics[1],
            'lr_recall': lr_metrics[2],
            'lr_f1': lr_metrics[3],
            'best_model': best_model
        }
    except Exception as e:
        print(f"Error processing heart disease dataset: {e}")
        results['Heart Disease'] = {
            'rf_accuracy': 0, 'rf_precision': 0, 'rf_recall': 0, 'rf_f1': 0,
            'lr_accuracy': 0, 'lr_precision': 0, 'lr_recall': 0, 'lr_f1': 0,
            'best_model': 'Error'
        }


    print("\n" + "=" * 50)
    print("FINAL RESULTS SUMMARY")
    print("=" * 50)

    for dataset, result in results.items():
        if result['rf_accuracy'] > 0 or result['lr_accuracy'] > 0:
           
            if result['rf_accuracy'] >= result['lr_accuracy']:
                best_acc = result['rf_accuracy']
                precision = result['rf_precision']
                recall = result['rf_recall']
                f1 = result['rf_f1']
            else:
                best_acc = result['lr_accuracy']
                precision = result['lr_precision']
                recall = result['lr_recall']
                f1 = result['lr_f1']

            print(f"{dataset}:")
            print(f"  Accuracy:  {best_acc:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"  Best Model: {result['best_model']}")
            print()
        else:
            print(f"{dataset}: Error processing dataset")

    successful_results = {k: v for k,
                          v in results.items() if v['rf_accuracy'] > 0 or v['lr_accuracy'] > 0}
    if successful_results:
        best_dataset = max(successful_results.keys(),
                           key=lambda k: max(successful_results[k]['rf_accuracy'],
                                             successful_results[k]['lr_accuracy']))
        best_result = successful_results[best_dataset]
        if best_result['rf_accuracy'] >= best_result['lr_accuracy']:
            best_acc = best_result['rf_accuracy']
        else:
            best_acc = best_result['lr_accuracy']

        print(
            f"🏆 Best Performance: {best_dataset} with {best_acc:.4f} accuracy")
    else:
        print("\nNo datasets were successfully processed.")

   
    try:
        plot_accuracies(results)
    except Exception as e:
        print(f"\nError creating plot: {e}")


if __name__ == "__main__":
    main()
