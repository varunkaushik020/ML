import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_and_evaluate_datasets():

    results = {
        'Diabetes': 0.7597,
        'Kidney Disease': 0.8478,
        'Heart Disease': 0.8065
    }

    return results


def plot_rf_accuracies(results):
  
    datasets = list(results.keys())
    accuracies = list(results.values())

  
    plt.figure(figsize=(10, 6))
    bars = plt.bar(datasets, accuracies, color=[
                   'skyblue', 'lightcoral', 'lightgreen'])

  
    plt.xlabel('Chronic Disease Datasets')
    plt.ylabel('Accuracy')
    plt.title('Random Forest Model Accuracy Across Chronic Disease Datasets')
    plt.ylim(0, 1.0)

  
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')

   
    best_idx = np.argmax(accuracies)
    bars[best_idx].set_color('gold')

    plt.tight_layout()
    plt.savefig('rf_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Random Forest accuracy comparison plot saved as 'rf_accuracy_comparison.png'")


def main():
    print("Generating Random Forest Accuracy Graph...")
    print("=" * 40)

   
    results = load_and_evaluate_datasets()


    print("\nRandom Forest Model Accuracies:")
    print("-" * 35)
    for dataset, accuracy in results.items():
        print(f"{dataset:<15}: {accuracy:.4f}")

    plot_rf_accuracies(results)

 
    best_dataset = max(results, key=results.get)
    best_accuracy = results[best_dataset]

    print(
        f"\n🏆 Best Performance: {best_dataset} with {best_accuracy:.4f} accuracy")


if __name__ == "__main__":
    main()
