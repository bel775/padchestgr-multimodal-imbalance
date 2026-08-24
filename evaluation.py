import torch
import numpy as np
import os
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

def calculate_metrics(all_labels, all_preds, all_probs):
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs, average="macro")
    except ValueError:
        auc = float("nan")

    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    auprc_per_class = np.asarray([
        average_precision_score(all_labels[:, idx], all_probs[:, idx])
        if np.any(all_labels[:, idx] == 1) else float("nan")
        for idx in range(all_labels.shape[1])
    ])
    macro_auprc = float(np.nanmean(auprc_per_class))

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": acc,
        "auc": auc,
        "f1_per_class": f1_per_class,
        "auprc_per_class": auprc_per_class,
        "macro_auprc": macro_auprc
    }

def print_final_metrics(metrics, class_names = None, fold = None):
    if fold is not None:
        print(f"\n=== Fold {fold + 1} Test Metrics ===")
    else:
        print("\n=== Final Metrics ===")
    print("F1-Score \t Precision \t Recall \t Accuracy \t AUC \t Macro-AUPRC")
    print(f"{metrics['f1']:.4f}\t{metrics['precision']:.4f}\t{metrics['recall']:.4f}\t{metrics['accuracy']:.4f}\t{metrics['auc']:.4f}\t{metrics['macro_auprc']:.4f}")

    if class_names is not None:
        class_metrics = pd.DataFrame({
            "Class": class_names,
            "F1-Score": metrics["f1_per_class"],
            "AUPRC": metrics["auprc_per_class"]
        })
        print("\n=== Per-Class Metrics ===")
        print(class_metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

def evaluate_sklearn_model(model, sentences, labels, class_names = None, fold = None):
    predictions = model.predict(sentences)
    decision_scores = model.decision_function(sentences)
    metrics = calculate_metrics(labels, predictions, decision_scores)
    print_final_metrics(metrics, class_names=class_names, fold=fold)
    return metrics

def save_experiment_result(metrics, save_path, experiment_config):
    """Save or update one non-cross-validation experiment in the results table."""
    result = {
        **experiment_config,
        "Macro-F1": metrics["f1"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "Accuracy": metrics["accuracy"],
        "ROC-AUC": metrics["auc"],
        "Macro-AUPRC": metrics["macro_auprc"]
    }
    result_df = pd.DataFrame([result])

    if os.path.exists(save_path):
        experiments_df = pd.read_csv(save_path)
        identity_columns = list(experiment_config.keys())

        # Older tables may not contain all of the current configuration columns.
        for column in result_df.columns:
            if column not in experiments_df.columns:
                experiments_df[column] = np.nan

        same_experiment = pd.Series(True, index=experiments_df.index)
        for column in identity_columns:
            same_experiment &= experiments_df[column].eq(experiment_config[column])

        experiments_df = experiments_df.loc[~same_experiment]
        experiments_df = pd.concat([experiments_df, result_df], ignore_index=True)
    else:
        experiments_df = result_df

    experiments_df.to_csv(save_path, index=False)

    print("\n=== Final-Cleaning Experiment Results ===")
    print(experiments_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

def save_cross_validation_results(metrics_by_fold, class_names, save_path, experiment_name):
    fold_rows = []
    class_rows = []

    for fold, metrics in enumerate(metrics_by_fold):
        fold_rows.append({
            "Experiment": experiment_name,
            "Fold": fold + 1,
            "Macro-F1": metrics["f1"],
            "Macro-AUPRC": metrics["macro_auprc"]
        })
        for class_idx, class_name in enumerate(class_names):
            class_rows.append({
                "Experiment": experiment_name,
                "Fold": fold + 1,
                "Class": class_name,
                "F1-Score": metrics["f1_per_class"][class_idx],
                "AUPRC": metrics["auprc_per_class"][class_idx]
            })

    fold_df = pd.DataFrame(fold_rows)
    class_fold_df = pd.DataFrame(class_rows)
    class_summary_df = (
        class_fold_df
        .groupby(["Experiment", "Class"], sort=False)
        .agg(
            F1_Mean=("F1-Score", "mean"),
            F1_Std=("F1-Score", "std"),
            AUPRC_Mean=("AUPRC", "mean"),
            AUPRC_Std=("AUPRC", "std")
        )
        .reset_index()
    )
    overall_average = {
        "Experiment": experiment_name,
        "Class": "Overall Average",
        "F1_Mean": class_summary_df["F1_Mean"].mean(),
        "F1_Std": class_summary_df["F1_Std"].mean(),
        "AUPRC_Mean": class_summary_df["AUPRC_Mean"].mean(),
        "AUPRC_Std": class_summary_df["AUPRC_Std"].mean()
    }
    class_summary_df = pd.concat(
        [class_summary_df, pd.DataFrame([overall_average])],
        ignore_index=True
    )

    fold_df.to_csv(f"{save_path}_folds.csv", index=False)
    class_fold_df.to_csv(f"{save_path}_per_class_folds.csv", index=False)
    class_summary_df.to_csv(f"{save_path}_per_class_summary.csv", index=False)

    print("\n=== Five-Fold Global Results ===")
    print(fold_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nMean Macro-F1: {fold_df['Macro-F1'].mean():.4f} +/- {fold_df['Macro-F1'].std():.4f}")
    print(f"Mean Macro-AUPRC: {fold_df['Macro-AUPRC'].mean():.4f} +/- {fold_df['Macro-AUPRC'].std():.4f}")
    print("\n=== Five-Fold Per-Class Mean Results ===")
    print(class_summary_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

def save_comparative_result(metrics, save_path, text_model, text_cleaning, label_count):
    result = {
        "Text Model": text_model,
        "Text Cleaning": text_cleaning,
        "Label Count": label_count,
        "F1-Score": metrics["f1"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "Accuracy": metrics["accuracy"],
        "AUC": metrics["auc"],
        "Macro-AUPRC": metrics["macro_auprc"]
    }
    result_df = pd.DataFrame([result])

    if os.path.exists(save_path):
        comparison_df = pd.read_csv(save_path)
        same_experiment = (
            (comparison_df["Text Model"] == text_model) &
            (comparison_df["Text Cleaning"] == text_cleaning) &
            (comparison_df["Label Count"] == label_count)
        )
        comparison_df = comparison_df[~same_experiment]
        comparison_df = pd.concat([comparison_df, result_df], ignore_index=True)
    else:
        comparison_df = result_df

    cleaning_order = {
        "Before_Cleaning": 0,
        "After_Label_Cleaning": 1,
        "After_Lexical_Filter": 2
    }
    comparison_df["_Cleaning Order"] = comparison_df["Text Cleaning"].map(cleaning_order)
    comparison_df = comparison_df.sort_values(["Text Model", "Label Count", "_Cleaning Order"])
    comparison_df = comparison_df.drop(columns=["_Cleaning Order"])
    comparison_df.to_csv(save_path, index=False)

    print("\n=== Text Leakage Comparative Table ===")
    print(comparison_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def evaluate_model(model, test_loader,training_mode, freezeText, eval_test = False, criterion = None, class_names = None, fold = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []
    running_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            if training_mode == 0:
                images = batch['image_feat'].to(device)
                labels = batch['label'].to(device)

                outputs = model(images)

            elif training_mode == 1:
                if freezeText:
                    sentences = batch['sentence'].to(device)
                else:
                    sentences = batch['sentence']

                labels = batch['label'].to(device)
                outputs = model(sentences)

            elif training_mode == 2:
                if freezeText:
                    sentences = batch['sentence'].to(device)
                else:
                    sentences = batch['sentence']

                images = batch['image_feat'].to(device)
                labels = batch['label'].to(device)
                outputs = model(images, sentences)
            else:
                raise ValueError(f"Unknown mode: {training_mode}")

            if criterion is not None:
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                #num_batches += 1

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    # Convert to NumPy arrays
    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)
    all_probs = np.vstack(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    if eval_test:
        metrics = calculate_metrics(all_labels, all_preds, all_probs)
        print_final_metrics(metrics, class_names=class_names, fold=fold)
        return metrics
    else:
        print(f"Val Accuracy:  {acc:.4f} - F1 Score:  {f1:.4f}")

        val_loss = running_loss / len(test_loader)
        return val_loss, f1
