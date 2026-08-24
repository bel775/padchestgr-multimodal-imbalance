import os
from data.data_loader import get_data
from train import train_model
from evaluation import evaluate_model, evaluate_sklearn_model, save_comparative_result, save_cross_validation_results, save_experiment_result
import pandas as pd
import argparse
import numpy as np
import gc
import torch
import joblib
from models.textModels import TFIDF_LinearSVM
from utils import get_model, get_criterion, preProcessData, get_multilabel_split_indices, print_sklearn_splits, set_seed
dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def main(wrs_mode, classWeighted, dataAug, oversampling, imagemodel, raddinoHead, textmodel, freezeImage,freezeText, label_count, textCleaning, crossValidation, saveModel=False, seed=42):
    set_seed(seed)
    print(f"Using random seed: {seed}")
    print("Start the Main ...")

    data = pd.read_csv(os.path.join(dir,"padchestgr-multimodal-imbalance/master_table.csv"))

    images_src = os.path.join(dir,'padchestgr-multimodal-imbalance/images/PadChest_GR')
    RadDino_src = os.path.join(dir,'dinov2_src')
    RadDinoWeights = os.path.join(dir,'padchestgr-multimodal-imbalance/backbones/backbone_compatible.safetensors')
    #RadDinoWeights = os.path.join(dir,'models/radDinoMaria2.safetensors')

    Head_RadDinoWeights = os.path.join(dir,'padchestgr-multimodal-imbalance/backbones/dino_head.safetensors')

    supects_terms_path = os.path.join(dir,"padchestgr-multimodal-imbalance/terminos_sospechosos_coincidencia_lexica.xlsx")
    #model = radDino_Head(768,Head_RadDinoWeights, 25).to('cuda')
    #print(model)
    #model = None

    if textmodel == 4 and imagemodel == 3:
        raise ValueError("Invalid configuration: both text and image training are disabled. Select at least one model.")
    if textmodel == 5 and imagemodel != 3:
        raise ValueError("TF-IDF + Linear SVM is a text-only baseline. Select --imagemodel 3.")
    if textmodel == 5 and (wrs_mode or classWeighted or oversampling or freezeText):
        raise ValueError("TF-IDF + Linear SVM does not use --wrs, --cw, --os or --freezeText.")


    wrs_str = "_WRS" if wrs_mode else ""
    cw_str = "_CW" if classWeighted else ""
    dataAug_str = "_DataAug" if dataAug else ""
    os_str = "_OS" if oversampling else ""
    ftImage_str = "Freeze" if freezeImage else ""
    ftText_str = "Freeze" if freezeText else ""

    textmodel_str = ""
    if textmodel == 0: textmodel_str = "_BertTokenizer"
    elif textmodel == 1: textmodel_str = "_BioBert"
    elif textmodel == 2: textmodel_str = "_CXR-Bert"
    elif textmodel == 3: textmodel_str = "_CXRBert_Especialized"
    elif textmodel == 5: textmodel_str = "_TF-IDF_LinearSVM"
    else: textmodel_str = ""

    comparison_model_str = f"{textmodel_str.strip('_')}{ftText_str}{wrs_str}{cw_str}{os_str}"

    text_cleaning_names = {
        0: "Before_Cleaning",
        1: "After_Label_Cleaning",
        2: "After_Lexical_Filter"
    }
    text_cleaning_str = f"_{text_cleaning_names[textCleaning]}"

    training_mode = 0 # 0 = Image-Only, 1 = Text-Only, 2 = MultiModal
    imagemodel_str = ""
    if imagemodel == 0: imagemodel_str = "_ResNet50"
    elif imagemodel == 1: imagemodel_str = "_RadDino_MAIRA1"
    elif imagemodel == 2: imagemodel_str = "_RadDino_MAIRA2"
    else: imagemodel_str = ""

    if textmodel == 4:
        training_mode = 0
        print("Image unimodal of type:", imagemodel_str)
    elif imagemodel == 3:
        training_mode = 1
        print("Text unimodal of type:", textmodel_str)
    else:
        training_mode = 2
        print("Multimodal of types:", textmodel_str, "and", imagemodel_str)

    batch_size = 32
    dataset, mlb = preProcessData(data, label_count = label_count, xlsx_path = supects_terms_path, textCleaning = textCleaning)

    comparison_path = os.path.join(dir,'padchestgr-multimodal-imbalance/graphs/text_leakage_comparison.csv')
    experiment_results_path = os.path.join(dir,'padchestgr-multimodal-imbalance/graphs/experiment_results.csv')
    experiment_name = f"{imagemodel_str.strip('_')}{ftImage_str}{comparison_model_str}{text_cleaning_str}{dataAug_str}_Labels{label_count}"
    experiment_config = {
        "Experiment": experiment_name,
        "Text Model": textmodel_str.strip('_') or "None",
        "Image Model": imagemodel_str.strip('_') or "None",
        "Class Weighted": classWeighted,
        "WRS": wrs_mode,
        "Data Augmentation": dataAug,
        "Oversampling": oversampling,
        "Freeze Image": freezeImage,
        "Freeze Text": freezeText,
        "RadDINO Head": raddinoHead,
        "Label Count": label_count,
        "Text Cleaning": text_cleaning_names[textCleaning],
        "Cross Validation": crossValidation,
        "Seed": seed
    }
    cross_validation_path = os.path.join(dir,f'padchestgr-multimodal-imbalance/graphs/cross_validation_{experiment_name}')
    save_models_path = os.path.join(dir,'padchestgr-multimodal-imbalance/save_models')
    if saveModel:
        os.makedirs(save_models_path, exist_ok=True)
    folds = range(5) if crossValidation else range(1)
    metrics_by_fold = []

    if textmodel == 5:
        all_labels = np.asarray(dataset['multi_hot'].to_list())
        for fold in folds:
            if crossValidation:
                print(f"\n================ CROSS-VALIDATION FOLD {fold + 1}/5 ================")
            train_idx, val_idx, test_idx = get_multilabel_split_indices(
                all_labels,
                test_fold=fold,
                crossValidation=crossValidation,
                seed=seed
            )
            print_sklearn_splits(all_labels, train_idx, val_idx, test_idx, mlb.classes_)

            model = TFIDF_LinearSVM(seed=seed)
            model.fit(
                dataset.iloc[train_idx]['final_sentence'].tolist(),
                all_labels[train_idx]
            )
            metrics = evaluate_sklearn_model(
                model,
                dataset.iloc[test_idx]['final_sentence'].tolist(),
                all_labels[test_idx],
                class_names=mlb.classes_,
                fold=fold if crossValidation else None
            )
            metrics_by_fold.append(metrics)

            if saveModel:
                fold_str = f"_Fold{fold + 1}" if crossValidation else ""
                model_path = os.path.join(save_models_path,f'{experiment_name}{fold_str}.joblib')
                joblib.dump({
                    "model": model,
                    "classes": mlb.classes_,
                    "configuration": experiment_config,
                    "fold": fold + 1 if crossValidation else None
                }, model_path)
                print("Model saved in:", model_path)

        if crossValidation:
            save_cross_validation_results(
                metrics_by_fold,
                mlb.classes_,
                cross_validation_path,
                experiment_name
            )
        else:
            save_comparative_result(
                metrics_by_fold[0],
                comparison_path,
                comparison_model_str,
                text_cleaning_names[textCleaning],
                label_count
            )
            if textCleaning == 2:
                save_experiment_result(
                    metrics_by_fold[0],
                    experiment_results_path,
                    experiment_config
                )
        return

    for fold in folds:
        if crossValidation:
            print(f"\n================ CROSS-VALIDATION FOLD {fold + 1}/5 ================")

        train_loader, val_loader, test_loader, pos_weight, feat_dim = get_data(dataset, mlb, training_mode, images_src,RadDino_src,RadDinoWeights, IMAGE_SIZE = 518,
                                                                                          wrs_mode= wrs_mode, imagemodel = imagemodel, textmodel = textmodel,
                                                                                          DataAug = dataAug, oversampler = oversampling,
                                                                                          freezeImage = freezeImage, freezeText = freezeText,
                                                                                          test_fold = fold, crossValidation = crossValidation,
                                                                                          seed = seed)

        fold_str = f"_Fold{fold + 1}" if crossValidation else ""
        save_loss_path = os.path.join(dir,f'padchestgr-multimodal-imbalance/graphs/loss_curve{imagemodel_str}{ftImage_str}{textmodel_str}{ftText_str}{text_cleaning_str}{wrs_str}{cw_str}{dataAug_str}{os_str}_Labels{label_count}{fold_str}.png')

        model,optimizer = get_model(textmodel,imagemodel,freezeImage,freezeText,RadDino_src,
                                    RadDinoWeights,Head_RadDinoWeights,num_classes=label_count,
                                    fusion_dim = feat_dim, raddinoHead = raddinoHead)
        criterion = get_criterion(classWeighted, pos_weight)
        model = train_model(model,optimizer,criterion, training_mode, train_loader, val_loader,freezeText, save_loss_path)

        metrics = evaluate_model(
            model,
            test_loader,
            training_mode,
            freezeText,
            eval_test=True,
            class_names=mlb.classes_,
            fold=fold if crossValidation else None
        )
        metrics_by_fold.append(metrics)

        if saveModel:
            model_path = os.path.join(save_models_path,f'{experiment_name}{fold_str}.pth')
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": mlb.classes_.tolist(),
                "configuration": experiment_config,
                "fold": fold + 1 if crossValidation else None
            }, model_path)
            print("Model saved in:", model_path)

        # Release the current fold before extracting and caching the next fold.
        # RadDINO produces a large [tokens, embedding_dim] tensor per image, so
        # retaining two folds at once can exceed the SLURM system-memory limit.
        del model, optimizer, criterion
        del train_loader, val_loader, test_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if crossValidation:
        save_cross_validation_results(
            metrics_by_fold,
            mlb.classes_,
            cross_validation_path,
            experiment_name
        )
    elif training_mode == 1:
        save_comparative_result(
            metrics_by_fold[0],
            comparison_path,
            comparison_model_str,
            text_cleaning_names[textCleaning],
            label_count
        )

    if not crossValidation and textCleaning == 2 and textmodel != 5:
        save_experiment_result(
            metrics_by_fold[0],
            experiment_results_path,
            experiment_config
        )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run main training script with configurable options.")

    ## Balance Strategy
    parser.add_argument("--wrs", action="store_true", help="Enable Weighted Random Sampling mode (default: False)")
    parser.add_argument("--cw", action="store_true", help="Enable Class Weighted mode (default: False)")
    parser.add_argument("--dataAug", action="store_true", help="Enable Data Augmentation mode (default: False)")
    parser.add_argument("--os", action="store_true", help="Enable OverSampling mode (default: False)")

    ## Model Configuration
    parser.add_argument("--imagemodel", type=int, choices=[0, 1, 2, 3], default=3,
                        help="Select RadDino type: 0=ResNet50, 1=MAIRA-1, 2=MAIRA-2, 3=WithoutImage (default: 3)")
    parser.add_argument("--raddinohead", action="store_true", help="Enable PreTrained Head mode (default: False)")
    parser.add_argument("--textmodel", type=int, choices=[0, 1, 2, 3, 4, 5], default=4,
                        help="Select model type: 0=BertTokenizer, 1=BioBert, 2=CXR-Bert, 3=CXRBert-Especial, 4=WithoutText, 5=TF-IDF+LinearSVM (default: 4)")
    parser.add_argument("--freezeImage", action="store_true", help="Enable Image Freeze RadDino Backbone mode (default: False)")
    parser.add_argument("--freezeText", action="store_true", help="Enable Text Freeze Backbone mode (default: False)")

    ## Aux modes
    parser.add_argument("--label_count", type=int, choices=[25, 20, 15, 10, 5, 3, 2], default=25,
                        help="Select Label counts: 25 label, 20 label, 15 label, 10 label, 5 label, (default: 25)")
    parser.add_argument("--textCleaning", type=int, choices=[0, 1, 2], default=2,
                        help="Select text cleaning stage: 0=Before cleaning, 1=After removing label names, 2=After final lexical filter (default: 2)")
    parser.add_argument("--crossValidation", action="store_true",
                        help="Enable five-fold multilabel-stratified cross-validation (default: False)")
    parser.add_argument("--saveModel", action="store_true",
                        help="Save the trained model checkpoint (default: False)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed used for reproducible runs (default: 42)")
    args = parser.parse_args()

    main(args.wrs, args.cw,args.dataAug,args.os, args.imagemodel,args.raddinohead,args.textmodel,args.freezeImage,args.freezeText,args.label_count,args.textCleaning,args.crossValidation,args.saveModel,args.seed)
