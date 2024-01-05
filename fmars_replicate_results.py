#!/usr/bin/env python3

from typing import Any, Tuple, Union, Dict, Callable, List, Iterable
import gc
import os
import sys
import pickle

import torch
import numpy as np
import matplotlib.pyplot as plt

import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

NN_COMPUTE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

adec_constructors_dict: Dict[str, Callable] = {
    "lof": lambda contamination : LocalOutlierFactor(novelty=True, contamination=contamination),
    "svm": lambda contamination : OneClassSVM(nu=contamination),
    "ifo": lambda contamination : IsolationForest(contamination=contamination),
    "cov": lambda contamination : EllipticEnvelope(contamination=contamination)
}

def compress_features(Xtr_features: np.ndarray, Xte_features: np.ndarray, ytr: np.ndarray, yte: np.ndarray, compression_param: int) -> Tuple[np.ndarray, ...]:    
    if compression_param == 0:
        return Xtr_features, ytr, Xte_features, yte
    
    pca: PCA = PCA(n_components=compression_param, random_state=1)
    pca = pca.fit(Xtr_features)
    
    Ztr_features = pca.transform(Xtr_features)
    Zte_features = pca.transform(Xte_features)

    return Ztr_features, ytr, Zte_features, yte

def fit_adecs(Ztr_f: np.ndarray, ytr_: np.ndarray, adec_name: str, adec_param: Any) -> Dict[int, Any]:
    num_classes: int = len(np.unique(ytr_))
    adecs_dict: Dict[int, Any] = {}

    for i in range(num_classes):
        adec = adec_constructors_dict[adec_name](adec_param)
        adecs_dict[i] = adec.fit(Ztr_f[np.where(ytr_ == i)[0]])

    return adecs_dict      

def compute_threshold(sensitivities, novel_rejects, thresholds):
    metrics_diff = np.abs(sensitivities - novel_rejects)
    best_thr_idx = np.argmin(metrics_diff)
    best_thr     = thresholds[best_thr_idx]
    
    return round(best_thr, 4), best_thr_idx

def find_best(sensitivities, novel_rejects, parameters):
    metrics_diff = np.abs(sensitivities - novel_rejects)
    best_param_idx = np.argmin(metrics_diff)
    best_param = parameters[best_param_idx]

    return round(best_param, 4), best_param_idx

def test_function_PCA_score(adecs_dict: Dict[str, Dict[int, OneClassSVM]], compressed_features_dict: Dict[int, tuple], tau=-0.25) -> Tuple[str, ...]:    
    _, _, Zte_features, yte_ = compressed_features_dict[DATASET]    
    num_classes = len(np.unique(yte_))

    classes_sizes = np.zeros(num_classes)
    response_matrix = np.zeros((num_classes + 2, num_classes))
    
    for cl in range(num_classes):    
        _, _, Zte_f, yte_ = compressed_features_dict[DATASET]
        inc_class_samples = Zte_f[np.where(yte_ == cl)[0]]
        classes_sizes[cl] = len(inc_class_samples)
        persample_system_preds = np.zeros((len(inc_class_samples), num_classes))
        persample_system_scores = np.zeros((len(inc_class_samples), num_classes))

        loo_system_scores = np.zeros((len(inc_class_samples), num_classes))
        
        for i, adec in adecs_dict[DATASET].items():
            single_adec_preds  = adec.predict(inc_class_samples)
            single_adec_scores = adec.decision_function(inc_class_samples)
            single_adec_preds[single_adec_preds < 0] = 0
            persample_system_preds[:, i]  = single_adec_preds.T 
            persample_system_scores[:, i] = single_adec_scores.T
            if i == cl:
                loo_system_scores[:, i] = np.zeros(len(inc_class_samples))
            else:
                loo_system_scores[:, i] = -np.sign(single_adec_scores.T - tau)

        mostconfident_adec = np.argmax(persample_system_scores, axis=1)
        for i in range(len(inc_class_samples)):
            persample_system_preds[i] *= 0
            if persample_system_scores[i, mostconfident_adec[i]] >= tau:
                persample_system_preds[i, mostconfident_adec[i]] = 1
        
        inc_true_positives = np.sum(persample_system_preds, axis=0) 
        inc_global_response = np.sum(persample_system_preds, axis=1)
        global_rejects = np.count_nonzero(inc_global_response == 0)
        loo_rejects = np.sum(loo_system_scores, axis=1)
        rejects_when_novel = np.count_nonzero(loo_rejects == num_classes - 1)
        
        response_matrix_row = np.zeros(num_classes + 2)
        response_matrix_row[0] = inc_true_positives[cl]
        response_matrix_row[-2]  = global_rejects
        response_matrix_row[-1]  = rejects_when_novel
        response_matrix[:, cl] = response_matrix_row
    
    response_matrix = response_matrix.T.astype(np.uint64)

    summary_matrix = np.zeros((num_classes, 3))
    
    summary_matrix[:, 0] = np.sum(response_matrix[:, :-2], axis=1) / classes_sizes
    summary_matrix[:, 1] = response_matrix[:, -2] / classes_sizes
    summary_matrix[:, 2] = response_matrix[:, -1] / classes_sizes

    avg_correct_predictions = np.mean(summary_matrix[:, 0])
    std_correct_predictions = np.std(summary_matrix[:, 0])
    avg_global_rejects      = np.mean(summary_matrix[:, 1])
    std_global_rejects      = np.std(summary_matrix[:, 1])
    avg_rej_when_novel      = np.mean(summary_matrix[:, 2])
    std_rej_when_novel      = np.std(summary_matrix[:, 2])

    return (
        avg_correct_predictions,
        avg_global_rejects,
        avg_rej_when_novel,
        std_correct_predictions,
        std_global_rejects,
        std_rej_when_novel        
    )

def _pytorch_load_subdir(subfolder_path: str, output_width: int, n_processes=4) -> Tuple[torch.Tensor, torch.Tensor, int]:
    is_imgfile = lambda filename : any(
        map(lambda ext : filename.lower().endswith(ext), 
            [".jpg", ".jpeg", ".png", ".tif"])
    )
    
    def _pytorch_build_common_transforms(output_width: int):
        from torchvision import transforms 
        resize_transform = transforms.Resize((output_width, output_width), transforms.InterpolationMode.BICUBIC)    
        
        transforms_list = [
            resize_transform,
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
        
        transform = transforms.Compose(transforms_list)
        print("Using the following Transforms: ")
        print(transform)

        return transform

    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    BATCH_SIZE = 256
    common_transforms = _pytorch_build_common_transforms(output_width)

    dataset = ImageFolder(subfolder_path, transform=common_transforms, is_valid_file=is_imgfile)
    loader = DataLoader(dataset, BATCH_SIZE, shuffle=False, num_workers=n_processes)
    
    X: torch.Tensor = torch.empty((len(dataset), 3, output_width, output_width))
    y: torch.Tensor = torch.empty((len(dataset), 1))

    for i, batch in enumerate(loader):
        X[i * BATCH_SIZE : (i+1) * BATCH_SIZE] = batch[0]
        y[i * BATCH_SIZE : (i+1) * BATCH_SIZE] = batch[1].reshape(-1, 1)

    return X, y, len(dataset.classes)        

def pytorch_load_dataset_from_directory(dataset_location: str, output_width: int, n_processes=4, test_size: float = 0.2):

    dataset_subfolders = os.listdir(dataset_location)

    train_subdir = None
    test_subdir  = None
    if "train" in dataset_subfolders:
        train_subdir = os.path.join(dataset_location, "train")
    if "test"  in dataset_subfolders:
        test_subdir  = os.path.join(dataset_location, "test")

    if train_subdir is None:
        raise RuntimeError(
            f"Expected at least a train directory!\n" + \
            f"Make sure that you have a directory named 'train' in {dataset_location}"
        )

    Xtr, ytr, tr_num_classes = _pytorch_load_subdir(train_subdir, output_width, n_processes)    


    if test_subdir is None:
        print(f"No test data available, using Hold-Out Cross-Validation with ratio {int(100 - test_size*100)}/{int(test_size*100)}")
        Xtr, Xte, ytr, yte = train_test_split(Xtr, ytr, test_size=test_size, random_state=1, stratify=ytr.flatten())

        return Xtr, ytr, Xte, yte, tr_num_classes

    Xte, yte, te_num_classes = _pytorch_load_subdir(test_subdir, output_width, n_processes)

    if tr_num_classes != te_num_classes:
        print(
            f"WARNING: Test Data contains f{te_num_classes} classes, " + \
            f"which is different from the number of classes in Train Data ({tr_num_classes})." + \
            f"\nThis can cause unexpected behaviours."
        )

    return Xtr, ytr, Xte, yte, te_num_classes

def load_pretrained_vitl16():
    import timm
    from timm.models.vision_transformer import VisionTransformer
    
    class VisionTransformerFex(torch.nn.Module):
        def __init__(self, base_model: VisionTransformer) -> None:
            super().__init__()
            self.base_model = base_model
            self.base_model.eval()

        def forward(self, x: torch.Tensor) -> torch.Tensor:        
            x = self.base_model.forward_features(x)
            return x[:, 0]
    
    
    model_key = "vit_large_patch16_224.augreg_in21k"    
    base_model: torch.nn.Module = timm.create_model(
        model_name=model_key,
        pretrained=True
    )

    feature_extractor = VisionTransformerFex(base_model)
    for param in feature_extractor.parameters():
        param.requires_grad = False
    feature_extractor.eval()
    
    feature_extractor.to(NN_COMPUTE_DEVICE)

    return feature_extractor

def extract_features(feature_extractor, Xtr, Xte, batch_size = 256, normalize = True, standardize = True):
    from torch.utils.data import DataLoader
    
    Xtr_features = np.zeros((len(Xtr), ) + (1024, ), dtype=np.float32)
    Xte_features = np.zeros((len(Xte), ) + (1024, ), dtype=np.float32)
    train_loader = DataLoader(Xtr, batch_size=batch_size, pin_memory=True)
    test_loader  = DataLoader(Xte, batch_size=batch_size, pin_memory=True)

    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            x_batch: torch.Tensor = feature_extractor(batch.float().to(NN_COMPUTE_DEVICE))
            Xtr_features[i * batch_size : (i+1)*batch_size] = x_batch.cpu().numpy()
        
        for i, batch in enumerate(test_loader):
            x_batch: torch.Tensor = feature_extractor(batch.float().to(NN_COMPUTE_DEVICE))
            Xte_features[i * batch_size : (i+1) * batch_size] = x_batch.cpu().numpy()

    if normalize:
        tr_min = np.min(Xtr_features, axis=0)
        tr_max = np.max(Xtr_features, axis=0)

        Xtr_features = (Xtr_features - tr_min) / (tr_max - tr_min)
        Xte_features = (Xte_features - tr_min) / (tr_max - tr_min)

    if standardize:
        tr_mean = np.mean(Xtr_features, axis=0)
        tr_std  = np.std(Xtr_features, axis=0) + 1e-6 # numerical stability

        Xtr_features = (Xtr_features - tr_mean) / tr_std
        Xte_features = (Xte_features - tr_mean) / tr_std

    return Xtr_features, Xte_features

def store_visual_features(Xtr, Xte, ytr, yte, dataset, fex_model_name, subdir):
    try:
        os.makedirs(f"./VisualFeatures/{dataset}/{subdir}", exist_ok=True)
    except:
        raise RuntimeError("Could not create base folders for storing visual features")
    
    try:
        np.save(f"./VisualFeatures/{dataset}/{subdir}/Xtr_{fex_model_name}.npy", Xtr)
        np.save(f"./VisualFeatures/{dataset}/{subdir}/Xte_{fex_model_name}.npy", Xte)
        np.save(f"./VisualFeatures/{dataset}/{subdir}/ytr_{fex_model_name}.npy", ytr)
        np.save(f"./VisualFeatures/{dataset}/{subdir}/yte_{fex_model_name}.npy", yte)
        
        print(f"Successfully stored {fex_model_name} features from {dataset} in directory './VisualFeatures/{dataset}/{subdir}'")
    
    except:
        raise RuntimeError(f"Unable to store {fex_model_name} features from {dataset} in directory './VisualFeatures/{dataset}/{subdir}'")

def load_visual_features(dataset, fex_model_name, subdir):
    try:
        Xtr = np.load(f"./VisualFeatures/{dataset}/{subdir}/Xtr_{fex_model_name}.npy")
        Xte = np.load(f"./VisualFeatures/{dataset}/{subdir}/Xte_{fex_model_name}.npy")
        ytr = np.load(f"./VisualFeatures/{dataset}/{subdir}/ytr_{fex_model_name}.npy")
        yte = np.load(f"./VisualFeatures/{dataset}/{subdir}/yte_{fex_model_name}.npy")
        num_classes = len(np.unique(ytr))
    
    except:
        print("No saved feature available from given arguments")
        raise RuntimeError(f"Unable to load {fex_model_name} features from {dataset} in directory './VisualFeatures/{dataset}/{subdir}'")
    
    return Xtr, Xte, ytr, yte

def testing_function(dataset, fex_model_name):
    CONTAMINATION = 0.075
    COMP_DIM      = 50    
    
    thresholds = np.arange(-1.0, 1.05, 0.05)  

    print("Loading Deep Visual Features...")
    
    Xtr_features, Xte_features, ytr, yte = load_visual_features(dataset, fex_model_name, subdir="plain_deep_features")
    Xtr_features, Xval_features, ytr, yval = train_test_split(Xtr_features, ytr, test_size=0.20, shuffle=True, stratify=ytr.flatten(), random_state=1024)

    compressed_features_dict = {}
    compressed_features_dict[dataset] = compress_features(Xtr_features, Xval_features, ytr, yval, compression_param=COMP_DIM)
    Ztr_f, ytr_, _, _ = compressed_features_dict[dataset]
    adecs_dict = {}
    adecs_dict[dataset] = fit_adecs(Ztr_f, ytr_, adec_name="svm", adec_param=CONTAMINATION)

    avg_sens = []
    avg_grej = []
    avg_nrej = []
    std_sens = []
    std_grej = []
    std_nrej = []
    
    print("Automatic Threshold Tuning on the hold-out Validation Set...")

    for threshold in thresholds:                
        sens, grej, nrej, sens_s, grej_s, nrej_s = test_function_PCA_score(adecs_dict, compressed_features_dict, tau=threshold)
        avg_sens.append(sens)
        avg_grej.append(grej)
        avg_nrej.append(nrej)
        std_sens.append(sens_s)
        std_grej.append(grej_s)
        std_nrej.append(nrej_s)
        print(f"Threshold: {threshold:.2f}, Sensitivity: {sens:.3f}%, FNR: {grej:.3f}%, Specificity:{nrej:.3f}%")

    avg_sens = np.array(avg_sens)
    avg_nrej = np.array(avg_nrej)

    best_thr, best_idx = compute_threshold(avg_sens, avg_nrej, thresholds)
    resulting_sens   = avg_sens[best_idx]
    resulting_sens_s = std_sens[best_idx]
    resulting_grej   = avg_grej[best_idx]
    resulting_grej_s = std_grej[best_idx]
    resulting_nrej   = avg_nrej[best_idx]
    resulting_nrej_s = std_nrej[best_idx]

    print(f"Best Threshold: {best_thr:.2f}")
    print(f"\t -- Validation Sensitivity: {resulting_sens:.3f} +/- {resulting_sens_s:.3f}")
    print(f"\t -- Validation FNR:         {resulting_grej:.3f} +/- {resulting_grej_s:.3f}")
    print(f"\t -- Validation Specificity; {resulting_nrej:.3f} +/- {resulting_nrej_s:.3f}")

    print("Evaluating Performance on the Test Set")
    
    compressed_features_dict = {}
    compressed_features_dict[dataset] = compress_features(Xtr_features, Xte_features, ytr, yte, compression_param=COMP_DIM)
    Ztr_f, ytr_, _, _ = compressed_features_dict[dataset]
    adecs_dict = {}
    adecs_dict[dataset] = fit_adecs(Ztr_f, ytr_, adec_name="svm", adec_param=CONTAMINATION)
    sens, grej, nrej, sens_s, grej_s, nrej_s = test_function_PCA_score(adecs_dict, compressed_features_dict, tau=best_thr)

    print(f"Test Set Evaluation")
    print(f"\t -- Test Sensitivity: {sens:.3f} +/- {sens_s:.3f}")
    print(f"\t -- Test FNR:         {grej:.3f} +/- {grej_s:.3f}")
    print(f"\t -- Test Specificity; {nrej:.3f} +/- {nrej_s:.3f}")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="Dataset name", type=str)
if __name__ == "__main__":
    cmd_args = parser.parse_args()
    # Config and constants
    FEX_MODEL = "ViTL-16_22K"
    IMAGE_WIDTH = 224
    COMP_DIM = 50
    BATCH_SIZE = 256
    DATASET = cmd_args.dataset
    datasets_paths = {
        "WHOI22": "./data/WHOI22",
        "WHOI40": "./data/WHOI40"
    }

    features_dict: dict[int, tuple] = {}

    feature_extractor = load_pretrained_vitl16()
    Xtr, ytr, Xte, yte, num_classes = pytorch_load_dataset_from_directory(datasets_paths[DATASET], output_width=IMAGE_WIDTH)
    Xtr_features, Xte_features = extract_features(feature_extractor, Xtr, Xte, BATCH_SIZE, normalize=True, standardize=True)
    
    store_visual_features(Xtr_features, Xte_features, ytr, yte, dataset=DATASET, fex_model_name=FEX_MODEL, subdir="plain_deep_features")

    del Xtr
    del Xte
    del ytr
    del yte
    del Xtr_features
    del Xte_features
    del feature_extractor
    
    torch.cuda.empty_cache()
    gc.collect()

    testing_function(DATASET, FEX_MODEL)