"""Clasificación anatómica 3D de fragmentos PENGWIN.

La cabeza recibe las características del backbone correspondientes
a cada instancia segmentada y clasifica cada fragmento como:

    0 = sacro
    1 = coxal izquierdo
    2 = coxal derecho

Durante entrenamiento se utilizan las instancias del ground truth.
Durante inferencia se utilizan las instancias producidas por la
cabeza de segmentación.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


CLASS_NAMES = (
    "sacrum",
    "left_hipbone",
    "right_hipbone",
)


class ClassificationHead3D(nn.Module):
    """Clasifica el vector de características de una instancia."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_classes: int = 3,
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(
                in_channels,
                hidden_channels,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(
                hidden_channels,
                num_classes,
            ),
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        return self.classifier(features)


def fragment_code_to_class(
    code: int,
) -> int:
    """Código PENGWIN -> categoría anatómica."""

    if 1 <= code <= 10:
        return 0

    if 11 <= code <= 20:
        return 1

    if 21 <= code <= 30:
        return 2

    raise ValueError(
        f"Código de fragmento inválido: {code}"
    )


def _prepare_mask(
    mask: torch.Tensor,
    spatial_size,
) -> torch.Tensor:
    """Lleva una máscara de instancias al tamaño del feature map."""

    if mask.ndim == 3:
        mask = mask.unsqueeze(0)

    mask = F.interpolate(
        mask.unsqueeze(1).float(),
        size=spatial_size,
        mode="nearest",
    )

    return mask[:, 0].long()


def _masked_average(
    feature_map: torch.Tensor,
    instance_mask: torch.Tensor,
) -> torch.Tensor:
    """Promedio de las características dentro de una instancia."""

    weights = instance_mask.float().unsqueeze(0)

    denominator = weights.sum().clamp(min=1.0)

    vector = (
        feature_map * weights
    ).sum(dim=(1, 2, 3)) / denominator

    return vector


def extract_gt_instance_features(
    feature_map: torch.Tensor,
    fragment_mask: torch.Tensor,
):
    """Extrae un vector por fragmento usando las máscaras reales.

    feature_map:
        [B, C, D, H, W]

    fragment_mask:
        [B, D0, H0, W0] o [D0, H0, W0]

    Devuelve:
        features [N, C]
        targets  [N]
        metadata lista con batch, fragment_id y clase
    """

    mask_small = _prepare_mask(
        fragment_mask,
        feature_map.shape[-3:],
    )

    vectors = []
    targets = []
    metadata = []

    for batch_id in range(
        feature_map.shape[0]
    ):
        codes = torch.unique(
            mask_small[batch_id]
        )

        codes = codes[
            codes > 0
        ]

        for code_tensor in codes:
            code = int(
                code_tensor.item()
            )

            instance_mask = (
                mask_small[batch_id] == code
            )

            if not instance_mask.any():
                continue

            vector = _masked_average(
                feature_map[batch_id],
                instance_mask,
            )

            class_id = fragment_code_to_class(
                code
            )

            vectors.append(vector)
            targets.append(class_id)

            metadata.append({
                "batch_id": batch_id,
                "fragment_id": code,
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
            })

    if not vectors:
        empty_features = feature_map.new_empty(
            (0, feature_map.shape[1])
        )

        empty_targets = torch.empty(
            0,
            dtype=torch.long,
            device=feature_map.device,
        )

        return (
            empty_features,
            empty_targets,
            metadata,
        )

    features = torch.stack(
        vectors
    )

    targets = torch.tensor(
        targets,
        dtype=torch.long,
        device=feature_map.device,
    )

    return features, targets, metadata


def extract_pred_instance_features(
    feature_map: torch.Tensor,
    instance_map: torch.Tensor,
):
    """Extrae características de las instancias predichas.

    instance_map:
        0 = fondo
        1..N = IDs de instancia.
    """

    mask_small = _prepare_mask(
        instance_map,
        feature_map.shape[-3:],
    )

    vectors = []
    metadata = []

    for batch_id in range(
        feature_map.shape[0]
    ):
        instance_ids = torch.unique(
            mask_small[batch_id]
        )

        instance_ids = instance_ids[
            instance_ids > 0
        ]

        for instance_tensor in instance_ids:
            instance_id = int(
                instance_tensor.item()
            )

            mask = (
                mask_small[batch_id]
                == instance_id
            )

            if not mask.any():
                continue

            vector = _masked_average(
                feature_map[batch_id],
                mask,
            )

            vectors.append(vector)

            metadata.append({
                "batch_id": batch_id,
                "instance_id": instance_id,
            })

    if not vectors:
        empty = feature_map.new_empty(
            (0, feature_map.shape[1])
        )

        return empty, metadata

    return torch.stack(vectors), metadata


def classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Cross Entropy para las tres categorías anatómicas."""

    if logits.shape[0] == 0:
        return logits.sum() * 0.0

    return F.cross_entropy(
        logits,
        targets.long(),
    )


@torch.no_grad()
def classification_prediction(
    logits: torch.Tensor,
):
    """Clase y probabilidades por instancia."""

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    labels = torch.argmax(
        probabilities,
        dim=1,
    )

    return labels, probabilities


def _roc_auc_binary(
    target: np.ndarray,
    score: np.ndarray,
) -> float:
    """AUC binaria por rangos."""

    target = target.astype(bool)

    n_pos = int(target.sum())
    n_neg = int((~target).sum())

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(score)

    ranks = np.empty(
        len(score),
        dtype=float,
    )

    ranks[order] = np.arange(
        1,
        len(score) + 1,
    )

    rank_sum = ranks[target].sum()

    auc = (
        rank_sum
        - n_pos * (n_pos + 1) / 2
    ) / (
        n_pos * n_neg
    )

    return float(auc)


@torch.no_grad()
def classification_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
):
    """F1 y AUC por categoría anatómica y promedio macro."""

    if logits.shape[0] == 0:
        return {
            "macro_f1": float("nan"),
            "macro_auc": float("nan"),
        }

    pred, probs = classification_prediction(
        logits
    )

    pred = pred.cpu().numpy()
    probs = probs.cpu().numpy()
    target = targets.cpu().numpy()

    f1_values = []
    auc_values = []

    metrics = {}

    for class_id, name in enumerate(
        CLASS_NAMES
    ):
        true_class = (
            target == class_id
        )

        pred_class = (
            pred == class_id
        )

        tp = (
            true_class & pred_class
        ).sum()

        fp = (
            ~true_class & pred_class
        ).sum()

        fn = (
            true_class & ~pred_class
        ).sum()

        denominator = (
            2 * tp + fp + fn
        )

        f1 = (
            2 * tp / denominator
            if denominator > 0
            else 0.0
        )

        auc = _roc_auc_binary(
            true_class,
            probs[:, class_id],
        )

        metrics[
            f"f1_{name}"
        ] = float(f1)

        metrics[
            f"auc_{name}"
        ] = float(auc)

        f1_values.append(
            float(f1)
        )

        if not np.isnan(auc):
            auc_values.append(
                float(auc)
            )

    metrics["macro_f1"] = float(
        np.mean(f1_values)
    )

    metrics["macro_auc"] = (
        float(np.mean(auc_values))
        if auc_values
        else float("nan")
    )

    return metrics