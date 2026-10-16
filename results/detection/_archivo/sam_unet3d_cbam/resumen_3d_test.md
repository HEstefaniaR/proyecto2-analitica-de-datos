# Baseline SAM zero-shot con el modelo 3D (rúbrica 4.2)

SAM ViT-B (`sam_vit_b_01ec64.pth`) zero-shot. Prompt: cada caja axial PREDICHA por el modelo 3D
(`unet3d_cbam`, época 130; post-NMS, puntaje ≥ 0.5); la región de la caja es el hueso que el modelo
predice dentro de ella. La máscara real nunca entra como prompt. Split test: 15 CT, 1 de cada
1 cortes axiales.

## Dice e IoU 3D por hueso (promedio de los CT)

```
                  dice_sam  iou_sam  dice_modelo  iou_modelo
region                                                      
left_hipbone         0.893    0.808        0.915       0.845
right_hipbone        0.898    0.818        0.920       0.852
sacrum               0.803    0.677        0.858       0.753
macro (3 huesos)     0.865    0.768        0.898       0.817
```

## Dice e IoU 2D por corte (promedio de los cortes con hueso real o predicho)

```
                  dice_sam  iou_sam  dice_modelo  iou_modelo
region                                                      
left_hipbone         0.871    0.795        0.880       0.810
right_hipbone        0.873    0.804        0.878       0.811
sacrum               0.591    0.502        0.620       0.538
macro (3 huesos)     0.778    0.700        0.793       0.720
```

## Latencia de SAM por corte con cajas (mps)

Media 0.69 s, mediana 0.65 s (set_image + predict).
