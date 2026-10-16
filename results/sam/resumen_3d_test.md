# Baseline SAM zero-shot con el modelo 3D (rúbrica 4.2)

SAM ViT-B (`sam_vit_b_01ec64.pth`) zero-shot. Prompt: cada caja axial PREDICHA por el modelo 3D
(`unet3d_cbam_trab`, época 130; post-NMS, puntaje ≥ 0.5); la región de la caja es el hueso que el modelo
predice dentro de ella. La máscara real nunca entra como prompt. Split test: 15 CT, 1 de cada
1 cortes axiales.

## Dice e IoU 3D por hueso (promedio de los CT)

```
                  dice_sam  iou_sam  dice_modelo  iou_modelo
region                                                      
left_hipbone         0.902    0.824        0.958       0.919
right_hipbone        0.908    0.835        0.961       0.925
sacrum               0.832    0.716        0.937       0.883
macro (3 huesos)     0.881    0.791        0.952       0.909
```

## Dice e IoU 2D por corte (promedio de los cortes con hueso real o predicho)

```
                  dice_sam  iou_sam  dice_modelo  iou_modelo
region                                                      
left_hipbone         0.886    0.813        0.939       0.897
right_hipbone        0.895    0.826        0.949       0.909
sacrum               0.694    0.593        0.810       0.743
macro (3 huesos)     0.825    0.744        0.899       0.850
```

## Latencia de SAM por corte con cajas (mps)

Media 1.57 s, mediana 0.64 s (set_image + predict).
