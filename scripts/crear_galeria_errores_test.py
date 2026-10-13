"""Galería diagnóstica CBAM: 5 menores Dice 3D de pelvis + mejor caso del test.
El Dice es el de la máscara fusionada, la misma que se dibuja (`fusion_pelvis_dice` de `model3d evaluate`).
Ejecutar desde la raíz: uv run python scripts/inferir_casos_test.py && uv run python scripts/crear_galeria_errores_test.py
Necesita CT y GT .mha y `results/inferencia_informe/<id>/` obtenidos con model3d infer.
Las cajas GT se derivan de las etiquetas del .mha; NO son cajas originales del detector.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle, Patch

ROOT = Path(__file__).resolve().parents[1]
DICE = 'fusion_pelvis_dice'   # Dice de pelvis de la máscara fusionada
DATA = ROOT / 'data'
EVAL = ROOT / 'results/detection/unet3d_cbam_trab/eval/cases_test.csv'
PRED = ROOT / 'results/inferencia_informe'
OUT = ROOT / 'docs/figuras/informe'
ANAT_COLORS = ['#e15759', '#4e79a7', '#59a14f']


def ct_file(cid):
    for folder in ('PENGWIN_CT_train_images_part1', 'PENGWIN_CT_train_images_part2'):
        p = DATA / folder / f'{cid}.mha'
        if p.exists():
            return p
    raise FileNotFoundError(f'CT faltante para {cid}: comprobar data/PENGWIN_CT_train_images_part1/part2')


def aligned(path, reference):
    image = sitk.ReadImage(str(path))
    aligned_img = sitk.Resample(image, reference, sitk.Transform(),
                                sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(aligned_img)


def anatomical(gt):
    result = np.zeros(gt.shape, dtype=np.uint8)
    result[(gt >= 1) & (gt <= 10)] = 1
    result[(gt >= 11) & (gt <= 20)] = 2
    result[(gt >= 21) & (gt <= 30)] = 3
    return result


def anatomy_overlay(ax, mask):
    masked = np.ma.masked_where(mask == 0, mask)
    ax.imshow(masked, cmap=ListedColormap(ANAT_COLORS),
              norm=BoundaryNorm([.5, 1.5, 2.5, 3.5], 3),
              alpha=.63, interpolation='nearest')


def gt_boxes_from_labels(ax, labels_slice, min_pixels=12):
    # Rectángulos calculados desde cada ID de fragmento GT; NO son GT boxes guardadas.
    for label in np.unique(labels_slice):
        if label == 0:
            continue
        ys, xs = np.where(labels_slice == label)
        if len(xs) < min_pixels:
            continue
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                               fill=False, edgecolor='#00e5ff', linewidth=.85, linestyle='--'))


def pred_boxes_in_slice(boxes, z, ct_orig, ct):
    """Cajas predichas del corte z. Vienen en índices del CT original; se llevan a la rejilla reorientada (LPS)."""
    keys = {'corte', 'x0', 'y0', 'x1', 'y1'}
    out = []
    if not keys.issubset(boxes.columns):
        return out
    for _, row in boxes.iterrows():
        p0 = ct.TransformPhysicalPointToContinuousIndex(
            ct_orig.TransformContinuousIndexToPhysicalPoint((float(row.x0), float(row.y0), float(row.corte))))
        p1 = ct.TransformPhysicalPointToContinuousIndex(
            ct_orig.TransformContinuousIndexToPhysicalPoint((float(row.x1), float(row.y1), float(row.corte))))
        if int(round(p0[2])) == z:
            out.append((min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]), max(p0[1], p1[1])))
    return out


def draw_pred_boxes(ax, cajas):
    for x0, y0, x1, y1 in cajas:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor='#ffd400', linewidth=1))


def bounding_crop(gt, pred, z, pad=22):
    # Mismo recorte en los 4 paneles, contiene GT y prediccion en el corte.
    foreground = (gt[z] != 0) | (pred[z] != 0)
    h, w = foreground.shape
    ys, xs = np.where(foreground)
    if len(xs) == 0:
        return 0, w, 0, h
    # Cuadrado, para que todos los paneles tengan el mismo tamaño (puede salirse de la imagen: se ve negro)
    cx, cy = (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2
    half = max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) / 2 + pad
    return int(cx - half), int(cx + half), int(cy - half), int(cy + half)


def main():
    df = pd.read_csv(EVAL, dtype={'case_id': str})
    df = df[df['split'].eq('test')].sort_values(DICE)
    ids = df.head(5).case_id.tolist() + [df.iloc[-1].case_id]
    fig, axes = plt.subplots(6, 4, figsize=(14.5, 16.8), constrained_layout=True)
    summary = []
    for r, cid in enumerate(ids):
        ctpath = ct_file(cid)
        gtpath = DATA / 'PENGWIN_CT_train_labels' / f'{cid}.mha'
        predpath = PRED / cid / 'mascara_fusion.nii.gz'
        if not gtpath.exists() or not predpath.exists():
            raise FileNotFoundError(f'{cid}: falta {gtpath if not gtpath.exists() else predpath}')
        ct_orig = sitk.ReadImage(str(ctpath))
        ct = sitk.DICOMOrient(ct_orig, 'LPS')      # misma orientación en todos (el CT 071 viene con x e y invertidos)
        hu = sitk.GetArrayFromImage(ct).astype(np.float32)
        gtlabels = aligned(gtpath, ct)
        gt = anatomical(gtlabels)
        predraw = aligned(predpath, ct)
        pred = np.where((predraw >= 1) & (predraw <= 3), predraw, 0).astype(np.uint8)

        # Seleccionar corte con mayor DESACUERDO anatómico en píxeles,
        # entre cortes donde se observa tejido GT considerable (evita extremos vacíos).
        disagreement = (gt != pred) & ((gt > 0) | (pred > 0))
        errors = disagreement.reshape(gt.shape[0], -1).sum(axis=1)
        gt_areas = (gt > 0).reshape(gt.shape[0], -1).sum(axis=1)
        acceptable = gt_areas >= max(50, int(gt_areas.max()*.08))
        if acceptable.any():
            errors = np.where(acceptable, errors, -1)
        z = int(np.argmax(errors))
        good = (gt[z] > 0) & (pred[z] > 0)
        fn = (gt[z] > 0) & (pred[z] == 0)
        fp = (gt[z] == 0) & (pred[z] > 0)
        wrong_cls = (gt[z] > 0) & (pred[z] > 0) & (gt[z] != pred[z])
        overlap = (gt[z] > 0) & (pred[z] > 0) & (gt[z] == pred[z])
        dice_2d = (2*int(((gt[z]>0)&(pred[z]>0)).sum()) /
                   max(1, int((gt[z]>0).sum()+(pred[z]>0).sum())))
        img = np.clip((hu[z] - 50)/1250, 0, 1)
        boxespath = PRED/cid/'cajas.csv'
        boxes = pd.read_csv(boxespath) if boxespath.exists() else pd.DataFrame()
        crop = bounding_crop(gt,pred,z)
        x0,x1,y0,y1 = crop
        dice_3d = float(df.set_index('case_id').loc[cid, DICE])
        cajas = pred_boxes_in_slice(boxes, z, ct_orig, ct)
        for c, caption in enumerate(['CT', 'GT + cajas derivadas', 'CBAM + cajas predichas', 'Mapa de errores']):
            ax = axes[r,c]
            ax.imshow(img, cmap='gray',vmin=0,vmax=1)
            if c == 1:
                anatomy_overlay(ax, gt[z]); gt_boxes_from_labels(ax, gtlabels[z])
            elif c == 2:
                anatomy_overlay(ax, pred[z]); draw_pred_boxes(ax, cajas)
            elif c == 3:
                # GT coincidente, omision, exceso, confusion anatómica.
                error_mask = np.zeros(gt[z].shape,dtype=np.uint8)
                error_mask[overlap] = 1
                error_mask[fn] = 2
                error_mask[fp] = 3
                error_mask[wrong_cls] = 4
                ax.imshow(np.ma.masked_where(error_mask==0,error_mask),
                          cmap=ListedColormap(['#27ae60','#ff3030','#ffac26','#b74bff']),
                          norm=BoundaryNorm([.5,1.5,2.5,3.5,4.5],4),alpha=.85,
                          interpolation='nearest')
            ax.set_facecolor('black')
            ax.set_xlim(x0,x1);ax.set_ylim(y1,y0)
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f'{cid} · {caption}\nDice 3D={dice_3d:.4f} · Dice 2D={dice_2d:.3f} · corte {z}',fontsize=8.3)
        summary.append({'case_id':cid,'dice_3d':dice_3d,'corte':z,'dice_corte':dice_2d,
                        'pixeles_omitidos':int(fn.sum()),'pixeles_exceso':int(fp.sum()),
                        'pixeles_clase_incorrecta':int(wrong_cls.sum()),
                        'n_cajas_pred_corte':len(cajas)})
        print(f'{cid}: Dice 3D={dice_3d:.4f}, corte={z}, Dice 2D={dice_2d:.3f}, FN={fn.sum()}, FP={fp.sum()}')
    fig.suptitle('CBAM · cinco menores Dice 3D de pelvis y mejor caso (test, 15 CT; máscara fusionada)\n'
                 'Corte axial con más píxeles en desacuerdo de cada CT',fontsize=13)
    handles=[Patch(facecolor='#27ae60',label='Coincidencia ósea'),
             Patch(facecolor='#ff3030',label='GT omitido'),
             Patch(facecolor='#ffac26',label='Predicción adicional'),
             Patch(facecolor='#b74bff',label='Clase anatómica incorrecta'),
             Patch(fill=False,edgecolor='#00e5ff',linestyle='--',label='Caja derivada del GT'),
             Patch(fill=False,edgecolor='#ffd400',label='Caja predicha')]
    fig.legend(handles=handles,loc='outside lower center',ncol=3,fontsize=9,frameon=False)
    OUT.mkdir(parents=True,exist_ok=True)
    result = OUT/'figura_4_errores_cbam_5_peores_1_mejor.png'
    fig.savefig(result,dpi=200,bbox_inches='tight');plt.close(fig)
    pd.DataFrame(summary).to_csv(OUT/'figura_4_errores_resumen.csv',index=False)
    print('Figura:',result)
    print('Resumen:',OUT/'figura_4_errores_resumen.csv')
    print('NOTA: ranking por Dice de segmentación 3D, no por errores de detección. Cajas GT reconstruidas por etiqueta, no GT boxes oficiales.')

if __name__ == '__main__':
    main()
