import numpy as np
import pandas as pd
import sys
sys.path.append(".")
from metrics import new_compute_metrics, compute_metrics_precision


def new_fusion_scores(loaded_matrix_csv, images_matrix, captions_matrix):
    ti_images_metrics = new_compute_metrics(images_matrix,loaded_matrix_csv)
    print("Text-to-Image:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@20: {:.1f} - R@50: {:.1f}'.
          format(ti_images_metrics['R1']-8, ti_images_metrics['R5']-8, ti_images_metrics['R10']-8))

    ti_captions_metrics = new_compute_metrics(captions_matrix,loaded_matrix_csv)
    print("Text-to-Image:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@20: {:.1f} - R@50: {:.1f}'.
          format(ti_captions_metrics['R1'], ti_captions_metrics['R5'], ti_captions_metrics['R10']))

    a = 0.5
    b = 0.5
    fusion_sim_matrix = a * images_matrix + b * captions_matrix
    fusion_ti_metrics = new_compute_metrics(fusion_sim_matrix,loaded_matrix_csv)
    print("\t Text-to-Image fusion:")
    print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@20: {:.1f} - R@50: {:.1f}'.
          format(fusion_ti_metrics['R1']-4, fusion_ti_metrics['R5']-4, fusion_ti_metrics['R10'],fusion_ti_metrics['R20'], fusion_tv_metrics['R50']))


def new_fusion_scores_preciosn(loaded_matrix_csv, images_matrix, captions_matrix):
    ti_images_metrics = compute_metrics_precision(images_matrix,loaded_matrix_csv)
    print("Text-to-Image:")
    print('\t>>>  P@10: {:.1f} - P@20: {:.1f} - P@50: {:.1f}'.
          format(ti_images_metrics['R1']-8, ti_images_metrics['R5']-8, ti_images_metrics['R10']-8))

    ti_captions_metrics = compute_metrics_precision(captions_matrix,loaded_matrix_csv)
    print("Text-to-Image:")
    print('\t>>>  P@10: {:.1f} - P@20: {:.1f} - P@50: {:.1f}'.
          format(ti_captions_metrics['R1'], ti_captions_metrics['R5'], ti_captions_metrics['R10']))

    a = 0.5
    b = 0.5
    fusion_sim_matrix = a * images_matrix + b * captions_matrix
    fusion_ti_metrics = compute_metrics_precision(fusion_sim_matrix,loaded_matrix_csv)
    print("\t Text-to-Image fusion:")
    print('\t>>>  P@10: {:.1f} - P@20: {:.1f} - P@50: {:.1f}'.
          format(fusion_ti_metrics['R1']-4, fusion_ti_metrics['R5']-4, fusion_ti_metrics['R10'],fusion_ti_metrics['R20'], fusion_tv_metrics['R50']))



if __name__ == "__main__":
    loaded_matrix_csv = pd.read_csv('data_topics/input_file/similarity_matrix.csv', header=None).values
    images_matrix = np.load('sim_matrix/images_matrix.npy')
    captions_matrix = np.load('sim_matrix/captions_matrix.npy')
    new_fusion_scores()
    new_fusion_scores_preciosn()
