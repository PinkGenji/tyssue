# This script contains a function that auto updates the drawing specifications for a bilayer tissue sheet.

import numpy as np

def bilayer_draw_spec_update(sheet, specs):
    """
    Update drawing specifications for bilayer tissue sheet.

    Parameters
    ----------
    sheet : tyssue.Sheet
        The tissue sheet instance containing face_df and edge_df DataFrames.
    specs : dict
        The current drawing specifications dictionary to be updated.
    """

    # --- FACE COLOR UPDATE ---
    # Use NumPy vectorization to assign colors:
    # If 'cell_class' == 'STB', then color = 0.7
    # Else, then color = 0.1
    sheet.face_df['color'] = np.where(
        sheet.face_df['cell_class'] == 'STB', 0.7, 0.1
    )

    # Update the specs dictionary with the new face colors
    specs['face']['color'] = sheet.face_df['color'].to_numpy()

    # Set transparency (alpha) for faces
    specs['face']['alpha'] = 0.2

    # --- EDGE WIDTH UPDATE ---
    # Use NumPy vectorization to assign edge widths:
    # If 'is_active' == 0, then width = 2
    # Else, then width = 0.5
    sheet.edge_df['width'] = np.where(
        sheet.edge_df['is_active'] == 0, 2, 0.5
    )

    # Update the specs dictionary with the new edge widths
    specs['edge']['width'] = sheet.edge_df['width'].to_numpy()
