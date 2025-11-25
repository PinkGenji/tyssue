""" The function in this script auto-controls the edges are active or not for a bilayer tissue sheet."""

def bilayer_dummy_set(sheet):
    """
    Set edges as active or dummy for bilayer tissue sheet.

    Parameters
    ----------
    sheet : tyssue.Sheet
        The tissue sheet instance containing face_df and edge_df DataFrames.
    """

    # Iterate through each edge in the edge DataFrame
    for i in sheet.edge_df.index:
        # Check if the edge has an opposite edge (i.e., it's internal)
        if sheet.edge_df.loc[i, 'opposite'] != -1:
            # Get the associated cell (face) for this edge
            associated_cell = sheet.edge_df.loc[i, 'face']
            # Get the opposite edge index
            opposite_edge = sheet.edge_df.loc[i, 'opposite']
            # Get the opposite cell (face) for the opposite edge
            opposite_cell = sheet.edge_df.loc[opposite_edge, 'face']
            # If both associated and opposite cells are of class 'STB', set edge as dummy (inactive)
            if (sheet.face_df.loc[associated_cell, 'cell_class'] == 'STB' and
                    sheet.face_df.loc[opposite_cell, 'cell_class'] == 'STB'):
                sheet.edge_df.loc[i, 'is_active'] = 0
                sheet.edge_df.loc[opposite_edge, 'is_active'] = 0
            else:
                # Otherwise, set edge as active
                sheet.edge_df.loc[i, 'is_active'] = 1
        else:
            # For boundary edges, set as active
            sheet.edge_df.loc[i, 'is_active'] = 1
