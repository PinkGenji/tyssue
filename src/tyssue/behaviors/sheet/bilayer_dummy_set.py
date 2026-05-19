""" The function in this script auto-controls the edges are active or not for a bilayer tissue sheet."""

from ...config.draw import sheet_spec

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
    print("Bilayer dummy edges have been set based on cell classes.")

def update_draw_specs(sheet):
    """
    Update drawing specifications for faces and edges based on cell class and edge activity.

    Returns:
        draw_specs (dict): ready to pass into sheet_view(...)
    """

    draw_specs = sheet_spec()
    draw_specs['face']['visible'] = True
    # Assign color by cell class
    sheet.face_df['color'] = sheet.face_df['cell_class'].map(
        lambda c: 0.7 if c == 'STB' else 0.1
    )
    draw_specs['face']['color'] = sheet.face_df['color']
    draw_specs['face']['alpha'] = 0.2

    # Edge appearance
    draw_specs['edge']['visible'] = True
    sheet.edge_df['width'] = sheet.edge_df['is_active'].map(
        lambda active: 0.5 if active == 1 else 2
    )
    draw_specs['edge']['width'] = sheet.edge_df['width']

    return draw_specs

def deactivate_cells(sheet, id_list):
    """
    Deactivate specific cells (e.g., corner cells) so they do not
    participate in energy minimisation. Also deactivate their vertices.
    """
    for cell_index in id_list:
        # Skip if the face no longer exists (after topology changes)
        if cell_index not in sheet.face_df.index:
            continue

        # Mark the cell as dead
        sheet.face_df.loc[cell_index, "is_alive"] = 0
        # Find all vertices belonging to edges of this cell
        edges = sheet.edge_df[sheet.edge_df["face"] == cell_index]
        for v in edges["srce"].tolist():
            if v in sheet.vert_df.index:
                sheet.vert_df.loc[v, "is_active"] = 0
