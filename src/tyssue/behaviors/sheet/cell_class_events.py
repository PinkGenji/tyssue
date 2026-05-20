"""
Event module for cell class transition rules, modify the details accordingly to your model.
=======================

"""

import numpy as np
from ...geometry.planar_geometry import PlanarGeometry
from ...topology.sheet_topology import cell_division
from ...behaviors.sheet.cell_activity_events import fuse_single_cell, stb_detach, stb_extrusion
from ...behaviors.sheet.bilayer_dummy_set import auto_dummy_edges, update_draw_specs

def cell_cycle_transition(sheet, manager, dt, p_recruit=0.1,
                          G1_duration= 8,
                          S_duration = 7,
                          G2_duration= 3,
                          M_duration = 0.5,
                          F_duration = 24,
                          E_duration = 30
                          ):
    """
    Controls cell class state transitions for cell cycle based on their residence time in specific class.
        - Cells in G1 transition to S after their timer elapses.
        - Cells in S transition to G2 with probability p_recruit after their timer elapses, otherwise they stay in S.
        - Cells in G2 transition to M after their timer elapses.
        - Cells in M undergo cell division after their timer elapses, and both mother and daughter cells reset to G1 with new timers.
    """
    # Generate two df that are cells that need to change its cell class or stay in its current class.
    cells_stay_in_class = sheet.face_df.loc[sheet.face_df['timer'] > 0].index.tolist()
    cells_to_change = sheet.face_df.loc[sheet.face_df['timer'] <= 0]

    # For cells that still needs to elapse the timer, just reduce the timer by dt
    for cell in cells_stay_in_class:
        sheet.face_df.loc[cell,'timer'] -= dt

    # Then we change the cell type based on the cell cycle diagram.
    G1_cells = cells_to_change.loc[cells_to_change['cell_class'] == 'G1'].index.tolist()
    S_cells = cells_to_change.loc[cells_to_change['cell_class'] == 'S'].index.tolist()
    G2_cells = cells_to_change.loc[cells_to_change['cell_class'] == 'G2'].index.tolist()
    M_cells = cells_to_change.loc[cells_to_change['cell_class'] == 'M'].index.tolist()
    F_cells = cells_to_change.loc[cells_to_change['cell_class'] == 'F'].index.tolist()
    STB_units = cells_to_change.loc[cells_to_change['cell_class'] == 'STB'].index.tolist()
    E_units = cells_to_change.loc[cells_to_change['cell_class'] == 'E'].index.tolist()

    # For cells in G1_cells indices, we simply change the cell class to S
    sheet.face_df.loc[G1_cells, 'cell_class'] = 'S'

    # For cells in S_cells , we need to loop and decide based on the random number generated for if it moves into G2.
    for cell in S_cells:
        if np.random.rand() < p_recruit:
            sheet.face_df.loc[cell, 'cell_class'] = 'G2'
            sheet.face_df.loc[cell, 'timer'] = G2_duration
        else:
            sheet.face_df.loc[cell, 'cell_class'] = 'F'
            sheet.face_df.loc[cell, 'timer'] = F_duration

    # For cells in G2_cells, we move them into M class
    sheet.face_df.loc[G2_cells, 'cell_class'] = 'M'

    # For cells in M_cells, they need to be undergone cell division
    for cell in M_cells:
        daughter = cell_division(sheet, mother=cell, geom = PlanarGeometry )
        # Set parent and daughter to G1 with G1 timer, note that variable daughter is the index of the new row already.
        sheet.face_df.loc[cell, 'cell_class'] = 'G1'
        sheet.face_df.loc[daughter, 'cell_class'] = 'G1'
        sheet.face_df.loc[cell, 'timer'] = G1_duration
        sheet.face_df.loc[daughter, 'timer'] = G1_duration

    # For cells in F_cells, we need to perform fusion.
    for cell in F_cells:
        fuse_single_cell(sheet,F_cell = cell)
        sheet.face_df.loc[cell, 'timer'] = F_duration

    # For cells in E class, we need to perform extrusion.
    for unit in E_units:
        stb_extrusion(sheet,unit)
    # For units in STB class that needs to be changed, we perform detachment.
    for unit in STB_units:
        stb_detach(sheet,geom,unit)
        sheet.face_df.loc[unit, 'cell_class'] = 'STB'
        sheet.face_df.loc[unit, 'timer'] = E_duration

    geom.update_all(sheet)
    auto_dummy_edges(sheet)
    update_draw_specs(sheet)
    manager.append(cell_cycle_transition, dt = dt, p_recruit = p_recruit,G2_duration = G2_duration, G1_duration = G1_duration)
