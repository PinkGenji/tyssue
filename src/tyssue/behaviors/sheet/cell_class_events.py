"""
Event module for cell class transition rules, modify the details accordingly to your model.
=======================

"""

from ...geometry.sheet_geometry import SheetGeometry
from ...topology.sheet_topology import cell_division

def cell_cycle_transition(sheet, manager, dt, cell_id, p_recruit=0.1, G2_duration=0.4, G1_duration=0.11):
    """
    Controls cell class state transitions for cell cycle based on timers and probabilities.

    Parameters
    ----------
    sheet: tyssue.Sheet
        The tissue sheet.
    manager: EventManager
        The event manager scheduling the behaviour.
    cell_id: Integer
        ID of the cell being controlled.
    p_recruit: float
        Probability for an 'S' cell to be recruited to 'G2'.
    dt: float
        Time step increment.
    G2_duration: float
        Fixed duration cells stay in G2 phase.
    G1_duration: float
        Fixed duration cells stay in G1 phase.
    """

    # Record the current cell class
    current_class = sheet.face_df.loc[cell_id, 'cell_class']
    # (1) Recruit mature 'S' cells into G2 with probability p_recruit
    if current_class == 'S':
        if np.random.rand() < p_recruit:
            sheet.face_df.loc[cell_id, 'cell_class'] = 'G2'
            sheet.face_df.loc[cell_id, 'timer'] = G2_duration
        # append to next deque
        manager.append(cell_cycle_transition, dt=dt, cell_id=cell_id)

    # (2) Decrement timers for cells in G2; when timer ends, move to M
    elif current_class == 'G2':
        sheet.face_df.loc[cell_id, 'timer'] -= dt
        if sheet.face_df.loc[cell_id, 'timer'] <= 0:
            sheet.face_df.loc[cell_id, 'cell_class'] = 'M'
        # append to next deque
        manager.append(cell_cycle_transition, dt=dt, cell_id=cell_id)

    # (3) For cells in M, perform division and set daughters to G1 with timer
    elif current_class == 'M':
        daughter = cell_division(sheet, mother=cell_id, geom = SheetGeometry )
        # Set parent and daughter to G1 with G1 timer
        sheet.face_df.loc[cell_id, 'cell_class'] = 'G1'
        sheet.face_df.loc[daughter, 'cell_class'] = 'G1'
        sheet.face_df.loc[cell_id, 'timer'] = G1_duration
        sheet.face_df.loc[daughter, 'timer'] = G1_duration
        # append to next deque
        manager.append(cell_cycle_transition, dt=dt, cell_id=cell_id)
        manager.append(cell_cycle_transition, dt=dt, cell_id=daughter)

    # (4) Decrement timers for G1 cells; when timer ends, move to S
    elif current_class == 'G1':
        sheet.face_df.loc[cell_id, 'timer'] -= dt
        if sheet.face_df.loc[cell_id, 'timer'] <= 0:
            sheet.face_df.loc[cell_id, 'cell_class'] = 'S'
        # append to next deque
        manager.append(cell_cycle_transition, dt=dt, cell_id=cell_id)






