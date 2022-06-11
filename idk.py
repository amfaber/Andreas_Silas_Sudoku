import numpy as np
def convert_to_sparse(onehot):
    idx = np.where(onehot)
    out = np.zeros((n**2, n**2), dtype = int)
    out[idx[:2]] = idx[2] + 1
    return out
def solver(partial_sudoku):
    n = int(np.sqrt(len(partial_sudoku)))
    sol = np.zeros((n**2, n**2, n**2, 2), dtype = int)
    done = sol[:, :, :, 1]
    in_progress = sol[:, :, :, 0]

    indices = tuple(np.meshgrid(np.arange(n**2), np.arange(n**2), indexing="ij"))
    roll_transform = lambda i, j: (i//n*n+j//n, i%n*n+j%n)
    roller = roll_transform(*indices)
    # roller = indices[0]//n*n+indices[1]//n, indices[0]%n*n+indices[1]%n
    transpose = indices[::-1]
    where_initial_onehot = np.where(partial_sudoku != 0)
    where_initial_onehot = tuple([*where_initial_onehot, partial_sudoku[where_initial_onehot]-1])
    done[where_initial_onehot] = 1
    idx = np.where((done.sum(axis = 0, keepdims = True) + done.sum(axis = 1, keepdims = True) + done.sum(axis = 2, keepdims = True) +
                    done[roller].sum(axis = 1, keepdims = True).repeat(n**2, axis = 1)[roller]) == 0)
    in_progress[idx] = 1

    def global_declare_done():
        nonlocal in_progress
        # global in_progress
        if done.sum() == n**4:
            # print("DONE!")
            return
        
        n_possible_per_cell = in_progress.sum(axis = 2, keepdims = True)
        coordinates = np.where((n_possible_per_cell == 1) & in_progress)
        cells_to_update = coordinates[:2]
        numbers_to_update = coordinates[2]

        if len(coordinates[0]) > 0:
        # if True:
            in_progress[coordinates] = 0

            in_progress[:, cells_to_update[1], numbers_to_update] = 0
            in_progress[cells_to_update[0], :, numbers_to_update] = 0
            rolled_cells_to_update = roll_transform(*cells_to_update)
            in_progress = in_progress[roller]
            in_progress[rolled_cells_to_update[0], :, numbers_to_update] = 0
            in_progress = in_progress[roller]
            done[coordinates] = 1
            global_declare_done()
    
    global_declare_done()

    return convert_to_sparse(done)

n = 3
ezsudoku=np.array([[6, 8, 0, 2, 0, 0, 1, 0, 0],
       [9, 0, 0, 6, 8, 7, 0, 0, 0],
       [0, 2, 0, 9, 0, 0, 0, 6, 8],
       [0, 0, 0, 0, 9, 0, 4, 0, 3],
       [3, 0, 9, 0, 0, 6, 0, 5, 1],
       [0, 0, 0, 3, 0, 5, 9, 2, 0],
       [0, 9, 8, 1, 0, 0, 7, 0, 4],
       [1, 0, 2, 4, 7, 0, 6, 0, 0],
       [7, 0, 4, 5, 0, 8, 0, 0, 0]])
ezsudoku_sol = np.array(
[[6, 8, 3, 2, 5, 4, 1, 9, 7],
[9, 1, 5, 6, 8, 7, 3, 4, 2],
[4, 2, 7, 9, 1, 3, 5, 6, 8],
[2, 5, 6, 8, 9, 1, 4, 7, 3],
[3, 4, 9, 7, 2, 6, 8, 5, 1],
[8, 7, 1, 3, 4, 5, 9, 2, 6],
[5, 9, 8, 1, 6, 2, 7, 3, 4],
[1, 3, 2, 4, 7, 9, 6, 8, 5],
[7, 6, 4, 5, 3, 8, 2, 1, 9]])
for i in range(1000):
    solver(ezsudoku)