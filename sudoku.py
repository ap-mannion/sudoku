## Reference for some of the algorithms implemented in this code
#
# Generating highly balanced sudoku problems as hard problems, C. Ansotegui et al, Journal of Heuristics 2011
# https://www.cs.cornell.edu/gomes/pdf/2011_ansotegui_heuristics_sudoku.pdf
#
# Dancing Links, Donald E. Knuth
# https://arxiv.org/abs/cs/0011047

import numpy as np
from math import floor, ceil
from random import sample
from c2linkedlist import CDLList
# TODO:
#   - solvability checking for hole poking: hole pokes will have to be heavily refactored - make a copy of the
# grid that will have to be updated along with the indicator matrix to make sure there's a unique solution at
# each step, although it should be possible not to have to do the ENTIRE hole poking one-by-one with a call to
# solve() at every update
#       steps;
#       - insert check condition in solve() to break recursion & signal when the puzzle doesn't have a unique
# solution
#       - function PuzzleGrid.hasBeenSolved() to check whether `dlx_repr` corresponds to a filled grid
#   - method to convert a unique `dlx_repr` back to a grid
#   - hole poking algos: fully balanced, random, symmetric


class PuzzleGrid(np.ndarray):
    """
    Numpy array configured to always be a sudoku grid or Latin square.
    This will generate a filled sudoku on initialisation, and its methods
    are called within SudokuGenerator member functions.
    """

    def __new__(cls, d, m):
        assert(d%m == 0)
        puzzlegrid = super().__new__(cls, (d,d), dtype=np.int16)
        puzzlegrid.m = m

        return puzzlegrid

    def __init__(self, *args, **kwargs):
        # generate a canonical filled sudoku grid using permutations
        d = self.shape[0]
        n = int(d/self.m)
        symbols = np.arange(d, dtype=np.int16)+1
        np.random.shuffle(symbols)
        onecycle = lambda arr: np.concatenate((arr[1:], arr[:1]))
        prm_index = self.m*(n-1)
        grid = np.copy(symbols)
        for i in range(self.m):
            row_iters = n-1 if i == 0 else n
            for _ in range(row_iters):
                permutation = np.concatenate((symbols[prm_index:], symbols[:prm_index]))
                grid = np.vstack((grid, permutation))
                symbols = permutation
            symbols = np.concatenate(
                [onecycle(symbols[self.m*k:self.m*(k+1)]) for k in range(n)]
            )
        self.setfield(grid, self.dtype)
        self.d = d
        self.n = n

    def holesPoked(self):
        """
        Check whether or not the puzzle has been prepared yet
        """
        return len(self[self == 0]) > 0

    def latinSquareTraversal(self, iterations=None, random_params=False):
        """
        Implementation of the Markov chain-based algorithm for sampling from the set of all
        possible Latin squares described in section 3 of Ansotegui et al 2011
        """
        # parameter initialisation
        if random_params:
            r1, r2, symbol = (np.random.randint(self.d) for _ in range(3))
            symbol += 1 # has to be non-zero
        else:
            r1, r2, symbol = self._chooseLSTparams()
        iters = np.random.randint(self.d-1)+1 if iterations is None else iterations
        if r1 == r2: # these two can't be equal
            r2 += 1
            r2 = r2%self.d

        # main traversal
        symbol_index, = np.where(self[r1,:] == symbol)[0]
        for i in range(iters):
            tmp = self[r1,symbol_index]
            self[r1,symbol_index] = self[r2,symbol_index]
            self[r2,symbol_index] = tmp
            new_r1_indices, = np.where(self[r1,:] == self[r1,symbol_index])
            try:
                symbol_index = new_r1_indices[new_r1_indices != symbol_index][0]
            except IndexError:
                # this is thrown when the sequence of swaps has gone through a full
                # cycle, i.e. the last symbol to have been swapped into row 1 is the
                # original symbol selected randomly at the start of the function, so
                # the grid is back to a valid Latin square and the loop can be exited.
                # Because the base grid is constructed using n-cycles, this will happen
                # whenever iters>n and both selected rows are as yet unmodified. 
                # In general, any pair of rows will have a max. number of possible
                # swaps corresponding to the max. length of the disjoint cycles required
                # to transform one into the other, and whenever the randomly selected
                # number of iterations is greater than that max, the traversal will have
                # to terminate before the loop finishes
                i = -1
                break

        # adjust the grid back to a Latin square
        if i >= 0:
            # complete a cycle back to the original symbol using another row
            final_swap_row, = np.where(self[:,symbol_index] == symbol)
            tmp = self[r1,symbol_index]
            self[r1,symbol_index] = self[final_swap_row,symbol_index]
            self[final_swap_row,symbol_index] = tmp
            if type(final_swap_row) not in {int, np.int16, np.int32, np.int64}:
                final_swap_row = final_swap_row[0] 
            if not self.isLS():
                self._adjustRowPairToLS(r2, final_swap_row)

    def isLS(self):
        for i, row in enumerate(self):
            for a in (row, self[:,i]):
                if len(a) != len(np.unique(a)):
                    return False

        return True

    def isSudoku(self):
        if not self.isLS():
            return False
        
        for i in range(self.m):
            for j in range(self.n):
                block = self[self.n*j:self.n*(j+1),self.m*i:self.m*(i+1)]
                if len(np.unique(block)) != block.size:
                    return False

        return True

    def nHolesByDiff(self, diff):
        """
        Return a range for the number of holes to make in the grid based on how hard we want the
        puzzle to be
        """
        lower_limits = {4:4, 6:8, 8:14, 9:17, 10:22, 12:32, 16:55}
        cmax = int(self.d**2/2+self.d/2)
        if diff == "beginner":
            cmin = cmax-self.d+1
        elif diff == "easy":
            cmax -= self.d+1
            cmin = cmax-ceil(self.d/2)
        elif diff == "medium":
            cmax -= self.d+1
            cmin = cmax-self.d+1
        elif diff == "hard":
            cmax -= ceil(self.d*1.5)
            cmin = cmax-self.d
            if cmin-lower_limits[self.d] > 3*self.d:
                cmax -= 2*self.d
                cmin = cmax-self.d
        elif diff == "elite":
            cmin = lower_limits[self.d]
            cmax = cmin+self.d
        else:
            raise ValueError(f"Invalid `diff` str passed: {diff}")
    
        if cmin < lower_limits[self.d]:
            cmin = lower_limits[self.d]
            cmax = cmin+self.d

        hmin, hmax = map(lambda clues: self.d**2-clues, (cmax, cmin))

        return hmin, hmax

    def solve(self, current_solution=[]):
        """
        Use the DLX algorithm to solve the puzzle as an exact cover problem
        """
        if not self.holesPoked():
            raise RuntimeError("Puzzle is filled!")
        if hasattr(self, "solution_coord_list"):
            raise RuntimeError("Puzzle already solved!")
        if not hasattr(self, "dlx_repr"):
            self._generateAlgoXConstraintMatrix()

        def getMinColSum(cap=None, exclude=[]):
            cols = self.dlx_repr["C"]
            if len(cols) == len(exclude):
                # this should only happen when the grid has been solved
                return 0
            min_colsum = cap if cap is not None else len(self.dlx_repr["R"])
            for c, ll in cols.items():
                if c not in exclude:
                    if ll.top.colsum < min_colsum:
                        min_colsum = ll.top.colsum
            return min_colsum

        def getColIndexFromRow(node, row_idx):
            if node.val == "rc":
                idx1, idx2 = row_idx[:2]
            elif node.val == "rn":
                idx1, idx2 = row_idx[0], row_idx[2]
            elif node.val == "cn":
                idx1, idx2 = row_idx[1:]
            elif node.val == "bn":
                idx1 = self.getBoxNum(*row_idx[:2])
                idx2 = row_idx[2]
            return node.val, idx1, idx2

        # get all columns with the min. number of non-zero rows
        min_colsum = getMinColSum()
        min_cols = []
        for ll in self.dlx_repr["C"].values():
            if ll.top.colsum == min_colsum:
                min_cols.append(ll)
        
        # select one of these columns at a time
        while len(min_cols) > 0:
            selected_col = min_cols.pop(0) 
            solution_col_node = selected_col.top.next
            while solution_col_node.val != selected_col.top.val:
                # this is where a solution branch (row label to add to the partial solution) is selected
                # if it doesn't work out we backtrack and iterate on to the next row
                solution_row_idx = solution_col_node.val
                removal_row_indices = []
                row = self.dlx_repr["R"][solution_row_idx]
                row_node = row.top

                # select all non-zero columns in that row and all non-zero rows in each of those columns
                full_removal_col_indices = [getColIndexFromRow(row_node, solution_row_idx)]
                row_node = row_node.next
                while row_node.val != row.top.val:
                    # iterate through the row nodes to get its non-zero columns
                    rem_col_idx = getColIndexFromRow(row_node, solution_row_idx)
                    full_removal_col_indices.append(rem_col_idx)
                    rem_col = dlx_repr["C"][rem_col_idx]
                    rem_col_node = rem_col.top.next
                    while rem_col_node.val != rem_col.top.val:
                        # iterate through each selected column to remove non-zero rows
                        if rem_col_node.val not in removal_row_indices:
                            removal_row_indices.append(rem_col_node.val)
                        rem_col_node = rem_col_node.next
                    row_node = row_node.next

                # remove all the values in these rows from all columns
                backtrack_dict = {}
                for col_idx, col_ll in dlx_repr["C"].items():
                    backtrack_dict[col_idx] = []
                    for row_idx in removal_row_indices:
                        try:
                            if col_idx not in full_removal_col_indices:
                                partial_removal_col_node = col_ll.getNodeByVal(row_idx)
                                bktr_val = (row_idx, partial_removal_col_node.next.val)
                            else:
                                bktr_val = row_idx
                            col_ll.remove(row_idx)
                            backtrack_dict[col_idx].append(bktr_val)
                        except ValueError:
                            continue
                
                min_colsum = getMinColSum(exclude=full_removal_col_indices)
                if min_colsum == 0:
                    if sum([ll.top.colsum for ll in self.dlx_repr["C"].values()]) > 0:
                        # backtracking...
                        for col_idx, bktr_vals in backtrack_dict.items():
                            # the next-node memory backtracking for columns that aren't
                            # being totally removed needs the backtrack to iterate backwards
                            # through the memory while the columns being completely deleted
                            # need to iterate forwards :)
                            if type(bktr_vals[0]) is tuple:
                                for t in reversed(bktr_vals):
                                    self.dlx_repr["C"][col_idx].insert(*t, before=True)
                            else:
                                for row_idx in bktr_vals:
                                    self.dlx_repr["C"][col_idx].insert(row_idx)
                        solution_col_node = solution_col_node.next
                    else:
                        # Finishing point
                        current_solution.append(solution_row_idx)
                        self.solution_coord_list = current_solution
                        return True
                else:
                    current_solution.append(solution_row_idx)
                    self.dlx_repr["R"] = {
                        k:v for k, v in self.dlx_repr["R"].items() if k not in removal_row_indices
                    }
                    self.dlx_repr["C"] = {
                        k:v for k, v in self.dlx_repr["C"].items() if k not in full_removal_col_indices
                    }
                    solved = self.solve(current_solution)
                    if solved:
                        break
                if solved:
                    break

    def _getBoxNum(r, c):
        """
        Returns an int in [0,d-1] - boxes numbered L to R along rows 
        """
        return (self.m-1)*floor(r/self.m)+floor(c/self.n) 

    def _generateAlgoXConstraintMatrix(self):
        """
        Generate Algorithm X constraint matrix: sparse representation using linked lists
        """
        dlx_cols, dlx_rows = {}, {}
        constraint_codes = ["rc", "rn", "cn", "bn"]

        # initialise column LLs
        for i in range(self.d):
            for j in range(self.d):
                for cc in constraint_codes:
                    col_idx = (cc, i, j if cc[1] == "n" else j+1)
                    dlx_cols[col_idx] = CDLList(col_idx, colheader=True)

        # reference dictionaries for easier construction of the box-constraint LLs
        box_ref = {"vals":{}, "coords":{}}
        br_key = 0
        for i in range(self.m):
            for j in range(self.n):
                ilims = self.m*i, self.m*(i+1)
                jlims = self.n*j, self.m*(j+1)
                box_ref["vals"][br_key] = self[ilims[0]:ilims[1],jlims[0]:jlims[1]]
                box_ref["coords"][br_key] = [(k,l) for k in range(*ilims) for l in range(*jlims)]
                br_key += 1

        symbols = np.arange(self.d)+1
        # fill value fields for the column LLs: each node value will be a row index (key)
        for c, ll in dlx_cols.items():
            header_cc = c[0]
            prev_node = ll.top
            if header_cc == "rc":
                i, j = c[1], c[2]
                if self[i,j] == 0:
                    for s in symbols:
                        ll.insert((i, j, s), prev_node.val)
                        prev_node = prev_node.next
                else:
                    ll.insert((i, j, self[i,j]), prev_node.val)
            elif header_cc == "rn":
                i, s = c[1], c[2]
                if s in self[i,:]:
                    j, = np.where(self[i,:] == s)
                    ll.insert((i, j.item(), s), prev_node.val)
                else:
                    for j in range(self.d):
                        ll.insert((i, j, s), prev_node.val)
                        prev_node = prev_node.next
            elif header_cc == "cn":
                j, s = c[1], c[2]
                if s in self[:,j]:
                    i, = np.where(self[:,j] == s)
                    ll.insert((i.item(), j, s), prev_node.val)
                else:
                    for i in range(self.d):
                        ll.insert((i, j, s), prev_node.val)
                        prev_node = prev_node.next
            elif header_cc == "bn":
                b, s = c[1], c[2]
                if s in box_ref["vals"][b]:
                    s_idx = np.where(self == s)
                    coord_set = set([(k,l) for k, l in zip(*s_idx)])
                    i, j = coord_set.intersection(set(box_ref["coords"][b])).pop()
                    ll.insert((i, j, s), prev_node.val)
                else:
                    for i, j in box_ref["coords"][b]:
                        ll.insert((i, j, s), prev_node.val)
                        prev_node = prev_node.next

        # Row LLs: the dictionary keys will be again a unique identifier, while the node values
        # will just contain the constraint type
        def makeRowLL():
            ll = CDLList(constraint_codes[0])
            prev_node = constraint_codes[0]
            for cc in constraint_codes[1:]:
                ll.insert(cc, prev_node)
                prev_node = cc
            return ll

        for r in range(self.d):
            for c in range(self.d):
                if self[r,c] != 0:
                    row_idx = (r, c, self[r,c])
                    dlx_rows[row_idx] = makeRowLL()
                else:
                    for s in symbols:
                        row_idx = (r, c, s)
                        dlx_rows[row_idx] = makeRowLL()

        self.dlx_repr = {"R":dlx_rows, "C":dlx_cols}

    def _chooseLSTparams(self):
        v_row_indices, v_symbols = [], []
        for i in range(self.m):
            for j in range(self.n):
                block = self[self.n*j:self.n*(j+1),self.m*i:self.m*(i+1)]
                if len(np.unique(block)) != block.size:
                    symbols, counts = np.unique(block, return_counts=True)
                    duplicates = symbols[counts != 1]
                    for duplicate in duplicates:
                        v_symbols.append(duplicate)
                        duplicate_row_indices, _ = np.where(block == duplicate)
                        for dr_index in duplicate_row_indices:
                            v_row_indices.append(self.n*j+dr_index)
        r1, r2 = (np.random.choice(v_row_indices) for _ in range(2))
        s = np.random.choice(v_symbols)

        return r1, r2, s

    def _overlappingDuplicateSwap(self, r1, r2, i):
        tmp = self[r1,i]
        self[r1,i] = self[r2,i]
        self[r2,i] = tmp

    def _adjustRowPairToLS(self, r1, r2):
        di1, di2 = map(self._getDuplicateIndices, (r1, r2))
        overlap = set(di1).intersection(set(di2))
        if len(overlap) == 1:
            for i in overlap:
                self._overlappingDuplicateSwap(r1, r2, i)
        else:
            swap_index = di1[0]
            tmp = self[r1,swap_index]
            self[r1,swap_index] = self[r2,swap_index]
            self[r2,swap_index] = tmp
            new_swap_indices, = np.where(self[r1,:] == self[r1,swap_index])
            while len(new_swap_indices) > 1:
                swap_index = new_swap_indices[new_swap_indices != swap_index].item()
                tmp = self[r1,swap_index]
                self[r1,swap_index] = self[r2,swap_index]
                self[r2,swap_index] = tmp
                new_swap_indices, = np.where(self[r1,:] == self[r1,swap_index])

    def _getDuplicateIndices(self, r):       
        symbols, counts = np.unique(self[r,:], return_counts=True)
        duplicate_indices, = np.where(self[r,:] == symbols[counts != 1].item())  

        return duplicate_indices


class SudokuGenerator:
    """
    Object to store & generate puzzles
    """

    def __init__(self, m=3, n=3, max_gen_cycles=10):
        self.max_gen_cycles = max_gen_cycles
        self.base_puzzle = PuzzleGrid(d=m*n, m=m)
        self.puzzle_list = [self.base_puzzle]
        self.max_shuffles_before_restart = int(2*(m*n)*1e3/3)

    def generate(self, log_mode=False):
        if log_mode:
            from sys import stdout
            from time import time

            start = time()

        new_grid = self.puzzle_list[-1]
        new_grid.latinSquareTraversal(random_params=True)
        loop = 0
        while not new_grid.isSudoku():
            new_grid.latinSquareTraversal()
            loop += 1
            if log_mode:
                stdout.write("\r")
                stdout.flush()
                stdout.write("# shuffles: "+str(loop))
                stdout.flush()
            if loop%self.max_shuffles_before_restart == 0:
                # start again if a valid Sudoku hasn't been reached after a certain number of tries
                new_grid = self.puzzle_list[-1]
                new_grid.latinSquareTraversal(random_params=True)
                if log_mode: print("Restarting...")
        self.out_puzzle = new_grid
        self.puzzle_list.append(new_grid)

        if log_mode:
            end = time()
            t = round(end-start, 6) 
            if not hasattr(self, 'log'):
                self.log = {'times':[], 'travs':[]}
            self.log['times'].append(t)
            self.log['travs'].append(loop)
            
    def singlyBalancedHolePattern(self, diff, n_steps=50):
        """
        Implements the hole pattern algorithm described in section 4.1 of Ansotegui et al 2011
        """
        if self.out_puzzle.holesPoked():
            raise RuntimeWarning("Attempted to apply hole pattern algorithm to puzzle that's already prepared")
            return
        hrange = self.out_puzzle.nHolesByDiff(diff)
        n_holes = np.random.choice([i for i in range(*hrange)])
        d = self.out_puzzle.d
        q = floor(n_holes/d)

        # canonical singly-balanced indicator matrix
        first_row = np.concatenate((np.ones(q), np.zeros(d-q)))
        h_indic = np.copy(first_row)
        permute = lambda arr, i: np.concatenate((arr[i:], arr[:i]))
        for i in range(d-1):
            h_indic = np.vstack((h_indic, permute(first_row, i+1)))

        # scramble it using 4-cycles
        step = 0
        while step < n_steps:
            i, j = (np.random.choice(d) for _ in range(2))
            k, l = map(lambda fi: np.random.choice([n for n in range(fi, d)]), (i, j))
            if (h_indic[i,j]+h_indic[k,l] == 2) & (h_indic[i,l]+h_indic[k,j] == 0):
                h_indic[i,j] -= 1; h_indic[k,l] -= 1
                h_indic[i,l] += 1; h_indic[k,j] += 1
                step += 1

        if n_holes%d != 0:
            # if the dimension of the puzzle doesn't divide the number of holes chosen,
            # we randomly select another n_holes mod d cells with no rows or columns in
            # common in which to put holes
            filled_coords = np.where(h_indic == 0)
            candidate_remdr_coords, rows_tracker, cols_tracker = [], [], []
            for i, j in zip(*filled_coords):
                if (i in rows_tracker) | (j in cols_tracker):
                    continue
                rows_tracker.append(i)
                cols_tracker.append(j)
                candidate_remdr_coords.append((i,j))
            remdr_coords = sample(candidate_remdr_coords, n_holes%d)
            for r, c in remdr_coords:
                h_indic[r,c] = 1

        self.out_puzzle[np.where(h_indic == 1)] = 0

    def doublyBalancedHolePattern(self, diff, n_steps=50):
        """
        Implements the hole pattern algorithm described in section 4.2 of Ansotegui et al 2011
        """
        if self.out_puzzle.holesPoked():
            raise RuntimeWarning("Attempted to apply hole pattern algorithm to puzzle that's already prepared")
            return
        hrange = self.out_puzzle.nHolesByDiff(diff)
        n_holes = np.random.choice([i for i in range(*hrange)])
        d, m, n = self.out_puzzle.d, self.out_puzzle.m, self.out_puzzle.n
        q = floor(n_holes/d)

        # use a canonical sudoku grid to initialise a doubly-balanced pattern
        pattern_ref_sdk = PuzzleGrid(d, m)
        h_indic = np.zeros((d,d))
        h_indic[np.where(sum((pattern_ref_sdk == k+1) for k in range(q)))] = 1
        if n_holes%d != 0:
            # select a random subset of (n_holes mod d) cells from another single-symbol pattern in the
            # reference grid to poke the remainder holes - in this case each box will have a singly-
            # balanced pattern, but (n_holes mod d) of them will have q+1 holes while the rest have q
            remdr_ref_pattern = np.where(pattern_ref_sdk == q+1)
            remdr_indices = np.random.choice(d, n_holes%d, replace=False)
            remdr_rows, remdr_cols = map(lambda x: x[remdr_indices], remdr_ref_pattern)
            for r, c in zip(remdr_rows, remdr_cols):
                h_indic[r,c] = 1

        # scramble using 4-cycles but make sure each is contained within a box
        box_xygen = lambda r=0: (np.random.choice(m-r), np.random.choice(n-r))
        step, change = 0, 0
        print(f"\nDBHP 4-cycling: max. {n_steps} steps...")

        def choose_higher_coord(fi, b):
            try:
                ret = np.random.choice([a for a in range(fi+1, b)])
            except ValueError:
                ret = fi+1
            return ret

        while step < n_steps:
            bx, by = box_xygen() # box coordinates on mxn grid
            i, j = box_xygen(1) # internal box coordinates: can't have i==m or j==n
            k, l = map(choose_higher_coord, (i, j), (m, n))
            i += bx*m; j += by*n
            k += bx*m; l += by*n
            if (h_indic[i,j]+h_indic[k,l] == 2) & (h_indic[i,l]+h_indic[k,j] == 0):
                h_indic[i,j] -= 1; h_indic[k,l] -= 1
                h_indic[i,l] += 1; h_indic[k,j] += 1
                change += 1
            step += 1
        print(f"{change} 4-cycle changes made")

        self.out_puzzle[np.where(h_indic == 1)] = 0            


# GENERATION TESTING
if __name__ == "__main__":
    from argparse import ArgumentParser
    from subprocess import check_output
    from os import path
    import matplotlib.pyplot as plt

    parser = ArgumentParser()
    parser.add_argument("--runs", type=int, default=25)
    parser.add_argument("-m", type=int, default=3)
    parser.add_argument("-n", type=int, default=3)
    args = parser.parse_args()

    filename = "sdk_generate_benchmarking_"+str(args.m)+"x"+str(args.n)+"_"
    fnum = 0
    sdkgen = SudokuGenerator(args.m, args.n)
    dev_files = check_output(["ls", "dev/generation_benchmarking"]).decode("utf-8").split("\n")
    while filename+str(fnum)+".txt" in dev_files:
        fnum += 1
    with open(path.join("dev/generation_benchmarking", filename+str(fnum)+".txt"), "w+") as outfile:
        outfile.write(f"=== SUDOKU FILLED {args.m}x{args.n} GRID GENERATION TIMING ===\n\n")
        for k in range(args.runs):
            print(f"Run {k+1}...")
            sdkgen.generate(log_mode=True)
            t = sdkgen.log['times'][-1]      
            print(f"\n{t} sec.")
            outfile.write(f"\n- Run {k+1}: Time {t}s\n{sdkgen.out_puzzle}\n")

        times_ms = [t*1e3 for t in sdkgen.log['times']]
        avgtime = round(np.mean(times_ms), 2)
        avgloop = round(np.mean(sdkgen.log['travs']), 2)

        outfile.write(f"MEAN GENERATION TIME {avgtime} ms, {avgloop} TRAVERSALS")
    
    plt.figure()
    plt.plot(times_ms, linestyle="-", label=f"Time (ms), mean {avgtime}")
    plt.plot(sdkgen.log['travs'], linestyle="-", label=f"#LS traversals, mean {avgloop}")
    plt.legend(loc="best")
    plt.xlabel("RUNS")
    plt.title(f"{args.m}x{args.n} SUDOKU GENERATION TIMING")
    plt.savefig(path.join("dev", filename+str(fnum)+".png"))