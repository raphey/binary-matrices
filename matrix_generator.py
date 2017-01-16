__author__ = 'raphey'

from random import randint
from math import factorial
import time


# Returns an n by n matrix of random bits
# Can leave the last row and column all zeros to take advantage of degrees of freedom with inversions.
# (No longer used in current version--random exploration is too slow.)
def make_random_matrix(n):
    mat = []
    for i in range(0, n):
        mat.append([])
        for j in range(0, n):
            mat[i].append(randint(0, 1))
    return mat


# Prints a matrix
def print_matrix(mat):
    print()
    for row in mat:
        printing_row = []
        for entry in row:
            printing_row.append(str(entry))
        print('\t'.join(printing_row))


# No longer used.
# Using row/col bit-flips, makes the bottom row and rightmost col of a matrix all 0s
def zero_edges(mat):
    n = len(mat)

    for i in range(0, n):
        if mat[i][n - 1] == 1:
            for j in range(0, n):
                mat[i][j] = 1 - mat[i][j]

    for j in range(0, n - 1):               # No need to check the last column
        if mat[n - 1][j] == 1:
            for i in range(0, n):
                mat[i][j] = 1 - mat[i][j]


# Using row bit-flips, minimizes the number of 1's in each row
def minimize_rows(mat):
    n = len(mat)
    is_changed = False
    for i in range(0, n):
        total = 0
        for j in range(0, n):
            total += mat[i][j]
        if total > n / 2:
            is_changed = True
            for j in range(0, n):
                mat[i][j] = 1 - mat[i][j]
    return is_changed


# Using col bit-flips, minimizes the number of 1's in each col
def minimize_cols(mat):
    n = len(mat)
    is_changed = False
    for j in range(0, n):
        total = 0
        for i in range(0, n):
            total += mat[i][j]
        if total > n / 2:
            is_changed = True
            for i in range(0, n):
                mat[i][j] = 1 - mat[i][j]
    return is_changed


# No longer used. Old function to minimize total score for a given matrix, to put it in canonical form.
def old_minimize_total(mat):
    n = len(mat)
    best_total_index = (0, 0)
    best_total = n ** 2
    for i in range(0, 2 ** n):
        rows_to_flip = []
        bin_i = bin(i + 2 ** n)[3:]
        for s in range(0, len(bin_i)):
            if bin_i[s] == "1":
                rows_to_flip.append(s)
        for k in rows_to_flip:
            invert_row(mat, k)

        for j in range(0, 2 ** (n - 1)):
            cols_to_flip = []
            bin_j = bin(j + 2 ** n)[3:]
            for t in range(0, len(bin_j)):
                if bin_j[t] == "1":
                    cols_to_flip.append(t)
            for l in cols_to_flip:
                invert_col(mat, l)
            new_total = total_score(mat)
            if new_total < best_total:
                best_total = new_total
                best_total_index = (i, j)
            for l in cols_to_flip:
                invert_col(mat, l)

        for k in rows_to_flip:
            invert_row(mat, k)

    bin_i_f = bin(best_total_index[0] + 2 ** n)[3:]
    for s in range(0, len(bin_i_f)):
        if bin_i_f[s] == "1":
            invert_row(mat, s)

    bin_j_f = bin(best_total_index[1] + 2 ** n)[3:]
    for t in range(0, len(bin_j_f)):
        if bin_j_f[t] == "1":
            invert_col(mat, t)


# Flips all bits in a row of a matrix.
def invert_row(mat, i):
    n = len(mat)
    for j in range(0, n):
        mat[i][j] = 1 - mat[i][j]


# Flips all bits in a column of a matrix
def invert_col(mat, j):
    n = len(mat)
    for i in range(0, n):
        mat[i][j] = 1 - mat[i][j]


# Assigns a total score to a matrix based on the number of 1's, with the numbers of free rows and free columns as
# tie-breakers. Note for future edits: a tuple is a better way to handle this than floats.
def total_score(mat):
    n = len(mat)
    total = 0
    for i in range(0, n):
        for j in range(0, n):
            total += mat[i][j]
    total -= num_free_rows(mat) / 1000          # Weights total score to slightly favor having free rows
    total -= num_free_cols(mat) / 1000000          # Same for columns, but further deprioritized
    return total


# Bubble sorting by row, with decreasing number of 1's. (Inefficient sort doesn't seem to slow anything down--number
# of items in the matrices is tiny.)
def sort_by_row(mat):
    n = len(mat)
    sums = []
    for i in range(0, n):
        sums.append(0)
        for j in range(0, n):
            sums[i] += mat[i][j]

    for q in range(0, n - 1):               # Each time through this loop is one pass through with bubble sort
        for i in range(0, n - 1):
            if sums[i + 1] > sums[i] or (sums[i + 1] == sums[i] and mat[i + 1] > mat[i]):
                temp_sum = sums[i]
                sums[i] = sums[i + 1]
                sums[i + 1] = temp_sum
                temp_row = mat[i][:]
                for j in range(0, n):
                    mat[i][j] = mat[i + 1][j]
                    mat[i + 1][j] = temp_row[j]


# Bubble sorting by col, with decreasing number of 1's
def sort_by_col(mat):
    n = len(mat)
    trans_arr = make_transposed_matrix(mat)
    sort_by_row(trans_arr)
    for i in range(0, n):
        for j in range(0, n):
            mat[i][j] = trans_arr[j][i]


# Returns a transpose of a matrix
def make_transposed_matrix(mat):
    n = len(mat)
    transposed_mat = []
    for j in range(0, n):
        transposed_mat.append([])
        for i in range(0, n):
            transposed_mat[j].append(mat[i][j])
    return transposed_mat


# Counts the number of free (all-zero) rows in a matrix.
def num_free_rows(mat):
    n = len(mat)
    count = 0
    for i in range(0, n):
        for j in range(0, n):
            if mat[i][j] == 1:
                break
        else:
            count += 1
    return count


# Counts the number of free (all-zero) columns in a matrix.
def num_free_cols(mat):
    n = len(mat)
    count = 0
    for j in range(0, n):
        for i in range(0, n):
            if mat[i][j] == 1:
                break
        else:
            count += 1
    return count


# Not used. Old function to help put matrix in canonical form. This doesn't work--repeated sorting by row and column
# doesn't stabilize.
def old_simplify(mat):
    old_minimize_total(mat)
    for s in range(0, len(mat)):
        sort_by_col(mat)
        sort_by_row(mat)


# Modifies an matrix so that each member of an equivalence class is transformed to a single canonical representative
def simplify(mat):
    n = len(mat)
    highest_key = 0      # Keeps track of the key integer associated with the highest score, to determine row/col order
    current_best = copy(mat)
    for inversions in find_best_inversions(mat):
        inversion_arr = perform_inversions(mat, inversions[0], inversions[1])
        sort_by_row(inversion_arr)
        sort_by_col(inversion_arr)
        for perm in tie_break_permutations(list_row_ties(inversion_arr), n):
            row_switch_arr = row_reorder(inversion_arr, perm)
            for perm2 in tie_break_permutations(list_col_ties(inversion_arr), n):
                col_switch_arr = col_reorder(row_switch_arr, perm2)
                new_key = index(col_switch_arr)
                if new_key > highest_key:
                    highest_key = new_key
                    current_best = copy(col_switch_arr)
    return current_best


# Interprets each matrix as a binary number to give it an integer key
def index(mat):
    binary_string = ""
    for row in mat:
        for entry in row:
            binary_string += str(entry)
    return int(binary_string, 2)


def copy(mat):
    n = len(mat)
    new_arr = []
    for i in range(0, n):
        new_arr.append([])
        for j in range(0, n):
            new_arr[i].append(mat[i][j])
    return new_arr


# Pads a matrix with zeros on the right and the bottom.
def expand(mat):
    n = len(mat)
    new_arr = []
    for i in range(0, n):
        new_arr.append([])
        for j in range(0, n):
            new_arr[i].append(mat[i][j])
        new_arr[i].append(0)
    new_arr.append([0] * (n + 1))
    return new_arr


# Given a matrix and a row-reordering scheme, returns the matrix with rows reordered. For example, reordering with
# [0, 2, 1, 5, 3, 4] would turn the rows ABCDEF into ACBEFD.
def row_reorder(mat, reordering):
    n = len(mat)
    new_mat = []
    for i in range(0, n):
        new_mat.append([])
    for i in range(0, n):
        for j in range(0, n):
            new_mat[reordering[i]].append(mat[i][j])
    return new_mat


# Given a matrix and a col-reordering scheme, returns the matrix with cols reordered.
def col_reorder(mat, reordering):
    n = len(mat)
    new_arr = []
    for i in range(0, n):
        new_arr.append(n * [0])
        for j in range(0, n):
            new_arr[i][reordering[j]] = mat[i][j]
    return new_arr


# Returns list of all permutations of input list seq
def permutations(seq):
    perms = []
    for i in range(0, len(seq)):
        leftover = seq[:i] + seq[i + 1:]
        if len(leftover) == 0:
            perms.append([seq[i]])
        for perm in permutations(leftover):
            perms.append([seq[i]] + perm)
    return perms


# Old version--DOESN'T WORK!
# Given a list of tied indices (indices that cannot be simply ranked, meaning, rows or columns with the same number of
# 1s, returns all permutations to try to compare the ties. Requires that tied indices be in consecutive chunks.
# For example, when passed [[0,1],[3,4]], 6, returns [[0,1,2,3,4,5], [1,0,2,3,4,5], [0,1,2,4,3,5], [1,0,2,4,3,5]].
def broken_tie_break_permutations(ties_list, n):
    perms = []
    num_perms = 1

    for tie in ties_list:
        num_perms *= factorial(len(tie))

    for i in range(0, num_perms):
        perms.append([x for x in range(0, n)])

    for tie in ties_list:
        initial_index = tie[0]
        tie_length = len(tie)
        tie_perms = permutations(tie)
        tie_perms_length = len(tie_perms)
        for i in range(0, num_perms):
            for j in range(0, tie_length):
                perms[i][initial_index + j] = tie_perms[i % tie_perms_length][j]

    return perms


# Given a list of tied indices (indices that cannot be simply ranked, meaning, rows or columns with the same number of
# 1s, returns all permutations to try to compare the ties. Requires that tied indices be in consecutive chunks.
# For example, when passed [[0,1],[3,4]], 6, returns [[0,1,2,3,4,5], [1,0,2,3,4,5], [0,1,2,4,3,5], [1,0,2,4,3,5]].
def tie_break_permutations(ties_list, n):
    perms = [[x for x in range(0, n)]]
    perm_count = 1

    for tie in ties_list:
        initial_index = tie[0]
        tie_length = len(tie)
        tie_perms = permutations(tie)
        tie_perms_length = len(tie_perms)
        for i in range(1, tie_perms_length):
            for j in range(0, perm_count):
                new_perm = perms[j][:]
                for t in range(0, tie_length):
                    new_perm[initial_index + t] = tie_perms[i][t]
                perms.append(new_perm)
        perm_count = len(perms)

    return perms


# Given a sorted matrix (sorted by number of 1's per row),returns a list of row ties.
def list_row_ties(mat):
    n = len(mat)
    sums = []
    for i in range(0, n):
        sums.append(0)
        for j in range(0, n):
            sums[i] += mat[i][j]

    ties = []
    current_tie = [0]
    for i in range(1, n):
        if sums[i] == sums[i - 1]:
            current_tie.append(i)
        else:
            if len(current_tie) > 1:
                is_valid = False                    # Only want to include tied sets that are not completely identical
                for t in range(1, len(current_tie)):
                    if mat[current_tie[t]] != mat[current_tie[0]]:
                        is_valid = True
                        break
                if is_valid:
                    ties.append(current_tie)
            current_tie = [i]
    if len(current_tie) > 1:
        is_valid = False
        for t in range(1, len(current_tie)):
            if mat[current_tie[t]] != mat[current_tie[0]]:
                is_valid = True
                break
        if is_valid:
            ties.append(current_tie)
    return ties


# Given a sorted matrix (sorted by number of 1's per row),returns a list of col ties.
def list_col_ties(mat):
    n = len(mat)
    trans_arr = make_transposed_matrix(mat)
    sums = []
    for i in range(0, n):
        sums.append(0)
        for j in range(0, n):
            sums[i] += trans_arr[i][j]

    ties = []
    current_tie = [0]
    for i in range(1, n):
        if sums[i] == sums[i - 1]:
            current_tie.append(i)
        else:
            if len(current_tie) > 1:
                is_valid = False                     # Only want to include tied sets that are not completely identical
                for t in range(1, len(current_tie)):
                    if trans_arr[current_tie[t]] != trans_arr[current_tie[0]]:
                        is_valid = True
                        break
                if is_valid:
                    ties.append(current_tie)
            current_tie = [i]
    if len(current_tie) > 1:
        is_valid = False
        for t in range(1, len(current_tie)):
            if trans_arr[current_tie[t]] != trans_arr[current_tie[0]]:
                is_valid = True
                break
        if is_valid:
            ties.append(current_tie)
    return ties


# Return new matrix with row and column inversions on a matrix as determined by the row and col strings
def perform_inversions(mat, row_str, col_str):
    inversion_mat = copy(mat)
    n = len(mat)
    for s in range(0, n):
        if row_str[s] == "1":
            invert_row(inversion_mat, s)
    for t in range(0, n):
        if col_str[t] == "1":
            invert_col(inversion_mat, t)
    return inversion_mat


# Carry out row inversions on a matrix as determined by the row string
def perform_row_inversions(mat, row_str):
    n = len(mat)
    for s in range(0, n):
        if row_str[s] == "1":
            invert_row(mat, s)


# Carry out column inversions on a matrix as determined by the col string
def perform_col_inversions(mat, col_str):
    n = len(mat)
    for t in range(0, n):
        if col_str[t] == "1":
            invert_col(mat, t)


# Given a matrix, returns a list of all the inversion combinations that result in a min value for total_score function
def find_best_inversions(mat):
    n = len(mat)
    best_inversions = []
    best_total = n ** 2

    for i in range(0, 2 ** n):
        bin_i = bin(i + 2 ** n)[3:]
        perform_row_inversions(mat, bin_i)

        for j in range(0, 2 ** (n - 1)):        # Avoids ever inverting first column, to prevent redundant inversions
            bin_j = bin(j + 2 ** n)[3:]
            perform_col_inversions(mat, bin_j)

            new_total = total_score(mat)

            if new_total < best_total:
                best_total = new_total
                best_inversions = [[bin_i, bin_j]]

            elif new_total == best_total:
                best_inversions.append([bin_i, bin_j])

            perform_col_inversions(mat, bin_j)

        perform_row_inversions(mat, bin_i)

    return best_inversions


# Counts the number of 1's in a string
def one_counter(string):
    counter = 0
    for char in string:
        if char == '1':
            counter += 1
    return counter


# Converts a set of binary strings to a matrix, adding a row and column of zeros on the bottom and right.
def bin_to_matrix(bin_list):
    n = len(bin_list)
    mat = []
    for i in range(0, n):
        mat.append([])
        for j in range(0, n):
            mat[i].append(int(bin_list[i][j]))
        mat[i].append(0)
    mat.append([0] * (n + 1))
    return mat


# Main function. Produces a set of n by n matrices that are distinct under row/column permutation and bitflipping. Also
# keeps track of the subset of these that are unique with the additional operation of transposition.
# Does this by crawling through all possible matrices, imposing without loss of generality the condition that the first
# row begin with a solid block of 1s (always achievable by column permutation) and each subsequent row contain no more
# 1's than the row directly above it.
# Each matrix is put in canonical form (minimum number of 1's, max number of free rows, max number of free columns,
# 1's pushed as much as possible to the top and left) and printed along with its binary key.
# Works quickly for n <= 5, takes <1 hour for n=6, and can't do n=7 in a reasonable amount of time.
def generate_matrices(n, verbose=False):

    time0 = time.time()
    bins_to_check = []

    for i in range(0, n):
        first_row = ['0'] * (n - 1)
        for x in range(0, i):
            first_row[x] = '1'
        first_row_str = ''.join(first_row)

        for j in range(0, 2 ** ((n - 1) * (n - 2))):
            previous_row_count = i
            is_valid = True
            bin_j = bin(j + 2 ** ((n - 1) * (n - 2)))[3:]
            new_arr = [first_row_str]
            for k in range(0, n - 2):
                new_row = bin_j[k * (n - 1):(k + 1) * (n - 1)]
                new_row_count = one_counter(new_row)
                if new_row_count <= previous_row_count:
                    new_arr.append(new_row)
                    previous_row_count = new_row_count
                else:
                    is_valid = False
                    break
            if is_valid:
                bins_to_check.append(new_arr)

    if verbose:
        print("Finished setting up %s matrices..." % len(bins_to_check))
        print("Elapsed time: %.2f s" % (time.time() - time0))

    array_dict = {}

    symmetry_counter = 0

    for i in range(0, len(bins_to_check)):
        if verbose and i % 1000 == 0:
            print("Finished with %s of %s. Current count is %s." % (i, len(bins_to_check), len(array_dict)))
            print("Elapsed time: %.2f s" % (time.time() - time0))
        simplified_array = simplify(bin_to_matrix(bins_to_check[i]))

        array_index = index(simplified_array)
        if array_index not in array_dict:
            transposed_array = simplify(make_transposed_matrix(simplified_array))
            transpose_array_index = index(transposed_array)
            if array_index == transpose_array_index:
                # This is where one could flag matrices that are redundant once transposes are included.
                # Tricky point: Some matrices can become their own transpose through only row/col bitflips and perms.
                symmetry_counter += 1
            array_dict[array_index] = copy(simplified_array)

    for key in array_dict:
        print()
        print("ID %s:" % key)
        print_matrix(array_dict[key])
        print()

    print("Total number of distinct size %s arrays found: %s" % (n, len(array_dict)))
    print("Removing transpose symmetry, total is: %s" % ((len(array_dict) + symmetry_counter) // 2))