__author__ = 'raphey'

import time
from fractions import gcd
from math import factorial
from itertools import chain, combinations


# Prints a matrix
def print_array(mat):
    print()
    for row in mat:
        printing_row = []
        for entry in row:
            printing_row.append(str(entry))
        print('\t'.join(printing_row))


# Recursive function that returns a list of all possible distinct partitions of n, with each partition as a list.
# Partitions are sorted largest to smallest. List of all partitions is sorted similarly, from [n] to [1, 1, ... 1]
# Optional second argument for the maximum chunk size is only used for the recursive calls; otherwise it defaults to n
def partitions(n, max_size=None):
    if n == 0:
        return [[]]
    if not max_size:
        max_size = n
    partition_list = []
    for i in range(max_size, 0, -1):
        sub_partitions = partitions(n - i, min(i, n - i))
        for sub_part in sub_partitions:
            partition_list.append([i] + sub_part)
    return partition_list


# For each partition of n, the "assignment map" is a list of length n that, for each list item, gives a tuple with
# a) the index of the partition chunk it would fall into. For example, a partition 8 into [3, 2, 2, 1] would have an
# assignment map with [0, 0, 0, 1, 1, 2, 2, 3, 4] for the first values of the tuples.
# b) how far into its particular partition chunk it is. The assignment map for the partition of 8 into [3, 2, 2, 1]
# would have [0, 1, 2, 0, 1, 0, 1, 0, 0] for the second values of the tuples
def partition_assignment_maps(p_list):
    assignment_maps = []
    for p in p_list:
        new_map = []
        assignment_index = 0
        for s in p:
            next_section = []
            for i in range(0, s):
                next_section.append((assignment_index, i))
            new_map += next_section
            assignment_index += 1
        assignment_maps.append(new_map)
    return assignment_maps


# For a given partition p of n, returns number of times it will occur as a cycle set for all permutations of n objects
def partition_weight(p):
    n = 0
    counter_by_size = {}
    for chunk in p:
        n += chunk
        if chunk not in counter_by_size:
            counter_by_size[chunk] = 1
        else:
            counter_by_size[chunk] += 1

    weight = factorial(n)

    for i in counter_by_size:
        j = counter_by_size[i]
        weight //= (factorial(j) * (i ** j))

    return weight


# For a list of unique items, returns all subsets (including empty set) with an even number of elements
def even_combos(l):
    all_combos = chain.from_iterable(combinations(l, r) for r in range(0, len(l)+1))
    evens_only = []
    for combo in all_combos:
        if len(combo) % 2 == 0:
            evens_only.append(combo)
    return evens_only


# For a list of bucket affectors and one particular bucket, finds the number of valid inversion sets using rows/cols
# that affect that bucket. For example, with this cell-to-bucket mapping: [[0, 1, 2], [1, 0, 2], [3, 3, 4]], there are
# two rows and two columns that affect bucket 0--r1, r2, c1, and c2. The combination r1 and r2 (implying not c1 or c2)
# is valid, since it can be part of set of inversions that is valid for all buckets, like {r1 r2} or {r1 r2 r3 c3}.
# However, the combination r1 and c1 (implying not r2 or c2) is invalid, since it leads to problems with bucket 2.
def bucket_multiplier(bucket_affectors, bucket_index):

    n = len(bucket_affectors[0])
    list_of_affectors = []
    multiplier_count = 1

    for i in range(0, n):
        if bucket_affectors[bucket_index][i]:
            list_of_affectors.append(i)
    combos_to_try = even_combos(list_of_affectors)      # We only want even-sized sets of rows/cols to invert

    for combo in combos_to_try[1:]:     # Ignore item 0, changing nothing--always works, so mult count started at 1.
        new_bucket_bits = len(bucket_affectors) * [0]
        new_bucket_affectors = []
        for k in range(0, len(bucket_affectors)):
            new_bucket_affectors.append(bucket_affectors[k][:])     # Mostly same as old list, so start with a copy
            for a in combo:
                if new_bucket_affectors[k][a]:          # But if it is affected by the ones we're flipping
                    new_bucket_bits[k] += 1             # Increment its bit
            for l in range(0, n):
                if bucket_affectors[bucket_index][l]:   # This affector appeared in the bucket we're focusing on
                    new_bucket_affectors[k][l] = False  # So moving forward, remove these degrees of freedom
        if True:                    # In the harder version, with row inversion, this only happens if there is a valid
            multiplier_count += 1   # set of column/row inversions from here on, which is checked with a function.

    for bucket in bucket_affectors:                     # After adjusting the multiplier count, all of these
        for affector in list_of_affectors:              # affectors are set to False.
            bucket[affector] = False

    return multiplier_count


# Returns the number of col inversion combinations that allow the state of all buckets to be preserved. The only way
# a bucket's state is *not* preserved is if a set of col inversions affects an odd number of cells in a bucket.
# Thus, this function returns the number of row/col inversion combinations that affect an even number of cells in
# every bucket.
# See related function in matrix_counter.py for more details on how this function works.
def valid_col_inversion_counter(bucket_list, n):
    cols_free = n * [True]
    extra_freedom_counter = n
    bucket_affectors = []
    for i in range(0, len(bucket_list)):
        bucket_affectors.append([False] * n)
        col_effect_counts = n * [0]
        for cell in bucket_list[i]:
            col_effect_counts[cell[1]] += 1
        for j in range(0, n):
            if col_effect_counts[j] % 2 == 1:
                bucket_affectors[i][j] = True
                if cols_free[j]:
                    cols_free[j] = False
                    extra_freedom_counter -= 1

    overall_count = (2 ** extra_freedom_counter)

    for k in range(0, len(bucket_list)):
        overall_count *= bucket_multiplier(bucket_affectors, k)

    return overall_count


# Main function. This uses the Polya enumeration theorem to count the number of binary n by n matrices that are distinct
# under permutation of rows, permutation of columns, and inversion of columns. Confirmed with http://oeis.org/A006383.
def enumerate_matrices_col_inv_only(n, verbose=False):
    time0 = time.time()

    partitions_list = partitions(n)
    partition_maps = partition_assignment_maps(partitions_list)
    partition_weights = [partition_weight(x) for x in partitions_list]

    grand_total = 0

    for x in range(0, len(partitions_list)):
        if verbose:
            print("%s of %s x-partitions complete in %.2f s" % (x, len(partitions_list), (time.time() - time0)))
        p = partitions_list[x]
        p_map = partition_maps[x]
        for y in range(0, len(partitions_list)):
            q = partitions_list[y]
            q_map = partition_maps[y]

            # At this point in the loop, we're considering one particular 2-D partition of the matrix.
            # Each partition defines a sub-cycle--for example, if the row partitions start with the first three rows
            # grouped together, that could be one of two 3-cycles of those rows. These will be generalized to represent
            # *every* possible permutation using the partition-weights function at the end of this loop.

            # This begins the process for assigning each cell to a particular bucket, explained further down.

            cell_to_bucket_map = [z[:] for z in [[None] * n] * n]

            buckets = []
            bucket_dict = {}
            bucket_count = 0

            for i in range(0, n):
                p_index = p_map[i][0]
                p_index_2 = p_map[i][1]
                p_size = p[p_index]
                for j in range(0, n):
                    q_index = q_map[j][0]
                    q_index_2 = q_map[j][1]
                    q_size = q[q_index]
                    c_index = (p_index_2 - q_index_2) % gcd(p_size, q_size)
                    if str((p_index, q_index, c_index)) not in bucket_dict:
                        bucket_dict[str((p_index, q_index, c_index))] = bucket_count
                        bucket_count += 1
                        buckets.append([(i, j)])
                    else:
                        buckets[bucket_dict[str((p_index, q_index, c_index))]].append((i, j))
                    cell_to_bucket_map[i][j] = bucket_dict[str((p_index, q_index, c_index))]

            # At this point, each cell has been assigned to a particular "bucket" which it shares with all other
            # cells whose positions it could come to occupy given the combination of row and column cycles. For
            # example, a cycle of 2 rows and 6 columns has 2 buckets, whereas a cycle of 3 rows and 5 columns has
            # only one bucket. The number of buckets is (I think?) the gcd of the two cycle sizes.

            valid_inversion_count = valid_col_inversion_counter(buckets, n)
            # This is a tricky subroutine to figure out, for a given set of buckets, how many sets of inversions
            # are valid, meaning they don't make it impossible for the contents of the bucket to be preserved under
            # the group operation.

            grand_total += partition_weights[x] * partition_weights[y] * valid_inversion_count * (2 ** len(buckets))
            # The first two terms take us from a particular row/col partition to the number of permutations.
            # The third term tells us how many col inversions are legit
            # The last term corresponds to the fact that, for each bucket, we can pick a single cell and assign it
            # either a 1 or a 0, and that determines the content of every other cell in the bucket. So one degree
            # of freedom per bucket.

    final_count = grand_total // ((factorial(n) ** 2) * (2 ** n))             # Denominator is the size of the group
    print("\nTotal number of distinct %s by %s binary matrices: %s" % (n, n, final_count))
    print("(Distinct under row/col permutation and col inversion)")
    print("\nElapsed time: %.2f s" % (time.time() - time0))


# N=5, 299, in 0.06 s
# N=6, 6128, in 0.15 s
# N=7, 8102, in 0.28 s
# N=8, 67013431, in 0.31 s
# N=9, 45770163273 in 0.4 s
# N=10, 108577103160005 in 0.6 s
# N=11, 886929528971819040 in 1.5 s
# N=12, 24943191706060101926577 in 3.9 s
# N=13, 2425246700258693990625775794 in 9.2 s
# N=14, 820270898724825121532156178527106 in 23 s
# N=15, 971629589818098915847153553138323244260 in 64 s
# N=16, 4058391063608578447081222101317429418552687581 in 2.8 m
