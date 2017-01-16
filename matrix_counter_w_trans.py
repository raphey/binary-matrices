__author__ = 'raphey'
# Python 3

from fractions import gcd
from math import factorial
import time
from itertools import chain, combinations, permutations


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


# Given a list of bucket affectors (for each bucket, a boolean list of whether it is affected an odd number of times
# for each row and column) and a list of bucket bits (incremented, odd or even), returns true if the given scenario
# can somehow work with subsequent assignments, false otherwise.
def hypo_test(bucket_affectors, bucket_bits):
    # print("Calling function with these affectors:")
    # print(bucket_affectors)
    # print("and these bits")
    # print(bucket_bits)
    n = len(bucket_affectors[0]) // 2                               # n is side length of matrix
    for i in range(0, len(bucket_affectors)):                       # i indexes every bucket
        if bucket_bits[i] % 2 == 1:                                 # and for every odd parity bucket
            for j in range(0, 2 * n):                               # it checks every potentially affecting row/col.
                if bucket_affectors[i][j]:                          # If it does affect the bucket, we need to explore.
                    new_bucket_bits = bucket_bits[:]                # So make a copy of the old bucket_bits
                    new_bucket_affectors = []                       # Start a new bucket affectors list
                    for k in range(0, len(bucket_affectors)):
                        new_bucket_affectors.append(bucket_affectors[k][:])     # Mostly same as old list
                        if new_bucket_affectors[k][j]:              # But if it is affected by the one we're flipping
                            new_bucket_bits[k] += 1                 # Increment its bit
                        for l in range(0, 2 * n):
                            if bucket_affectors[i][l]:              # This affector appeared in bucket i...
                                new_bucket_affectors[k][l] = False  # So moving forward, remove these d.o.f.s
                    if hypo_test(new_bucket_affectors, new_bucket_bits):    # Recursively see if what's left works
                        return True                                         # If so, return true
            return False                # Making it here means there's an intractable odd-bit bucket
    return True                         # Making it here means there were no odd_bit buckets


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

    n = len(bucket_affectors[0]) // 2
    list_of_affectors = []
    multiplier_count = 1

    for i in range(0, 2 * n):
        if bucket_affectors[bucket_index][i]:
            list_of_affectors.append(i)
    combos_to_try = even_combos(list_of_affectors)      # We only want even-sized sets of rows/cols to invert

    for combo in combos_to_try[1:]:     # Ignore item 0, changing nothing--always works, so mult count started at 1.
        new_bucket_bits = len(bucket_affectors) * [0]
        new_bucket_affectors = []
        for k in range(0, len(bucket_affectors)):
            new_bucket_affectors.append(bucket_affectors[k][:])     # Mostly same as old list, so start with a copy.
            for a in combo:
                if new_bucket_affectors[k][a]:          # But if it is affected by the ones we're flipping
                    new_bucket_bits[k] += 1             # Increment its bit
            for l in range(0, 2 * n):
                if bucket_affectors[bucket_index][l]:   # This affector appeared in the bucket we're focusing on
                    new_bucket_affectors[k][l] = False  # So moving forward, remove these degrees of freedom
        if hypo_test(new_bucket_affectors, new_bucket_bits):    # See if it works
            multiplier_count += 1                               # If so, increment multiplier count

    for bucket in bucket_affectors:                     # After adjusting the multiplier count, all of these
        for affector in list_of_affectors:              # affectors are set to False.
            bucket[affector] = False

    return multiplier_count


# Returns the number of row/col inversion combinations that allow the state of all buckets to be preserved. The only way
# a bucket's state is *not* preserved is if a set of row/col inversions affects an odd number of cells in a bucket.
# Thus, this function returns the number of row/col inversion combinations that affect an even number of cells in
# every bucket.
def valid_inversion_counter(bucket_list, n):

    rows_cols_free = 2 * n * [True]    # Start with assumption that all rows and columns can be tweaked individually,
    extra_freedom_counter = 2 * n      # which corresponds to 2n degrees of freedom.

    bucket_affectors = []
    # This list will help us track degrees of freedom with regard to inverting rows and columns.
    # Bucket_affectors[i][j] will be a Boolean variable that tells us if 1) bucket i is affected by row/col inversion
    # j (ranging from 0 to 2n, all individual row inversions followed by all individual col inversion), AND 2) we are
    # free to choose that particular row or column to invert or not invert (with cascading consequences that further
    # limit degrees of freedom but which *do not* depend on which option (invert or not invert) we chose.

    for i in range(0, len(bucket_list)):
        bucket_affectors.append([False] * (2 * n))    # First half of this list is for n rows, second half for n cols.
        row_col_effect_counts = (2 * n) * [0]   # Increment these for each time a row/col affects a bucket, use parity
        for cell in bucket_list[i]:
            row_col_effect_counts[cell[0]] += 1         # The row affects this bucket at the location of this cell
            row_col_effect_counts[cell[1] + n] += 1     # The col affects this bucket at the location of this cell
        for j in range(0, 2 * n):                       # Now go through each row/col individually
            if row_col_effect_counts[j] % 2 == 1:
                bucket_affectors[i][j] = True           # Mark if it affects the bucket an odd number of times
                if rows_cols_free[j]:          # If we thought we were free to invert this row/col however we like
                    rows_cols_free[j] = False     # Correct that assumption
                    extra_freedom_counter -= 1    # And decrement the extra degrees of freedom

    overall_count = (2 ** extra_freedom_counter)  # Start our count with the set of rows/cols we can set arbitrarily.
    # These could have been marked "True" in bucket_affectors, since they meet the requirements of affecting buckets
    # and giving us the freedom to invert or not invert, but instead we're marking them False and counting them all at
    # once.

    # For each bucket, index k, see how many additional degrees of freedom we have and narrow subsequent options.
    for k in range(0, len(bucket_list)):
        overall_count *= bucket_multiplier(bucket_affectors, k)     # Note: this function changes bucket_affectors!

    # At this point, every entry [i][j] in bucket_affectors is False--there are no more degrees of freedom.

    return overall_count // 2  # We've double counted; there are two row/col inversion combos per unique inversion state
    # (For example, inverting every column is the same as inverting every row)


# Main function. This uses the Polya enumeration theorem to count the number of binary n by n matrices that are distinct
# under permutation of rows, permutation of columns, inversion of rows, inversion of columns, and transposition.
def enumerate_matrices_w_trans(n, verbose=False):

    grand_total = 0

    time0 = time.time()

    # PHASE ONE is to consider all the elements of the group with no transposition, which is identical to the simpler
    # program written earlier.

    partitions_list = partitions(n)
    partition_maps = partition_assignment_maps(partitions_list)   # Tool to move between cell coord and partition chunk
    partition_weights = [partition_weight(x) for x in partitions_list]      # Prevalence of each partition

    part_symmetry_lookup = {}      # Dict used to avoid duplicating work (irrelevant savings compared to phase 2)

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

            if str((p, q)) in part_symmetry_lookup:
                grand_total += part_symmetry_lookup[str((p, q))]        # Dictionary shortcut
            else:

                # This begins the process for assigning each cell to a particular bucket, explained further down.

                cell_to_bucket_map = [z[:] for z in [[0] * n] * n]

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

                valid_inversion_count = valid_inversion_counter(buckets, n)
                # This is a tricky subroutine to figure out, for a given set of buckets, how many sets of inversions
                # are valid, meaning they don't make it impossible for the contents of the bucket to be preserved under
                # the group operation.

                new_subtotal = partition_weights[x] * partition_weights[y] * valid_inversion_count * (2 ** len(buckets))
                # The first two terms take us from a particular row/col partition to the number of permutations.
                # The third term tells us how many row/col inversions are legit
                # The last term corresponds to the fact that, for each bucket, we can pick a single cell and assign it
                # either a 1 or a 0, and that determines the content of every other cell in the bucket. So one degree
                # of freedom per bucket.

                part_symmetry_lookup[str((q, p))] = new_subtotal  # Enter what we found in symmetry shortcut dictionary

                grand_total += new_subtotal

    # PHASE TWO is to consider the elements of the group that undergo transposition. The problem here is that it is no
    # longer ok to consider only partitions of rows and columns--we need to consider all the permutations.

    permutation_list = [list(y) for y in permutations([x for x in range(0, n)])]

    perm_symmetry_lookup = {}

    for x in range(0, len(permutation_list)):
        row_perm = permutation_list[x]
        if verbose:
            print("%s of %s x-permutations complete in %.2f s" % (x, len(permutation_list), (time.time() - time0)))

        for col_perm in permutation_list:

            if str((row_perm, col_perm)) in perm_symmetry_lookup:
                grand_total += perm_symmetry_lookup[str((row_perm, col_perm))]

            else:
                cell_to_bucket_map = [z[:] for z in [[-1] * n] * n]      # Using a -1 for empty
                buckets = []
                bucket_count = 0

                for i in range(0, n):
                    for j in range(0, n):
                        if cell_to_bucket_map[i][j] == -1:
                            cell_to_bucket_map[i][j] = bucket_count
                            buckets.append([(i, j)])
                            i_prime = col_perm[j]
                            j_prime = row_perm[i]
                            while not (i_prime == i and j_prime == j):
                                cell_to_bucket_map[i_prime][j_prime] = bucket_count
                                buckets[bucket_count].append((i_prime, j_prime))
                                temp = row_perm[i_prime]
                                i_prime = col_perm[j_prime]
                                j_prime = temp
                            bucket_count += 1
                        else:
                            continue

                valid_inversion_count = valid_inversion_counter(buckets, n)

                trans_subtotal = valid_inversion_count * (2 ** len(buckets))

                perm_symmetry_lookup[str((col_perm, row_perm))] = trans_subtotal

                grand_total += trans_subtotal

    final_count = grand_total // ((factorial(n) ** 2) * (2 ** (2 * n - 1)) * 2)
    print("\nTotal number of distinct %s by %s binary matrices: %s" % (n, n, final_count))
    print("(Distinct under row/col permutation, row/col inversion, and transposition)")
    print("\nElapsed time: %.2f s" % (time.time() - time0))


# N=4, 10, in 0.41 s
# N=5, 30, in 2.37 s
# N=6, 242, in 3.7 m
# N=7 would take about 22 hours

enumerate_matrices_w_trans(7, True)