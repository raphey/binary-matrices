### Overview
This matrix enumeration tool was created to assist a colleague with his research on PR boxes and quantum nonlocality. I understand very little about the actual science part, but if you’re curious you can learn more [here](https://en.wikipedia.org/wiki/Quantum_nonlocality).

In the abstract form, the idea is to count and list distinct binary n by n matrices, where two binary matrices are only distinct if they cannot be made identical with a series of transformations. These transformations may include row permutation, column permutation, row inversion (flipping every bit in a row), column inversion, and transposition. For example, if we allow all of the aforementioned transformations, there are only two distinct 2x2 matrices:

``
0 0  </br>
0 0
``   

``1 0``   
``0 0``   

With the same set of transformations, there are three distinct 3x3 matrices:

``0 0 0``  
``0 0 0``  
``0 0 0``  

``1 0 0``  
``0 0 0``  
``0 0 0``  

``1 0 0``  
``0 1 0``  
``0 0 0``  

One of my programs finds all distinct matrices using a mostly brute-force approach, whereas the others simply count the number of distinct matrices by considering the group made up of all possible transformations and using Burnside’s Lemma/Polya Enumeration Theorem. Burnside's Lemma makes it possible to count things with lots of complicating symmetries, e.g. counting the number of distinct 6-bead bracelets that can be made using only red and blue beads. [Here’s](http://www.geometer.org/mathcircles/polya.pdf) a great introduction to the topic.



I’ve still got a lot to learn, so please forgive the repetition of code between the various files.


### Specific files

#### matrix_generator.py

The main function generate_matrices(n) prints all n by n matrices that are distinct under row/col permutation and row/col inversion. The function also keeps track of how many of these are still distinct with transposition allowed. Because of the brute-force approach used, this only works well through n=6. 

#### matrix_counter_col_inv_only.py

The main function enumerate_matrices_col_inv_only(n) counts the number of n by n matrices that are distinct under row/col permutation and column inversion. This has no bearing on quantum non-locality; it was created to test the enumeration method and check the results against [OEIS series A006383](http://oeis.org/A006383). This is not the fastest way to enumerate this quantity--[here's](http://ieeexplore.ieee.org/document/1672242) a cool paper on the subject.


#### matrix_counter.py

The main function enumerate_matrices(n) counts the number of n by n matrices that are distinct under row/col permutation and row/col inversion. I should probably submit this to OEIS. It works pretty quickly (< 5 min) up to N=12.


#### matrix_ounter_w_trans.py

The main function enumerate_matrices_w_trans(n) counts the number of n by n matrices that are distinct under row/col permutation, row/col inversion, and transposition. This program confirms that the 6 by 6 matrices created by matrix_generator.py are correct, but that’s as high as it can go. Because transposition greatly complicates the orbits of the group, it’s no longer sufficient to use weighted partitions as a stand-in for row and column permutations.

