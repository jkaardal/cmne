# cmne
Maximum Noise Entropy (Logistic Regression) in C

This code is an implementation of Maximum Noise Entropy (http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1001111) in C. This is equivalent to a (up to) second-order logistic regression but in a neural context. The code allows for fitting a low rank matrix for the second-order model, but the fit is non-convex and thus not recommended at this point. 

It is recommended that you use a OpenBLAS (https://github.com/xianyi/OpenBLAS) to speed up the matrix operations. If you do not wish to use OpenBLAS, comment out this line: "extern "C" void openblas_set_num_threads(int num_threads);". There are some other operations that have been parallelized using OpenMP so it is recommended that you also compile with OpenMP to decrease runtime.

For now, the conjugate gradient descent algorithm is a modified version based on Numerical Recipes in C (http://www.nr.com/). In the future, I plan to switch over to the GNU Scientific Library conjugate gradient algorithm.

Most other information about this code can be found in its comments. I will update this in the future to provide more thorough instructions once the code has reached full funcitonality. 
