## Basic NMF
decompose non negative matrix to components and activation with NMF

```math
Y \approx HU \\
Y \in R(m, n) \\
H \in R(m, k) \\
HU \in R(k, n)
```

approximate evaluation:
    
    euclid divergence

parameters:

    Y      :target matrix to decompose
    R      :number of bases to decompose
    n_iter :number fo executing objective function to optimize
    init_H :initial value of H matrix. default value is random matrix
    init_U :initial value of U matrix. default value is random matrix

return:

    Array of
    0:  matrix of H
    1:  matrix of U
    2:  array of cost transition
  
## Semi-Supervised NMF
decompose non negative matrix to components and activation with NMF

```math
Y \approx FG + HU \\
Y \in R(m, n) \\
F \in R(m, x) \\
G \in R(x, n) \\
H \in R(m, k) \\
U \in R(k, n)
```

approximate evaluation:
    
    euclid divergence

parameters:

    Y      :target matrix to decompose
    R      :number of bases to decompose
    n_iter :number fo executing objective function to optimize
    F      :matrix as supervised base components
    init_W :initial value of H matrix. default value is random matrix
    init_H :initial value of U matrix. default value is random matrix

return:

    Array of
    0:  matrix of F
    1:  matrix of G
    2:  matrix of H
    3:  matrix of U
    4:  array of cost transition
