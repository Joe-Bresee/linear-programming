# CSC 445 Assignment 1
**University of Victoria Summer 2026** *Author: Joseph Bresee V01005288*

---

## 1. Introduction

The conjecture under investigation is:

> *"In terms of performance, the Simplex Method is not a competitive option for solving LPs in practice."*

Dantzig first introduced the Simplex method while working for the U.S. Air Force in 1947. Since then, the algorithm has been extensively studied both practically and theoretically. Klee and Minty proved in 1972 that Simplex has exponential worst-case complexity on certain adversarial instances [1], Karmarkar's 1984 interior-point algorithm opened a polynomial-time alternative [2], and Spielman-Teng's 2004 Smoothed Analysis [3] proved Simplex polynomial time in practice.

To evaluate the conjecture I used a hybrid approach: a review of key theoretical results from the LP literature (Klee-Minty, Karmarkar, and Spielman-Teng) combined with my own empirical benchmarking across three solver backends. The theoretical component establishes what worst-case and smoothed-case analysis predict; the experimental component tests whether those predictions hold on my own practical instances. The benchmarks compare SciPy's revised-simplex implementation, the HiGHS simplex backend, and the HiGHS interior-point method (IPM) on a resource-allocation scheduling LP scaled from 10 to 1,000 jobs. I chose this LP structure because resource-allocation scheduling is representative of real-world optimization workloads in cloud computing, energy systems, and network traffic management. Code and data are available at [https://github.com/Joe-Bresee/linear-programming/tree/main/a1](https://github.com/Joe-Bresee/linear-programming/tree/main/a1). The hybrid analysis led me to refute the conjecture: What actually determines practical LP solver performance is not the algorithm itself (Simplex vs. IPM), but the engineering of its implementation.

## 2. Theoretical Context

### 2.1 Worst-Case Complexity: Klee-Minty
Klee and Minty (1972) constructed a family of LP instances in $d$ dimensions on which the standard Simplex method (using Dantzig's largest-coefficient pivot rule) visits all $2^{d}$ vertices before reaching the optimum [1]. Their custom polytope, dubbed the *Klee-Minty cube*, demonstrated that Simplex is not polynomial in the worst case. The pivotal point of the work was proving that a deliberately adversarial object can be crafted to exploit exact numerical alignment of the polytope's faces, forcing $O(2^{d})$ pivot steps. It is important to note that this polytope basically never occurs in the real-world.

### 2.2 Interior-Point Methods: Karmarkar
Karmarkar (1984) introduced the first practical polynomial-time LP algorithm, requiring $O(n^{3.5}L)$ arithmetic operations (which before Karmarkar was deemed computationally infeasible). Rather than traversing the boundary of the feasible polytope, Karmarkar's method moves through the interior via a sequence of projective transformations. This algorithm is used in my benchmark test as the HiGHS IPM backend, standing as a non-Simplex, performative, competitive option for solving LPs.

### 2.3 Smoothed Analysis: Spielman and Teng
Spielman and Teng (2004) resolved the apparent paradox between Klee-Minty's exponential worst case and Simplex's observed polynomial behaviour in practice [3]. Their framework of smoothed analysis perturbs an input with small Gaussian noise of variance $\sigma^{2}$ and bounds the expected running time over this. Their result is that the expected number of pivot steps on any perturbed instance is bounded by a polynomial, demonstrating an expected polynomial time.

The insight gathered from the Klee-Minty cube and smoothed analysis is that while the Simplex method has a technically worst-case exponential time, this only occurs in very specific polytope shapes. This in combination with common real-world factors like floating-point rounding and measurement errors causes Klee-Minty cube style polytopes to be highly unlikely in practice, leading to consistent polynomial behaviour from Simplex empirically [3].

### 2.4 Algorithmic Engineering: Huangfu and Hall
Modern industrial-grade Simplex solvers differ dramatically from the textbook description. Huangfu and Hall (2018) describe the design of the HiGHS dual revised simplex solver, which incorporates parallelism across multiple pivots (the PAMI strategy) and other improvements [4]. Their computational results show HiGHS simplex outperforming earlier leading open-source simplex implementations and achieving competitive performance with IPM on standard LP benchmark sets. This engineering gap - between a simple implementation and a production solver - is imperative to understanding my personal experimental results below.

## 3. LP Model and Problem Formulation

The benchmark is a resource-allocation scheduling LP with $n$ jobs and $m=10$ resources. For job $i \in \{1,\dots,n\}$ and resource $r \in \{1,\dots,m\}$, the decision variable $x_{i,r} \in [0,1]$ is the fraction of resource $r$ allocated to job $i$.

The LP is formulated as follows:

$$
\begin{aligned}
& \text{Maximize} & & \sum_{i=1}^{n} \sum_{r=1}^{m} c_{i,r} x_{i,r} \\
& \text{Subject to} & & \sum_{i=1}^{n} x_{i,r} = cap_{r}, \quad && \forall r \in \{1, \dots, m\} \quad && \text{(resource capacity)} \\
& & & \sum_{r=1}^{m} x_{i,r} = dem_{i}, \quad && \forall i \in \{1, \dots, n\} \quad && \text{(job demand)} \\
& & & 0 \le x_{i,r} \le 1, \quad && \forall i, r
\end{aligned}
$$

With $n=500$ and $m=10$, the LP has 5000 decision variables and $n+m=510$ equality constraints. Instances are generated by drawing resource capacities and job demands uniformly at random and rescaling to ensure feasibility. Five independent instances are generated per size; reported metrics are median solve times across trials. Please note that revised-simplex is known to have a weakness with a large number of equality constraints, which this problem has. This is expanded upon in Section 6.

## 4. Experimental Setup

Instances were generated using `generate_instances.py` at sizes $n \in \{10, 50, 100\}$ for the revised simplex algorithm and $n \in \{10, 50, 100, 500, 1000\}$ for HiGHS Simplex and HiGHS IPM and solved via `scipy.optimize.linprog` with three backends:

* **SciPy Revised Simplex:** a legacy revised simplex
* **HiGHS Simplex (`highs-ds`):** dual simplex with performance optimizations based on Huangfu and Hall [4].
* **HiGHS IPM (`highs-ipm`):** interior-point method with performance optimizations based on Karmarkar [2].

For each (size, backend) pair, five independently generated instances were solved and the median solve time recorded. The revised-simplex at n=500 and n=1000 was not completed due to machine limitations. All supplementary code and raw results data are available at: [https://github.com/Joe-Bresee/linear-programming/tree/main/a1](https://github.com/Joe-Bresee/linear-programming/tree/main/a1).

## 5. Results

### 5.1 Summary Table

*Table 1: Median solve times (seconds) by method and problem size. N/A = run did not complete on test machine.*

| $n_{\text{jobs}}$ | SciPy Rev. Simplex (s) | HiGHS Simplex (s) | HiGHS IPM (s) | Variables |
|---:|---:|---:|---:|---:|
| 10 | 0.014 | 0.00179 | 0.00161 | 100 |
| 50 | 45.1 | 0.00330 | 0.00348 | 500 |
| 100 | 92.5 | 0.00553 | 0.00557 | 1,000 |
| 500 | N/A | 0.0377 | 0.0350 | 5,000 |
| 1000 | N/A | 0.156 | 0.156 | 10,000 |

### 5.2 Solve Time vs. Problem Size

![Median solve time vs n_jobs](a1/scheduling-lp-benchmark/plots/time_vs_njobs_combined.png)
*Figure 1: Median solve time (log scale) vs. number of jobs for all three solver backends, each size measured over 5 independently seeded instances. The revised-simplex curve ends at $n=100$ due to machine limits. HiGHS simplex and HiGHS IPM are nearly the same across all tested sizes, reflecting the importance of HiGHS.*

### 5.3 Key Observations

* **SciPy revised simplex degrades a lot.** From 0.014 s at $n=10$ it reaches 45.1 s at $n=50$ and 92.5 s at $n=100$ - a roughly $6,600\times$ increase in time for a $10\times$ increase in problem size, unusable practically.
* **HiGHS simplex and HiGHS IPM are nearly identical.** Both scale from $\sim$0.002 s at $n=10$ to $\sim$0.156 s at $n=1000$. This shows the importance of the two HiGHS's implementation optimizations over their core algorithms.
* **The engineering gap is the dominant result.** At $n=50$ the ratio between SciPy revised-simplex and HiGHS simplex is 45.1 s / 0.003 s $\approx$ 15,000$\times$. At $n=100$ it is 92.5 s / 0.006 s $\approx$ 16,700$\times$. Both backends are simplex algorithms; the difference is entirely implementation.

## 6. Discussion

### 6.1 The Engineering Gap
The most noticeable result is just how slow the revised-simplex was. At $n=50$: 45.1 s vs. 0.003 s, a factor of roughly $15,000\times$ between it and its HiGHS implementation. At $n=100$: 92.5 s vs. 0.006 s is a factor of $\sim 16,700\times$. This gap is not evidence that the Simplex method is uncompetitive; it is evidence that the revised-simplex backend version is.

SciPy's revised-simplex is documented to be slow on problems with large numbers of equality constraints because it does not use the same optimization strategies as the two HiGHS do. Therefore the revised-simplex implementation should not be looked at as showing the simplex algorithm is weak, but as showing that unoptimized implementation of the simplex algorithm is weak.

### 6.2 HiGHS Simplex vs. HiGHS IPM
Another finding I had is that HiGHS simplex and HiGHS IPM are almost indistinguishable on this problem across all (machine-allowing) tested sizes, with neither method consistently faster than the other. At $n=1000$ both reach $\sim$0.156 s. Two different algorithms, a vertex-follower and an interior-point method, converge to almost the same performance on randomly generated scheduling instance problems. This is consistent with Spielman and Teng's smoothed analysis [3]: on practical instances with natural noise, meaning the worst-case runtime of Simplex is no longer important, and it is replaced by its smoothed worst-case. The shared HiGHS implementation of both causes their runtimes to converge (under $n=1000$).

### 6.3 How Results Compare with Theory
The HiGHS simplex scaling results reflect Spielman and Teng's smoothed analysis prediction [3]. The Simplex method here behaves polynomially because the inequality is (very likely) not an adversarial polytope. The plot evidently shows no exponential increase for the HiGHS simplex curve.

### 6.4 Limitations
This experiment uses a single LP structure (equality-constrained scheduling). Results may differ on inequality-dominated LPs. The SciPy comparison deliberately uses an unoptimized baseline to illustrate the implementation gap; citing it as representative of the Simplex class would be a category error. Machine limits prevented SciPy revised-simplex from reaching $n=500$. Machine limits prevented runs on large $n=3000+$ to test HiGHS Simplex vs. HiGHS IPM at larger scale.

## 7. Conclusion

The experimental and theoretical evidence examined here refutes the conjecture as stated. The claim that "the Simplex Method is not a competitive option for solving LPs in practice" is refuted when the distinction between algorithmic class implementation is made.

A more basic, unoptimized simplex method is uncompetitive. The data showed a far slower simplex solver from an unoptimized solver. But, a well-engineered simplex (HiGHS) runs polynomially, matching HiGHS IPM (at least at problem sizes $\le 1000$). Spielman and Teng's smoothed analysis [3] explains theoretically why Klee-Minty's exponential worst case [1] does not materialize on practical instances, and Huangfu and Hall's engineering work [4] demonstrates what a production-quality simplex implementation actually looks like compared to a true, practical polynomial-time LP solver, IPM, first introduced by Karmarkar [2].

What truly determines practical LP performance is not the choice between Simplex and interior-point at the algorithm level, but the engineering of the actual implementation. A well-engineered simplex proves to be competitive with polynomial solvers.

---

## References

1. V. Klee and G. J. Minty, "How good is the simplex algorithm?" in *Inequalities III*, O. Shisha, Ed. New York: Academic Press, 1972, pp. 159-175. [Online]. Available: [http://cgm.cs.mcgill.ca/~avis/Kyoto/courses/te/abs_examples.pdf](http://cgm.cs.mcgill.ca/~avis/Kyoto/courses/te/abs_examples.pdf)
2. N. Karmarkar, "A new polynomial-time algorithm for linear programming," *Combinatorica*, vol. 4, no. 4, pp. 373-395, 1984. [Online]. Available: [https://www.stat.uchicago.edu/~lekheng/courses/302/classics/karmarkar.pdf](https://www.stat.uchicago.edu/~lekheng/courses/302/classics/karmarkar.pdf)
3. D. A. Spielman and S.-H. Teng, "Smoothed analysis of algorithms: Why the simplex algorithm usually takes polynomial time," *J. ACM*, vol. 51, no. 3, pp. 385-463, 2004. [Online]. Available: [https://arxiv.org/pdf/math/0212413](https://arxiv.org/pdf/math/0212413)
4. Q. Huangfu and J. A. J. Hall, "Parallelizing the dual revised simplex method," *Math. Program. Comput.*, vol. 10, no. 1, pp. 119-142, 2018. [Online]. Available: [https://arxiv.org/abs/1503.01889](https://arxiv.org/abs/1503.01889)