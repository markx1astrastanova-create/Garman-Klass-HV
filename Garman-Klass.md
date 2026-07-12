# The Garman–Klass volatility estimator revisited 

Isaac Meilijson[∗] Tel-Aviv University 

October 23, 2018 

## Abstract 

The Garman–Klass unbiased estimator of the variance per unit time of a zero–drift Brownian Motion B, based on financial data that reports for time windows of equal length the open (OPEN ), minimum (MIN ), maximum (MAX) and close (CLOSE) values, is quadratic in the statistic S1 = (CLOSE − OPEN, OPEN − MIN, MAX − OPEN ). This estimator, with efficiency 7.4 with respect to the classical estimator (CLOSE−OPEN )[2] , is widely believed to be of minimal variance. The current report disproves this belief by exhibiting an unbiased estimator with slightly but strictly higher efficiency 7.7322. The essence of the improvement lies in the proposal that the data should be compressed to the statistic S2 defined on W (t) = B(0)+[B(t)−B(0)]sign[(B(1)−B(0)] as S1 was defined on the Brownian path B(t). The best S2–based quadratic unbiased estimator is presented explicitly. The Cram´er–Rao upper bound for the efficiency of unbiased estimators is 8.471. It corresponds to the large-sample efficiency of Maximum Likelihood estimators. This bound cannot be attained because the distribution is not of exponential type. 

Regression-fitted quadratic functions of S2 (with mean 1) markedly outperform those of S1 when applied to random walks with heavy-tail-distributed increments. Performance is empirically studied in terms of the tail parameter. 

Keywords and phrases: Garman–Klass, Brownian Motion, volatility, estimation. MSC2000: 62F10, 62P05 

## 1 Introduction 

As stressed repeatedly (see Magdon-Ismail & Atiya (2001)), volatility estimators of financial data ought to have as small a variance as possible, because volatilities change over time, so past data have decaying importance. The celebrated Garman–Klass (1980) variance estimator, introduced almost three decades ago, achieves better accuracy in estimating σ[2] than the classical, natural estimator average (CLOSE − OPEN )[2] does in seven times the observation period. This unbiased variance estimator is the minimum-variance unbiased 

> 0Research conducted on a sabbatical visit to Columbia University, 2008 

0 

quadratic function of the spreads c = CLOSE − OPEN, h = MAX − OPEN, l = MIN − OPEN (for close, high, low). These data S1 = (c, h, l) can be compressed without loss of sufficiency. 

A coarser (but incomplete) sufficient statistic. Consider the triple S2 = (C, H, L) where C = |c| , (H, L) = (h, l) if c > 0, while (H, L) = −(l, h) if c < 0. Without loss of relevant information about the variance, the Brownian Motion trajectory {B(t) ; t ∈ (0, 1)} may be replaced by the flipped path {W (t) ; t ∈ (0, 1)}, defined as W (t) = B(0) + [B(t) − B(0)]sign(B(1) − B(0)). That is, the three interval lengths (−L, C, H − C), in fact the further compression (C, min(−L, H −C), max(−L, H −C)), determined by (c, h, l), carry all relevant information contained in (c, h, l) about σ[2] , but do not determine (c, h, l). Although intuitively clear after some thought, sufficiency of (C, min(−L, H − C), max(−L, H − C)) can be formally inferred from Siegmund’s (1985) representation displayed as (14) in the sequel. The Rao–Blackwell theorem (Blackwell (1947), Rao (1946)) claims that under these conditions, for every S1-based unbiased estimator of some arbitrary parameter there is an S2-based unbiased estimator with smaller variance – strictly smaller unless the two coincide. As will be seen, the Garman–Klass estimator is a function of S2, so the RaoBlackwell improvement leaves it invariant. However, the Garman–Klass estimator, best among the quadratic function of S1, is not best possible as a function of S2. Had S2 been a complete minimal sufficient statistic, Garman–Klass and the proposed estimator would have equally been the UMVUE (uniformly minimum variance unbiased estimator) of the parameter. However, C[2] and 2[(H − C)[2] + L[2] ] are different unbiased estimators of σ[2] . Hence, S2 (whether minimal sufficient or not) is not complete. Loose some, win some: we will only conjecture rather than claim optimality of the proposed S2–based quadratic unbiased estimator of σ[2] ; on the other hand, the exchangeability property under which (−L, C, H − C) and (H − C, C, −L) are identically distributed, justifies searching for the best quadratic function of (−L, C, H − C) among those that are linear combinations of four rather than six quadratic terms. 

Four basic quadratic unbiased variance estimators. 

**==> picture [412 x 29] intentionally omitted <==**

The rationale for the somewhat bizarre coefficients is that each of these four terms is an unbiased estimator of σ[2] , with respective variances 

Var(ˆσ1[2][) = 0][.][797943][σ][4][,][Var(ˆ][σ] 2[2][) = 2][σ][4][,][Var(ˆ][σ] 3[2][) = 0][.][504753][σ][4][,][Var(ˆ][σ] 4[2][) = 1][.][004876][σ][4][(2)] 

1 

The proposed variance estimator vis `a vis Garman–Klass. The proposed estimator ˆ σ[2] =[�][4] 1[α][i][σ][ˆ] i[2][assigns][to][these][four][terms][respective][weights] 

**==> picture [370 x 12] intentionally omitted <==**

and achieves variance Var(ˆσ[2] ) = 0.258658σ[4] . The Garman–Klass estimator [3] 

**==> picture [345 x 14] intentionally omitted <==**

happens to pool these four basic estimators too, so the Rao–Blackwell theorem does not ˆ rule out the possibility that it coincides with σ[2] . However, as argued earlier, the two do not agree, and σˆGK[2][=][�][4] 1[β][i][σ][ˆ] i[2][pays][a][price][for][being][quadratic][in][(][c, h, l][).][Its][coefficients] are given by 

**==> picture [330 x 95] intentionally omitted <==**

that achieve Var(ˆσGK[2][) = 0][.][27][σ][4][.] 

Maximum Likelihood variance estimators and Fisher information. In principle, giving up on the requirement of unbiasedness, the computer–intensive maximum likelihood estimator (MLE) of σ[2] by Magdon-Ismail & Atiya (2001) could have been a competitor, since MLE’s are functions of any sufficient statistic. However, this estimator is based on (h, l) rather than on (c, h, l). Magdon-Ismail & Atiya report that their estimator has variance slightly higher than Garman–Klass’. 

The joint generating function of (c, h, l) is presented by Garman & Klass as an infinite series, from which these authors derived all pertinent second and fourth degree moments. 

Ball & Torous (1984) developed an infinite–series formula for the joint density of (c, h, l) and used it to construct numerically the MLE of σ[2] . They report estimated efficiency of the MLE for a selection of sample sizes, basing each value on a simulation sample size of 1000 runs, a great achievement in 1984, but insufficient for delicate comparisons. An attempt at numerical evaluation of the Fisher information, based on the Ball & Torous expression for the joint density, disclosed that their formula seems to have a missprint. This joint density was re-derived based on the formula by Siegmund quoted earlier, exhibited as (14) 

2 

in the sequel. The inverse of the Fisher information is the Cram´er–Rao lower bound for the variance per time–window of any unbiased estimator of σ[2] , for any sample size. It is also the asymptotic variance of the (not necessarily unbiased) MLE of σ[2] . Its value turns out to be 0.2361. This is the benchmark with which Garman–Klass’ 0.27 and the proposed estimate’s 0.258658 variances should be compared. 

The Cram´er–Rao bound 0.2361 is not attained by unbiased variance estimators: disproving exponentiality of a family of distributions. Under proper regularity assumptions (see Joshi (1976) ), the Cram´er–Rao bound is attained if and only if there is a linear relationship between the estimator and the score function (derivative with respect to the parameter of the logarithm of the density). However, for this to happen, there must exist a linear relationship between the score functions evaluated at different values of the parameter. It was ascertained numerically that this is not the case. In other words, the model is not of exponential type. We don’t know whether the sufficient statistic S2, shown above not to be complete, is minimal sufficient. As a result of all of these considerations, the proposed estimator may not be of minimal variance. 

Since both the proposed and Garman–Klass’ estimators are averages over time–windows, their variances per time–window are independent of sample size. It is conceivable, and Ball & Torous have provided evidence in this direction, that the MLE has variance per time– window that decreases as the sample size increases, so for small sample sizes the proposed estimator has in practice no competitor. 

Moreover, since the BM model doesn’t really hold in practice, a broader contribution of this paper is the introduction of more efficient quadratic statistics on which to base practical estimators. Simulation results for random walks with t-distributed increments are reported in Section 3. 

## 2 Derivation 

Following the steps of Garman & Klass, all second and fourth order moments of (C, L, H) will be identified. Some of these will be quoted from Garman & Klass, some will be derived once the joint densities of (C, H) and (C, L) are explicitly presented, and some will require some additional argument. Although it would perhaps be more natural to work only with the exchangeable variables ∆= H − C and δ = −L, work will be performed on the variables H and L as well, in order to link more easily with Garman & Klass’ triple (c, h, l). 

3 

## 2.1 The joint densities of C and each of H and L: four unbiased estimators 

Assume throughout the computations that the drift is 0 and the variance per unit time is 1. Thus, E[C[2] ] = E[c[2] ] = 1. 

By a common reflection argument, BM reaching at least as high as x > 0 and ending up at y = x − (x − y) ∈ (0, x) is tantamount to ending up at x + (x − y). Or, P (H > x, C ∈ [y, y + dy]) = P (C ∈ [2x − y, 2x − y + dy]) = 2φ(2x − y)dy, where · 1 φ( ) = √2π[exp][{−] 2[1][(][·][)][2][}][ is the standard normal density function (see Siegmund or expression] (14) in the Appendix for a generalization to (C, H, L)). 

Similarly, P (L < z, C ∈ [y, y + dy]) = P (C ∈ [2z − y, 2z − y + dy]) = 2φ(2z − y). Hence, the joint density of H and C is 

**==> picture [322 x 12] intentionally omitted <==**

and that of L and C is 

**==> picture [321 x 12] intentionally omitted <==**

These joint densities, essentially re-phrasings of a well known formula for the joint density of (h, h − c) (see Yor (1997)), lead to the first four of the following five second moments. The fifth is taken from Garman & Klass. Details are omitted. E[C[2] ]=1 by assumption. 

**==> picture [401 x 23] intentionally omitted <==**

As a corollary, 

## Lemma 1 The variance estimators σˆi , i = 1, 2, 3, 4 (see (1)) are unbiased. 

Seshadri’s (1988) theorem that 2h(h − c) is exponentially distributed with mean 1, and is independent of c, implies that 2H(H − C) is exponentially distributed with mean 1, and is independent of C. This is so, simply because the conditional distribution of (h, c) given that c > 0 is the (unconditional) distribution of (H, C). 

Of course, the same applies to 2l(l − c) and 2L(L − C). However, 2H(H − C) and 2L(L − C) are dependent (identities (10) yield correlation 1 + 2[7][ζ][(3)][ −][8 log(2)][=][−][0][.][3380] between the two), and dependent given C. 

Otherwise, it would have been very easy to sample (C, H, L) triples. As things stand, it is easy to sample pairs (c, h) (and (c, l)) or (C, H) (and (C, L)), by independently sampling 

4 

c and h(h − c). A practical approximate method to sample (C, H, L) triples is to sample (C[′] , H[′] ) correctly, then make the wrong choice L[′] = C[′] − H[′] , not on [0, 1] but on each of the N sub-intervals [[i][−] N[1][,] N[i][].][The][construction][is][correct][except][if][H][and][L][are][attained] in the same sub-interval, the probability of which decreases fast as N increases. Instead of letting L[′] = C[′] − H[′] , other copulas may be used, to better approximate features of the joint distribution of (C[′] , H[′] , L[′] ). 

## 2.2 The MLE’s of σ[2] based on (C, H) and on (C, L) are unbiased 

It may be of interest to notice that (6) (resp. (7)), reinterpreted as fH,C(x, y; σ) = 4[2][x] σ[−][3][y][φ][(][2][x] σ[−][y] ), identifies the MLE of σ[2] based on (C, H) (resp. (C, L)) as the average over the sample of 3[1][(2][H][−][C][)][2][=][1] 3[C][2][ +] 3[1][[4(][H][−][C][)][2][] +][1] 3[[4][C][(][H][−][C][)]][and][1] 3[(2][L][ −][C][)][2][=] 1[+][1][1][The][average][of][the][two,][the][simple][average][of][the][first][three] 3[C][2] 3[[4][L][2][] +] 3[[][−][4][CL][].] unbiased estimators in (1), achieves variance 0.3694, above Garman–Klass’. 

## 2.3 The fourth moments of (C, H, L) 

The following fourth moments are derived from the joint densities of (H, C) and (L, C). E[C[4] ] = 3 is Gaussian kurtosis. 

**==> picture [375 x 49] intentionally omitted <==**

The following fourth moment information is taken from Garman & Klass. ζ is Riemann’s zeta function, with ζ(3) =[�][∞] k=1 k1[3][≈][1][.][2020569.] 

**==> picture [371 x 93] intentionally omitted <==**

There is one more (C, H, L)-based fourth moment needed, whose value does not follow from Garman & Klass’. 

Lemma 2 E[CHL[2] ] = ζ(3)/16 − 2 log(2) +[47] 32[≈][0][.][1575842][.] 

5 

A proof of Lemma 2 can be found in the Appendix. Large sample empirical estimation of E[CHL[2] ] gave 0.15762, yielding Var(ˆσ4[2][)][very][close][to][1.][Had][E][[][CHL][2][]][been][equal][to] log(2)(3 − 4 log(2)) ≈ 0.15763 (initial conjecture), Var(ˆσ4[2][)][would][have][been][exactly][1.] From all the fourth moments above, 

**==> picture [359 x 311] intentionally omitted <==**

## 2.4 The covariance matrix of the four basic estimators 

Let Σ stand for the covariance matrix of the four basic estimators. Their variances are on the diagonal, their covariances off the diagonal. 

Applying the formulas of the previous sub–section, the variances of the basic estimators σˆi[2][(see][(1))][are] 

**==> picture [404 x 114] intentionally omitted <==**

6 

The covariances of the basic estimators are 

**==> picture [399 x 225] intentionally omitted <==**

## 2.5 Derivation of the proposed estimator 

Letting α (see (3)) stand for the weights assigned to the basic estimators, the weighted sum has variance α[T] Σα and mean α[T] 1. Using a Lagrange multiplier to constrain the mean to be 1, minimal variance is achieved at α = 1[T] ΣΣ[−][−][1] 1[1] 1[,][yielding][the][weights][displayed][in][(3).] The variance of the proposed estimator is 1[T] Σ1[−][1] 1[= 0][.][258658,][with corresponding efficiency] 21[T] Σ[−][1] 1 = 7.73221. 

## 3 Heavy tailed random walks - simulation results 

As is commonly observed in financial data, the logarithmic increments of returns have power-law tails, at least in the visible range, with tail parameter around 3. This means finite variance but infinite variance of the usual empirical variance estimators. Suppose that the basic process on which (Open, Close, Min, Max) data is reported per time window is a random walk with t-distributed increments. A simulation analysis will now be reported, in which the number of increments of the random walk per time window is 10, 30 and 50, and the degrees of freedom (df ) range from 1.5 to 5 with step size 0.5. Minimum sum-of-squares quadratic functions with mean 1 of the S1 and S2 statistics were fitted by Regression, with sample size 10[5] : the regression coefficients were identically calibrated so that the predictor of unity has mean 1 in each such sample. Each such Regression was repeated 100 times, and the averages of the corresponding regression coefficients and overall ”variances” were 

7 

recorded. Of course, second moments are finite only for df > 2 and fourth moments are finite only for df > 4, but the empirical study seems instructive. A sample of size 10[5] from the sum of N = 50 t{df =3}-distributed random variables typically displays lighter tails than df = 3 would entail. Table 1 reports the empirical minimum variance of the quadratic functions, and Table 2 reports the coefficients of the building blocks of expression (1) that yield the minimum-variance quadratic function for each case. These building blocks have expectation 1 for Brownian Motion but not for random walk, so their coefficients need not add up to unity. Table 1 displays performances similar to those derived for Brownian Motion for moderate df , fast deteriorating when df decreases, in which case S2 data progressively outperforms S1 data. S2 data yields lower variances than S1 data throughout the range, as well as for uniform and double exponentially distributed increments, although the difference in variance in these light-tail cases is as small as for BM. 

Table 1. Minimum variance of mean-1 quadratic functions of S1 and S2 data 

|df N|10, S2|10, S1|30, S2|30, S1|50, S2|50, S1|
|---|---|---|---|---|---|---|
|1.5|16.2403|51.0366|8.3438|32.4697|6.5322|28.3950|
|2.0|4.8444|6.6039|2.6532|3.8327|2.1972|3.2252|
|2.5|2.5864|2.8365|1.4297|1.5529|1.1718|1.2627|
|3.0|1.7359|1.8038|0.9527|0.9782|0.7630|0.7788|
|3.5|1.2334|1.2746|0.6809|0.6991|0.5467|0.5624|
|4.0|0.9469|0.9776|0.5409|0.5585|0.4532|0.4686|
|4.5|0.7864|0.8124|0.4792|0.4957|0.4094|0.4239|
|5.0|0.7071|0.7296|0.4473|0.4629|0.3896|0.4037|
|∞|0.4679|0.4826|0.3630|0.3765|0.3369|0.3496|
|∞, N =∞|||||0.2587|0.27|



It is of interest to observe how does S2 outperform S1 data for low df . Table 2 shows that the role of C is downplayed or even dampened in favor of those of H − C and −L, gradually incorporating C into the Regression as df increases. The rationale for this is that the tail parameter of sums of i.i.d. data is the same as that of the summands, whereas the tail parameter of extrema is the sum of those of the summands. This makes C theoretically as heavy tailed as each increment, but makes H − C and −L have lighter tails than the increments. In contrast, the [h, c, l] data of statistic S1 is less able to split variables into light tail and heavy tail components. Although h −|c| − l = H − C − L, the insistence on resorting to quadratic functions leaves it out of the S1 game. Still, both statistics seem to work fairly well even under low df . In contrast to the variances 2.1972 or 3.2252 for df = 2, 0.7630 or 0.7788 for df = 3 and 0.4532 or 0.4686 for df = 4 (see N = 50 in Table 

8 

- 1), the calibrated C[2] has respective empirical variance above 5000, 16 and 2.5, converging reasonably fast (2 + (df −64)N[)][to][2][thereafter.] 

Table 2. Coefficients of the minimum variance mean-1 quadratic function of S2 data for N = 50 increments per time window 

|df|2((H −C)2 +L2)|C2|2(H −C−L)C|−(H−C)L<br>2 log(2)−5/4|
|---|---|---|---|---|
|1.5|0.0209|-0.0000|0.0010|0.1724|
|2.0|0.1358|-0.0004|0.0352|0.1561|
|2.5|0.1745|-0.0034|0.1573|0.1215|
|3.0|0.1827|0.0140|0.2461|0.1149|
|3.5|0.2006|0.0666|0.2460|0.1228|
|4.0|0.2185|0.1081|0.2442|0.1317|
|4.5|0.2335|0.1271|0.2620|0.1399|
|5.0|0.2480|0.1395|0.2781|0.1473|
|∞|0.3974|0.2321|0.4390|0.2245|
|∞, N =∞|0.2736|0.1604|0.3652|0.2009|



## 4 Appendix - proof of Lemma 2 

For the sake of conciseness, the tedious integration to be presented will be restricted to the identification of E[CHL[2] ], although, in principle, more general joint moments and moment generating function of (C, H, L) could have been identified. 

Consider the infinitesimal event {BM (1) ∈ (ξ, ξ + dξ) , BM (s) ∈ (a, b) , ∀s ∈ [0, 1]}, where a < min(ξ, 0) ≤ 0 ≤ max(ξ, 0) < b. By Siegmund’s Corollary 3.43, its probability Q(ξ, a, b)dξ is as follows 

**==> picture [359 x 32] intentionally omitted <==**

The joint density fc,h,l(ξ, a, b) is (minus) the mixed second derivative of Q with respect to a and b, on {ξ ∈ (a, b) , a < 0 , b > 0}. The joint density fC,H,L is simply 2fc,h,l, restricted to {ξ ∈ (0, b) , a < 0 , b > 0}. The two terms in the j = 0 and second term in the j = 1 summands vanish because they are independent of at least one of a and b. 

To calculate E[CHL[2] ], the contribution of each summand in (14) will be integrated in three univariate steps. The first step will integrate over a ∈ (−∞, 0) the product of a[2] and ∂ the pertinent mixed second derivative. ∂a[φ][(][ξ][+][ Ka][ +][ Mb][)][da][is][to][be][interpreted][as][the] integration-by-parts element dφ(ξ + Ka + Mb), viewed as a function of a. 

**==> picture [153 x 26] intentionally omitted <==**

9 

**==> picture [348 x 76] intentionally omitted <==**

Now expression (15) will be multiplied by ξ and integrated over ξ ∈ (0, b). For K > 0 (K < 0) it is convenient to integrate Φ[∗] (Φ). These terms appear in (16) and (17). The free term in (15) contributes[2] K[M][2][b] 2[2][and][cancels][with][the][corresponding][b][2][term][in][(17).] 

**==> picture [360 x 154] intentionally omitted <==**

Finally, expressions (16) and (17), multiplied by b and integrated over b ∈ (0, ∞), via 

**==> picture [388 x 25] intentionally omitted <==**

yield a rational function of j (with M = 2j and K = −2j or K = −2(j − 1)) whose sum contains only terms of the form −[�][∞] 1[(][−][1)][j][ 1] j[=][log(2)][and][�][∞] 1 j1[3][=][ζ][(3),][as][in][the] statement of Lemma 2. Further details are omitted. 

## 5 Acknowledgements 

The topic under study was motivated by a project at ISTRA Research, Israel. The collaboration of Shlomo Ahal, Jonathan Lewin and Alon Wasserman is greatly appreciated. Ahal’s careful reading and constructive comments are an essential part of the paper. Warm thanks are extended to my hosts at Columbia University’s Statistics Department during a sabbatical visit in the Spring of 2008. 

10 

## References 

- [1] Ball, C. A. & Torous, W. N. (1984). The Maximum Likelihood estimation of security price volatility: theory, evidence and application to option pricing. The Journal of Business, 57, 97–112. 

- [2] Blackwell, D. (1947). Conditional expectation and unbiased sequential estimation. The Annals of Mathematical Statistics, 18, 105–110. 

- [3] Garman, M. B. & Klass, M. J. (1980). On the estimation of security price volatilities from historical data. Journal of Business, 53, 67–78. 

- [4] Joshi, V. M. (1976). On the attainment of the Cram´er–Rao lower bound. The Annals of Statisitics, 4, 998–1002. 

- [5] Magdon-Ismail, M. & Atiya, A. F. (2001). A maximum likelihood approach to volatility estimation for a Brownian motion using the high, low and close. Quantitative Finance, 1, 1–9. 

- [6] Rao, C. R. (1946). Minimum variance and the estimation of several parameters. Proceedings of the Cambridge Philosophical Society, 43, 280–283. 

- [7] Seshadri, V. (1988). Exponential models, Brownian Motion and independence. Canadian Journal of Statistics, 16, 209–221. 

- [8] Siegmund, David O. (1985). Sequential Analysis: tests and confidence intervals. Springer Series in Statistics. Springer Verlag: New York. 

- [9] Yor, M. (1997). Some remarks about the joint law of Brownian Motion and its supremum. S´eminaire de Probabilit´es (Strasburg), 31, 306–314. 

## Isaac Meilijson 

School of Mathematical Sciences 

Raymond and Beverly Sackler Faculty of Exact Sciences 

Tel-Aviv University , 69978 Tel-Aviv, Israel 

E-mail: MEILIJSON@MATH.TAU.AC.IL 

11 

