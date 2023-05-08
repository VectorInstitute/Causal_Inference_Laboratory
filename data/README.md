# Description of datasets
We briefly discuss the datasets here.

## IHDP
Infant Health and Development Program (IHDP) [1] is from a
randomized experiment studying the effect of home visits by specialists on
future cognitive test scores of children. The children of non-white mothers in
the treated set are removed to de-randomize the experiment. Each unit is
simulated for a treated and a control outcome (so we know the ground-truth of
the individual treatment effects).



The IHDP datasets are already split into the train (672 for each realization)
and test (75 for each realization) splits in a 90/10 split. Each `.npz` file
contains the following keys: x, t, yf, ycf, mu0, mu1, which are respectively
covariates, treatment, factual outcome, counterfactual outcome, noiseless
potential control outcome, and noiseless potential treated outcome.
- IHDP-100: 100 realizations of the IHDP dataset (included in our repo);
- IHDP-1000: 1000 realizations of the IHDP dataset
(downloadable from https://www.fredjo.com/);

## Jobs
Jobs is a dataset derived from LaLonde [2] where the original data set has job
training as the treatment and income and employment status after training as
outcomes. The Jobs dataset is proposed in [3] using the LaLonde experimental
sample (297 treated, 425 control) and the PSID comparison group (2490 control).




The Jobs datasets are already split into the train (2570 for each realization)
and test (642 for each realization) splits in a 80/20 split. Each `.npz` file
contains the following keys: x, t, yf, ate, which are respectively
covariates, treatment, factual outcome, and average treatment effect (scalar).
- Jobs: 10 realizations of the Jobs dataset (included in our repo);

## References
[1] J. L. Hill, “Bayesian nonparametric modeling for causal inference,” Journal
of Computational  and Graphical Statistics, vol. 20, no. 1, pp. 217–240, 2011.
[Online]. Available: https://doi.org/10.1198/jcgs.2010.08162

[2] R. J. LaLonde, “Evaluating the econometric evaluations of training programs
with experimental data,” The American Economic Review, vol. 76, no. 4, pp.
604–620, 1986. [Online]. Available: http://www.jstor.org/stable/1806062

[3] U. Shalit, F. D. Johansson, and D. Sontag, “Estimating individual treatment
effect: generalization bounds and algorithms,” in Proceedings of the 34th
International Conference on Machine Learning, ser. Proceedings of Machine
Learning Research, D. Precup and Y. W. Teh, Eds., vol. 70. PMLR, 06–11 Aug 2017
, pp. 3076–3085. [Online].
Available: https://proceedings.mlr.press/v70/shalit17a.html
