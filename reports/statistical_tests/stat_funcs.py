from scipy import stats
import numpy as np


def check_significance(p_value, alphas=[0.01, 0.05, 0.1]):
    for alpha in alphas:
        significant = p_value < alpha
        if significant:
            return alpha
    return None


# Perform the Shapiro-Wilk test for normality.
# The Shapiro-Wilk test tests the null hypothesis that the data was drawn from a normal distribution.
def shapiro_test(sample, alphas=[0.01, 0.05, 0.1]):
    _, p_value = stats.shapiro(sample)
    significance = check_significance(p_value, alphas)
    return significance


# Calculate the T-test for the mean of ONE group of scores OR on TWO RELATED samples of scores.
# The first one is a test for the null hypothesis that the expected value (mean) of a sample of independent observations is equal to the given population mean
# The second one is a test for the null hypothesis that two related or repeated samples have identical average (expected) values.
def t_test(
    sample1, sample2=None, pop_mean=0, alternative="less", alphas=[0.01, 0.05, 0.1]
):
    if sample2 is None:
        _, p_value = stats.ttest_1samp(
            sample1, popmean=pop_mean, alternative=alternative
        )
    else:
        _, p_value = stats.ttest_rel(sample1, sample2, alternative=alternative)
    significance = check_significance(p_value, alphas)
    return significance


# Calculate the Wilcoxon signed-rank test.
# The Wilcoxon signed-rank test tests the null hypothesis that two related paired samples come from the same distribution.
# In particular, it tests whether the distribution of the differences x - y is symmetric about zero.
# It is a non-parametric version of the paired T-test.
def wilcoxon_test(
    sample1,
    sample2=None,
    alternative="greater",
    alphas=[0.01, 0.05, 0.1],
    zero_method=["wilcox", "pratt", "zsplit"],
):
    significance = []
    if isinstance(zero_method, str):
        zero_method = [zero_method]
    if np.allclose(sample1, sample2):
        significance = ["identical", "identical"]
        zero_method = ["zsplit"]
    for zero_method in zero_method:
        _, p_value = stats.wilcoxon(
            sample1,
            sample2,
            zero_method=zero_method,
            correction=False,
            alternative=alternative,
        )
        significance.append(check_significance(p_value, alphas))
    return significance
