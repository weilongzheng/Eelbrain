.. currentmodule:: eelbrain

*******
Recipes
*******

.. contents:: Contents
   :local:


^^^^^^^^^^^^^^^^^^^^
Group Level Analysis
^^^^^^^^^^^^^^^^^^^^

To do group level analysis one usually wants to construct a :class:`Dataset`
that contains results for each participants along with condition and subject
labels. The following illustration assumes functions that compute results
for a single subject and condition:

  - ``result_for(subject, condition)`` returns an :class:`NDVar`.
  - ``scalar_result_for(subject, condition)`` returns a scalar
    (:class:`float`).

Given results by subject and condition, a Dataset can be constructed as
follows::

    >>> # create lists to collect data and labels
    >>> ndvar_results = []
    >>> scalar_results = []
    >>> subjects = []
    >>> conditions = []
    >>> # collect data and labels
    >>> for subject in ('s1', 's2', 's3', 's4'):
    ...	    for condition in ('c1', 'c2'):
    ...	        ndvar = result_for(subject, condition)
    ...         s = scalar_result_for(subject, condition)
    ...	        ndvar_results.append(ndvar)
    ...         scalar_results.append(s)
    ...	        subjects.append(subject)
    ...	        conditions.append(condition)
    ...
    >>> # create a Dataset and convert the collected lists to appropriate format
    >>> ds = Dataset()
    >>> ds['subject'] = Factor(subjects, random=True)  # treat as random effect
    >>> ds['condition'] = Factor(conditions)
    >>> ds['y'] = combine(ndvar_results)
    >>> ds['s'] = Var(scalar_results)


Now this Dataset can be used for statistical analysis, for example ANOVA::

    >>> res = testnd.anova('y', 'condition * subject', ds=ds)


.. _recipe-regression:

^^^^^^^^^^^^^^^^^
Regression Design
^^^^^^^^^^^^^^^^^

The influence of a continuous predictor on single trial level can be tested by
first calculating regression coefficients for each subject, and then performing
a one sample t-test across subjects to test the null hypothesis that between
subjects, the regression coefficient does not differ significantly from 0.

Assuming that ``ds_subject`` is a :class:`Dataset` containing single trial data
for one subject, with ``data`` the dependent variable and a predictor (called
``predictor``)::

    >>> ds_subject
    <Dataset 'example' n_cases=145 {'predictor':V, 'data':Vnd}>
    >>> ds_subject['data']
    <NDVar 'data': 145 (case) X 5120 (source) X 76 (time)>
    >>> print ds_subject
    predictor
    ---------
    1.9085
    0.30836
    -0.58802
    0.29686
    ...

The regression coefficient can be calculated the following way::

    >>> beta = ds_subject.eval("data.ols(predictor)")
    >>> beta
    <NDVar 'ols': 1 (case) X 5120 (source) X 76 (time)>

Thus, in order to collect beta values for each subject, you would loop through
subjects. We will call the NDVar with beta values 'beta'::

    >>> subjects = []
    >>> betas = []
    >>> for subject in ['R0001', 'R0002', 'R0003']:
    ...     ds_subject = my_load_ds_for_subject_function(subject)
    ...     beta = ds_subject.eval("data.ols(predictor, 'beta')")
    ...     subjects.append(subject)
    ...     betas.append(beta)
    ...
    >>> ds = Dataset()
    >>> ds['subject'] = Factor(subjects, random=True)
    >>> ds['beta'] = combine(betas)

Now you can perform a one-sample t-test::

    >>> res = testnd.ttest_1samp('beta', ...)

And analyze the results as for other :mod:`testnd` tests.


^^^^^^^^^^^^^^^^^^^^^
Plots for Publication
^^^^^^^^^^^^^^^^^^^^^

In order to produce multiple plots it is useful to set some plotting parameters
globally in order to ensure that they are consistent between plots, e.g.::

    import matplotlib as mpl

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 8
    for key in mpl.rcParams:
        if 'width' in key:
            mpl.rcParams[key] *= 0.5
    mpl.rcParams['savefig.dpi'] = 300  # different from 'figure.dpi'!


Matplotlib's :func:`~matplotlib.pyplot.tight_layout` functionality provides an
easy way for plots to use the available space, and most Eelbrain plots use it
by default. However, when trying to produce multiple plots with identical
scaling it can lead to unwanted discrepancies. In this case, it is better to
define layout parameters globally and plot with the ``tight=False`` argument::

    mpl.rcParams['figure.subplot.left'] = 0.25
    mpl.rcParams['figure.subplot.right'] = 0.95
    mpl.rcParams['figure.subplot.bottom'] = 0.2
    mpl.rcParams['figure.subplot.top'] = 0.95

    plot.UTSStat('uts', 'A', ds=ds, w=5, tight=False)

    # now we can produce a second plot without x-axis labels that has exactly
    # the same scaling:
    plot.UTSStat('uts', 'A % B', ds=ds, w=5, tight=False, xlabel=False, ticklabels=False)


If a script produces several plots, the GUI should not interrupt the script.
This can be achieved by setting the ``show=False`` argument. In addition, it
is usually desirable to save the legend separately and combine it in a layout
application::

    p = plot.UTSStat('uts', 'A', ds=ds, w=5, tight=False, show=False, legend=False)
    p.save('plot.svg', transparent=True)
    p.save_legend('legend.svg', transparent=True)
