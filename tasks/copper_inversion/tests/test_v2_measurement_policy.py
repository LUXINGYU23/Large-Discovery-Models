import numpy as np
import pytest
from tasks.copper_inversion.experiments.v2.measurement_policy import scales


@pytest.mark.parametrize('spacing',[[1,1,1],[1,1.5,2],[4,4,4]])
@pytest.mark.parametrize('noisy',[False,True])
def test_physical_noise_and_discretization_separate(spacing,noisy):
    policy=scales(spacing,noisy)
    assert policy['measurement_sigma']==(.25*min(spacing) if noisy else 0.)
    assert policy['effective_scale']>=policy['discretization_scale']>0
    converted=scales(np.array(spacing)*1000,noisy)
    for key in policy:assert converted[key]==pytest.approx(policy[key]*1000)


@pytest.mark.parametrize('spacing',[[0,1,1],[-1,1,1],[1,1],[float('nan'),1,1]])
def test_invalid_spacing(spacing):
    with pytest.raises(ValueError):scales(spacing,False)


def test_fixed_geological_policy_and_initialization():
    from tasks.copper_inversion.experiments.v2.calibrate_priors import check_numerics
    assert all(row['passed'] for row in check_numerics())


def test_positive_volume_prior_with_no_positive_observations():
    from tasks.copper_inversion.v2.fixtures import fixtures
    from tasks.copper_inversion.v2.baselines import initialize
    from tasks.copper_inversion.v2.geometry import canonicalize
    from tasks.copper_inversion.v2.geological_policy import public_priors
    problem=next(fixtures());problem['observations'][0]['labels'][:]=0.
    problem['priors']=public_priors(problem['domain_mask']&~problem['forbidden_mask'],problem['spacing'])
    mask,_=canonicalize(problem,initialize(problem))
    assert mask.any() and not np.any(mask & problem['forbidden_mask'])
