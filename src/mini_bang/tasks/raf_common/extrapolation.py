"""Separation between observable and predicted RAF network sizes."""
from collections.abc import Mapping


def observation_range(visible):
    """Return an explicit query range; instances without experimental_domain are unrestricted."""
    if 'experimental_domain' not in visible:
        return None
    domain = visible['experimental_domain']
    if not isinstance(domain, Mapping) or set(domain) != {'saturation'}:
        raise ValueError('experimental_domain must declare a saturation range')
    bounds = domain['saturation']
    if (not isinstance(bounds, Mapping) or set(bounds) != {'min', 'max'}
            or any(type(bounds[key]) is not int for key in ('min', 'max'))
            or not 0 < bounds['min'] <= bounds['max']):
        raise ValueError('Experimental saturation bounds must be positive ordered integers')
    targets = prediction_sizes(visible)
    if not targets or any(type(value) is not int or value <= bounds['max'] for value in targets):
        raise ValueError('Size extrapolation targets must all exceed the observable maximum')
    return dict(bounds)


def prediction_sizes(visible):
    if 'saturations' in visible:
        return list(visible['saturations'])
    return [value for sequence in visible.get('sequences', []) for value in sequence['saturations']]


def validate_observation(visible, request, fixed):
    bounds = observation_range(visible)
    if bounds is None:
        return
    sizes = [request.saturation] if type(request.saturation) is int else list(request.saturation)
    if any(not bounds['min'] <= value <= bounds['max'] for value in sizes):
        raise ValueError(
            f'Only network sizes {bounds["min"]}..{bounds["max"]} may be observed; '
            'larger sizes are prediction targets, including within a sequence or batch; no budget was spent')
    # Public APIs merge these parameters. Enforce them at the session boundary
    # as well, so raw requests cannot emulate a larger network with another M0.
    for group, values in fixed.items():
        actual = getattr(request, group)
        if any(key not in actual or actual[key] != value for key, value in values.items()):
            raise ValueError(f'{group} is fixed by this extrapolation instance; no budget was spent')
