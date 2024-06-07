import numpy as np
from scipy.integrate import solve_ivp

# debugging help
def pause():
    programPause = raw_input("Press the <ENTER> key to continue...")

def default_algorithm(prob, **kwargs):
    o = dict(kwargs)
    extra_kwargs = []
    alg = 'RK45'  # Default algorithm

    uEltype = type(prob['u0'][0])

    alg_hints = o.get('alg_hints', [])
    tol_level = o.get('tol_level', 'low_tol')
    callbacks = o.get('callbacks', False)
    mm = o.get('mass_matrix', False)

    if 'stiff' in alg_hints and 'nonstiff' in alg_hints:
        raise ValueError("The problem must either be designated as stiff or non-stiff")

    if not isinstance(prob['t_span'][0], float) and 'adaptive' not in o:
        extra_kwargs.append({'adaptive': False})

    if prob['f'].__name__ == 'split_function':
        alg = 'LSODA'
    elif prob['f'].__name__ == 'dynamical_ode_function':
        if tol_level in ['low_tol', 'med_tol']:
            alg = 'RK45'
        else:
            alg = 'DOP853'
    else:
        if 'nonstiff' in alg_hints:
            if uEltype not in [float, np.float32, np.complex64, np.complex128] or tol_level in ['extreme_tol', 'low_tol']:
                alg = 'DOP853'
            else:
                alg = 'RK45'
        elif 'stiff' in alg_hints or mm:
            if len(prob['u0']) > 500:
                alg = 'LSODA'
            elif len(prob['u0']) > 50:
                alg = 'LSODA'
            elif tol_level == 'high_tol':
                alg = 'Radau'
            else:
                alg = 'BDF'
        else:
            if uEltype not in [float, np.float32] or tol_level in ['extreme_tol', 'low_tol']:
                if len(prob['u0']) > 500:
                    alg = 'DOP853'
                elif len(prob['u0']) > 50:
                    alg = 'DOP853'
                else:
                    alg = 'DOP853'
            else:
                if len(prob['u0']) > 500:
                    alg = 'RK23'
                elif len(prob['u0']) > 50:
                    alg = 'RK23'
                else:
                    alg = 'RK45'

    return alg, extra_kwargs

def solve_ode(prob, **kwargs):
    alg, extra_kwargs = default_algorithm(prob, **kwargs)
    solver = kwargs.get('solver', alg)
    solution = solve_ivp(prob['f'], prob['t_span'], prob['u0'], method=solver, **kwargs)
    return solution