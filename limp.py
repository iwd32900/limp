'''
Experiments with a mini-language to help with
mixed integer linear programming (MILP).

Useful online resources:
- https://msi-jp.com/xpress/learning/square/10-mipformref.pdf
- http://web.mit.edu/15.053/www/AMP-Chapter-09.pdf
- https://download.aimms.com/aimms/download/manuals/AIMMS3OM_IntegerProgrammingTricks.pdf
- https://optimization.cbe.cornell.edu/index.php?title=Optimization_with_absolute_values
'''

from numbers import Real
import numpy as np
import scipy.optimize as opt

class Var:
    '''
    Named decision variable with optional upper and lower bounds.
    Prefer using the contvar(), intvar(), and binvar() methods of Problem over creating Vars directly.
    Vars may be combined into linear expressions using normal mathematical operators.
    
    The "semi" variable types are allowed to be zero even if their bounds do not include zero.
    I have not really figured out what one uses them for yet, though.
    '''
    CONTINUOUS = 0
    INTEGER = 1
    SEMI_CONT = 2
    SEMI_INT = 3
    
    def __init__(self, name: str, vartype: int, lower: float, upper: float):
        self.name = name
        self.vartype = vartype
        # Changing these later can screw up some expressions!
        self._lower = lower
        self._upper = upper
    # Make bounds into read-only properties to discourage people from trying to change them
    @property
    def lower(self):
        return self._lower
    @property
    def upper(self):
        return self._upper
    @property
    def is_int(self) -> bool:
        return self.vartype in (self.INTEGER, self.SEMI_INT)
    @property
    def is_binary(self) -> bool:
        return self.is_int and self.lower == 0 and self.upper == 1
    def __invert__(self):
        if not self.is_binary:
            raise ValueError("Can only invert binary variables")
        return 1 - as_expr(self)
    def __add__(self, other):
        return as_expr(self) + other
    def __sub__(self, other):
        return as_expr(self) - other
    def __mul__(self, other):
        return as_expr(self) * other
    def __truediv__(self, other):
        return as_expr(self) / other
    def __radd__(self, other):
        return other + as_expr(self)
    def __rsub__(self, other):
        return other - as_expr(self)
    def __rmul__(self, other):
        return other * as_expr(self)
    def __neg__(self):
        return -as_expr(self)
    def __repr__(self):
        return f'Var[{self.lower}<={self.name}<={self.upper}]'
        
class Expr:
    '''
    Linear expression representing a weighted sum of Vars plus some optional constant.
    In most cases, prefer to create Expr by using normal mathematical operations on Vars,
    or by using methods like sum() from Problem, or as_expr() on a Var or number.
    '''
    def __init__(self, terms: dict[Var,float] = {}, const: float = 0):
        self.terms = dict(terms)
        self.const = const
    def __eq__(self, other):
        other = as_expr(other)
        return self.terms == other.terms and self.const == other.const
    def __add__(self, other):
        other = as_expr(other)
        new = Expr(self.terms, self.const) # makes a copy; neither copy() nor deepcopy() does what we want
        for var, wt in other.terms.items():
            new.terms.setdefault(var, 0)
            new.terms[var] += wt
        new.const += other.const
        return new
    def __sub__(self, other):
        other = as_expr(other)
        return self.__add__(-other)
    def __mul__(self, other):
        if not isinstance(other, Real):
            raise ValueError("Can only multiply/divide an Expr by a real number")
        new = Expr(self.terms, self.const) # makes a copy; neither copy() nor deepcopy() does what we want
        for var in new.terms.keys():
            new.terms[var] *= other
        new.const *= other
        return new
    def __truediv__(self, other):
        return self.__mul__(1/other)
    def __radd__(self, other):
        return self.__add__(other)
    def __rsub__(self, other):
        return (-self).__add__(other)
    def __rmul__(self, other):
        return self.__mul__(other)
    def __neg__(self):
        return Expr({var: -wt for var, wt in self.terms.items()}, -self.const)
    def __repr__(self):
        return ' + '.join([str(self.const)] + [f'{wt}*{var.name}' for var, wt in self.terms.items()])
    @property
    def lower(self):
        'Lower bound for this expression.'
        return self.const + sum(min(var.lower*wt, var.upper*wt) for var, wt in self.terms.items())
    @property
    def upper(self):
        'Upper bound for this expression.'
        return self.const + sum(max(var.lower*wt, var.upper*wt) for var, wt in self.terms.items())
    def eval(self, var_to_val: dict[Var, float]) -> float:
        '''
        Evaluate this expression for specific values of the variables involved.
        '''
        return self.const + sum(var_to_val[var]*wt for var, wt in self.terms.items())

def as_expr(x):
    '''
    Accepts number or Var as input and returns the equivalent Expr.
    If an Expr is passed in, it is returned unchanged.
    '''
    if isinstance(x, Expr):
        return x
    elif isinstance(x, Var):
        return Expr(terms = {x: 1})
    else:
        return Expr(const = x)

class Problem:
    '''
    Mixed integer linear programming problem formulation
    '''
    def __init__(self):
        self.constraints = []
    def contvar(self, name: str, lower: float = -np.inf, upper: float = np.inf):
        '''Create a real-valued variable.'''
        return Var(name=name, vartype=Var.CONTINUOUS, lower=lower, upper=upper)
    def eqvar(self, name: str, x: Expr, lower: float = None, upper: float = None):
        '''
        Create a new variable that is constrained to equal some expression of other variables.
        Only do this if you want to call a function that does not accept Expr directly,
        such as min(), max(), or prod().
        '''
        if lower is None: lower = x.lower
        if upper is None: upper = x.upper
        v = Var(name=name, vartype=Var.CONTINUOUS, lower=lower, upper=upper)
        self.equal(v, x)
        return v        
    def intvar(self, name: str, lower: float = -np.inf, upper: float = np.inf):
        '''Create a integer-valued variable.  Best for relatively small bounds.'''
        return Var(name=name, vartype=Var.INTEGER, lower=lower, upper=upper)
    def binvar(self, name: str):
        '''
        Create a binary-valued variable (i.e. either 0 or 1).
        These are a workhorse for many problems!
        '''
        return Var(name=name, vartype=Var.INTEGER, lower=0, upper=1)
    def _var_array(self, name: str, shape: tuple[int], vartype: int, lower: float, upper: float):
        '''Create a Numpy array of decision variables named after their positional indices.'''
        @np.vectorize
        def make_var(*ix, vartype, lower, upper):
            name_i = '_'.join([name] + [str(i) for i in ix]) # like x_1_2_3
            return Var(name_i, vartype, lower, upper)
        return np.fromfunction(make_var, shape=shape, dtype=int, vartype=vartype, lower=lower, upper=upper)
    def intvar_array(self, name: str, shape: tuple[int], lower: float = -np.inf, upper: float = np.inf):
        '''Create a Numpy array of decision variables named after their positional indices.'''
        return self._var_array(name=name, vartype=Var.INTEGER, lower=lower, upper=upper, shape=shape)
    def binvar_array(self, name: str, shape: tuple[int]):
        '''
        Create a Numpy array of binary decision variables named after their positional indices.
        Many problems end up creating a 2- or 3-D array of binary variables, hence this function.
        '''
        return self._var_array(name=name, vartype=Var.INTEGER, lower=0, upper=1, shape=shape)
    def constraint(self, lower: Expr, middle: Expr, upper: Expr = None):
        '''
        Establish new constraint(s) that lower <= middle, and middle <= upper if upper is provided.
        '''
        lower = as_expr(lower)
        middle = as_expr(middle)
        if upper is None:
            # simple -- one-sided inequality
            # move all vars into middle, all consts into lower
            lb = lower.const - middle.const
            x = (middle - lower).terms
            ub = np.inf
            self.constraints.append((lb, x, ub))
        else:
            upper = as_expr(upper)
            if lower.terms == upper.terms:
                # can make lower and upper into real numbers, consolidate all vars in middle
                lb = lower.const - middle.const
                x = (middle - lower).terms
                ub = upper.const - middle.const
                self.constraints.append((lb, x, ub))
            else:
                # two one-sided constraints, not a single two-sided constraint
                self.constraint(lower, middle)
                self.constraint(middle, upper)
    def equal(self, left: Expr, right: Expr):
        '''
        Establish a constraint that left and right are equal, i.e. left <= right <= left.
        '''
        return self.constraint(left, right, left)
    def minimize(self, objective: Expr, **options):
        '''
        Minimize the given objective expression subject to established constraints.
        If the expression is a constant (e.g. zero), just find a feasible solution
        (one that satisfies all the constraints).

        `options` are the same ones that scipy.optimize.milp() accepts.
        '''
        # Assign every unique variable an integer index:
        vs = {}
        objective = as_expr(objective)
        for v in objective.terms.keys():
            vs.setdefault(v, len(vs))
        for lb, x, ub in self.constraints:
            for v in x.keys():
                vs.setdefault(v, len(vs))
        # Establish bounds for variables:
        L = np.zeros(len(vs))
        U = np.zeros(len(vs))
        for v, i in vs.items():
            L[i] = v.lower
            U[i] = v.upper
        bounds = opt.Bounds(lb=L, ub=U)
        # Establish bounds for constraints:
        A = np.zeros((len(self.constraints), len(vs)))
        L = np.zeros(len(self.constraints))
        U = np.zeros(len(self.constraints))
        for row, (lb, x, ub) in enumerate(self.constraints):
            L[row] = lb
            U[row] = ub
            for v, wt in x.items():
                col = vs[v]
                A[row,col] = wt
        linconst = opt.LinearConstraint(A, lb=L, ub=U)
        # Establish objective (can ignore constant for minimization):
        c = np.zeros(len(vs))
        for v, wt in objective.terms.items():
            i = vs[v]
            c[i] = wt
        # Capture variable types (continuous, integer, etc)
        integrality = [v.vartype for v in vs.keys()]
        soln = opt.milp(c=c, integrality=integrality, bounds=bounds, constraints=linconst, options=options)
        retval = dict(soln=soln, vs=vs, c=c, integrality=integrality, bounds=bounds, constraints=linconst, options=options)
        # In the event of a timeout, soln.success will be False, but variable values may be available
        # That is, constraints may be satisfied but objective is not yet fully minimized
        if soln.x is not None:
            retval['by_name'] = {v.name: x for v, x in zip(vs, soln.x)} # not ideal, variables could share same name
            retval['by_var'] = dict(zip(vs, soln.x)) # guaranteed unique keys
        return retval
    def maximize(self, objective: Expr, **options):
        '''Maximize the objective function, instead of minimizing it.'''
        return self.minimize(-as_expr(objective), **options)
    def solve(self, **options):
        '''
        Find a feasible solution without trying to minimize any specific objective,
        beyond just satisfying all the constraints.
        '''
        return self.minimize(0, **options)
    # Helper functions to formulate common ideas.
    # Some of these could be standalone, but others need to be able to create helper variables, constraints, etc
    # So for consistency, they're all implemented here
    def sum(self, vs: list[Var], wts=None) -> Expr:
        '''
        Create an expression representing a (possibly weighted) sum of variables.
        Python's sum() and numpy.sum() also work, and accept a mix of Var and constants.
        '''
        if wts is None:
            wts = [1]*len(vs)
        return Expr(terms=dict(zip(vs, wts)))
    def _check_binary(self, vs: list[Var]):
        if not all(v.is_binary for v in vs):
            raise ValueError("Only binary variables allowed")
    def at_least(self, n: int, vs: list[Var]):
        '''Constrain at least n of the provided binary variables to be true.'''
        self._check_binary(vs)
        return self.constraint(n, self.sum(vs))
    def at_most(self, n: int, vs: list[Var]):
        '''Constrain at most n of the provided binary variables to be true.'''
        self._check_binary(vs)
        return self.constraint(self.sum(vs), n)
    def exactly(self, n: int, vs: list[Var]):
        '''Constrain exactly n of the provided binary variables to be true.'''
        self._check_binary(vs)
        return self.constraint(n, self.sum(vs), n)
    def ifthen(self, e_if: Expr, e_then: Expr):
        '''
        If binary variable e_if is true, constrain e_then to be true.
        If e_if is false, then e_then may be either true or false.
        (That is, e_if <= e_then.)
        '''
        # if and then should be binary variables or negations thereof
        return self.constraint(e_if, e_then)
    def all(self, vs: list[Var]) -> Var:
        'Logical AND: create a binary variable that is true iff all(vs)'
        self._check_binary(vs)
        name = 'all('+','.join(v.name for v in vs)+')'
        d = self.binvar(name)
        self.constraint(self.sum(vs) - (len(vs) - 1), d)
        for v in vs:
            self.constraint(d, v)
        return d
    def any(self, vs: list[Var]) -> Var:
        'Logigal OR: create a binary variable that is true iff any(vs)'
        self._check_binary(vs)
        name = 'any('+','.join(v.name for v in vs)+')'
        d = self.binvar(name)
        self.constraint(d, self.sum(vs))
        for v in vs:
            self.constraint(v, d)
        return d
    def _check_finite(self, nums: list):
        if not np.all(np.isfinite(np.asarray(nums))):
            raise ValueError('Cannot use unbounded variables here')
    def min(self, vs: list[Var]) -> Var:
        'Create a new variable that is the minimum of the listed vars, which must all have finite bounds.'
        name = 'min('+','.join(v.name for v in vs)+')'
        min_lower = min(v.lower for v in vs)
        min_upper = min(v.upper for v in vs)
        self._check_finite([v.lower for v in vs] + [v.upper for v in vs])
        if all(v.is_int for v in vs):
            minval = self.intvar(name, min_lower, min_upper)
        else:
            minval = self.contvar(name, min_lower, min_upper)
        # decision variables to see if v is the smallest
        ds = [self.binvar(f'is_min({v.name})') for v in vs]
        self.exactly(1, ds)
        for d, v in zip(ds, vs):
            # If v is minimal, d=1 and v <= minval <= v, that is minval == v
            # Otherwise, we establish a pointless lower bound that is at most min_lower
            self.constraint(v - (v.upper - min_lower)*(1 - d), minval, v)
        return minval
    def max(self, vs: list[Var]) -> Var:
        'Create a new variable that is the maximum of the listed vars, which must all have finite bounds.'
        name = 'max('+','.join(v.name for v in vs)+')'
        max_lower = max(v.lower for v in vs)
        max_upper = max(v.upper for v in vs)
        self._check_finite([v.lower for v in vs] + [v.upper for v in vs])
        if all(v.is_int for v in vs):
            maxval = self.intvar(name, max_lower, max_upper)
        else:
            maxval = self.contvar(name, max_lower, max_upper)
        # decision variables to see if v is the biggest
        ds = [self.binvar(f'is_max({v.name})') for v in vs]
        self.exactly(1, ds)
        for d, v in zip(ds, vs):
            # If v is maximal, d=1 and v <= maxval <= v, that is maxval == v
            # Otherwise, we establish a pointless upper bound that is at least max_upper
            self.constraint(v, maxval, v + (max_upper - v.lower)*(1 - d))
        return maxval
    def prod(self, x: Var, d: Var) -> Var:
        '''
        Create a new variable that is the product of any variable x and a binary variable d.
        That is, the new variable equals x if d is true, and equals 0 if d is false.
        '''
        if not d.is_binary:
            raise ValueError("Can only multiply a variable with a binary")
        y = Var(f'{x.name}*{d.name}', x.vartype, min(0, x.lower), max(0, x.upper))
        self._check_finite([x.lower, x.upper])
        self.constraint(x.lower*d, y, x.upper*d)
        self.constraint(x.lower*(1-d), x - y, x.upper*(1-d))
        return y
    def abs(self, x: Expr) -> Var:
        '''
        Create a new variable (and appropriate constraints) for absolute value of an expression.
        This is the most general solution I know, which is usable everywhere (except unbounded expressions).
        Other, more efficient solutions are possible in some cases.  For discussion, see
            https://optimization.cbe.cornell.edu/index.php?title=Optimization_with_absolute_values
        '''
        x = as_expr(x)
        z = self.contvar(f'abs({x})')
        x_upper = x.upper
        x_lower = x.lower
        if x_lower >= 0: # x is always non-negative
            self.constraint(z, x)
            return z
        elif x_upper <= 0: # x is always non-positive
            self.constraint(z, -x)
            return z
        d = self.binvar(f'is_pos({x})')
        bigM = 2*max(abs(x_upper), abs(x_lower))
        self._check_finite([bigM])
        self.constraint(x, z)
        self.constraint(-x, z)
        self.constraint(z, x + bigM*(1-d))
        self.constraint(z, -x + bigM*d)
        return z
