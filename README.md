# LIMP

A Python library to simplify mixed integer linear programming (MIP / MILP)

I just wrote this to teach myself some linear programming!
You probably shouldn't use it for anything serious.
Although it is built on [SciPy milp()](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html)
(which uses HiGHS as the solver), my code probably has lingering bugs.

On the other hand, I think it ends up being quite an elegant little DSL,
and it encapsulates some MILP patterns that I found quite unintuitive to learn.

Anyway, the accepted solutions as of 2025 seem to be:
- [PuLP](https://coin-or.github.io/pulp/)
- [Python-MIP](https://www.python-mip.com/)

Here are some online resources I found it useful to learn from:
- https://msi-jp.com/xpress/learning/square/10-mipformref.pdf
- http://web.mit.edu/15.053/www/AMP-Chapter-09.pdf
- https://download.aimms.com/aimms/download/manuals/AIMMS3OM_IntegerProgrammingTricks.pdf
- https://optimization.cbe.cornell.edu/index.php?title=Optimization_with_absolute_values

The example notebooks demonstrate approaches to solving some interesting MILP problems.
Some use explicit loops, while others use Numpy to vectorize some of the expressions.
