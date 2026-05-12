import typing

from opfunu.utils import latex

if typing.TYPE_CHECKING:
    from IPython.display import Latex


class LatexFormula(type):
    name: typing.ClassVar[str] = "Benchmark name"
    latex_formula: typing.ClassVar[str] = r'f(\mathbf{x})'
    latex_formula_dimension: typing.ClassVar[str] = r'd \in \mathbb{N}_{+}^{*}'
    latex_formula_bounds: typing.ClassVar[str] = r'x_i \in [-2\pi, 2\pi], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum: typing.ClassVar[str] = r'f(0, ..., 0)=-1, \text{ for}, m=5, \beta=15'

    @property
    def formula(cls) -> "Latex":
        return latex.render_latex(cls.latex_formula)

    @property
    def formula_dimension(cls) -> "Latex":
        return latex.render_latex(cls.latex_formula_dimension)

    @property
    def formula_bounds(cls) -> "Latex":
        return latex.render_latex(cls.latex_formula_bounds)

    @property
    def formula_global_optimum(cls) -> "Latex":
        return latex.render_latex(cls.latex_formula_global_optimum)
