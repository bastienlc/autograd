use crate::objects::{Constant, Expr, MulExpr, SumExpr, Variable};
use std::fmt;

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Expr::C(c) => write!(f, "{}", c),
            Expr::V(v) => write!(f, "{}", v),
            Expr::S(s) => write!(f, "{}", s),
            Expr::M(m) => write!(f, "{}", m),
        }
    }
}

impl fmt::Display for Constant {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl fmt::Display for Variable {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl fmt::Display for SumExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({} + {})", self.left, self.right)
    }
}

impl fmt::Display for MulExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({} * {})", self.left, self.right)
    }
}
