use crate::objects::{Constant, Expr, MulExpr, SumExpr, Variable};

impl PartialEq for Expr {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Expr::C(c1), Expr::C(c2)) => c1 == c2,
            (Expr::V(v1), Expr::V(v2)) => v1 == v2,
            (Expr::S(s1), Expr::S(s2)) => s1 == s2,
            (Expr::M(m1), Expr::M(m2)) => m1 == m2,
            _ => false,
        }
    }
}

impl PartialEq for Constant {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl PartialEq for Variable {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.value == other.value
    }
}

impl PartialEq for SumExpr {
    fn eq(&self, other: &Self) -> bool {
        self.left == other.left && self.right == other.right
    }
}

impl PartialEq for MulExpr {
    fn eq(&self, other: &Self) -> bool {
        self.left == other.left && self.right == other.right
    }
}

impl PartialEq<Constant> for Expr {
    fn eq(&self, other: &Constant) -> bool {
        match self {
            Expr::C(c) => c == other,
            _ => false,
        }
    }
}

impl PartialEq<Variable> for Expr {
    fn eq(&self, other: &Variable) -> bool {
        match self {
            Expr::V(v) => v == other,
            _ => false,
        }
    }
}

impl PartialEq<SumExpr> for Expr {
    fn eq(&self, other: &SumExpr) -> bool {
        match self {
            Expr::S(s) => s == other,
            _ => false,
        }
    }
}

impl PartialEq<MulExpr> for Expr {
    fn eq(&self, other: &MulExpr) -> bool {
        match self {
            Expr::M(m) => m == other,
            _ => false,
        }
    }
}
