use crate::objects::{Constant, Expr, MulExpr, SumExpr};
use std::ops::{Add, Mul};
use std::rc::Rc;

impl Add for Expr {
    type Output = Expr;

    fn add(self, rhs: Expr) -> Expr {
        match (self, rhs) {
            (Expr::C(Constant { value: 0.0 }), Expr::C(c2)) => Expr::C(c2),
            (Expr::C(c1), Expr::C(Constant { value: 0.0 })) => Expr::C(c1),
            (Expr::C(c1), Expr::C(c2)) => Expr::C(Constant {
                value: c1.value + c2.value,
            }),
            (lhs, rhs) => Expr::S(SumExpr {
                left: Rc::new(lhs),
                right: Rc::new(rhs),
            }),
        }
    }
}

impl Mul for Expr {
    type Output = Expr;

    fn mul(self, rhs: Expr) -> Expr {
        match (self, rhs) {
            (Expr::C(Constant { value: 0.0 }), _) => Expr::C(Constant { value: 0.0 }),
            (_, Expr::C(Constant { value: 0.0 })) => Expr::C(Constant { value: 0.0 }),
            (Expr::C(Constant { value: 1.0 }), Expr::C(c2)) => Expr::C(c2),
            (Expr::C(c1), Expr::C(Constant { value: 1.0 })) => Expr::C(c1),
            (Expr::C(c1), Expr::C(c2)) => Expr::C(Constant {
                value: c1.value * c2.value,
            }),
            (lhs, rhs) => Expr::M(MulExpr {
                left: Rc::new(lhs),
                right: Rc::new(rhs),
            }),
        }
    }
}
