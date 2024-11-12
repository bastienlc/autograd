use crate::objects::{Constant, Expr, MulExpr, SumExpr, Variable};
use std::rc::Rc;
pub trait Differentiable {
    fn backward(&self, x: &Expr) -> Expr;
    fn forward(&self) -> f64;
}

impl Differentiable for Expr {
    fn backward(&self, x: &Expr) -> Expr {
        match self {
            Expr::C(c) => c.backward(x),
            Expr::V(v) => v.backward(x),
            Expr::S(s) => s.backward(x),
            Expr::M(m) => m.backward(x),
        }
    }

    fn forward(&self) -> f64 {
        match self {
            Expr::C(c) => c.forward(),
            Expr::V(v) => v.forward(),
            Expr::S(s) => s.forward(),
            Expr::M(m) => m.forward(),
        }
    }
}

impl Differentiable for Constant {
    fn backward(&self, _x: &Expr) -> Expr {
        Expr::C(Constant { value: 0.0 })
    }

    fn forward(&self) -> f64 {
        self.value
    }
}

impl Differentiable for Variable {
    fn backward(&self, x: &Expr) -> Expr {
        if x == self {
            return Expr::C(Constant { value: 1.0 });
        } else {
            return Expr::C(Constant { value: 0.0 });
        }
    }

    fn forward(&self) -> f64 {
        self.value.unwrap()
    }
}

impl Differentiable for SumExpr {
    fn backward(&self, x: &Expr) -> Expr {
        let left = self.left.backward(x);
        let right = self.right.backward(x);
        return SumExpr::new(Rc::new(left), Rc::new(right));
    }

    fn forward(&self) -> f64 {
        self.left.forward() + self.right.forward()
    }
}

impl Differentiable for MulExpr {
    fn backward(&self, x: &Expr) -> Expr {
        let left = MulExpr::new(Rc::new(self.left.backward(x)), Rc::clone(&self.right));
        let right = MulExpr::new(Rc::clone(&self.left), Rc::new(self.right.backward(x)));
        return SumExpr::new(Rc::new(left), Rc::new(right));
    }

    fn forward(&self) -> f64 {
        self.left.forward() * self.right.forward()
    }
}
