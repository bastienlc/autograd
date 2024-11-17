/* TODO: be more general than f64 */

use std::rc::Rc;

/* Node object that holds the computation graph */
#[derive(Clone)]
pub enum Expr {
    C(Constant),
    V(Variable),
    S(SumExpr),
    M(MulExpr),
}

/* Concrete nodes */
#[derive(Clone)]
pub struct Constant {
    pub value: f64,
}

#[derive(Clone)]
pub struct Variable {
    pub name: String,
    pub value: Option<f64>,
}

/* Operations nodes */
#[derive(Clone)]
pub struct SumExpr {
    pub left: Rc<Expr>,
    pub right: Rc<Expr>,
}

impl SumExpr {
    pub fn new(left: Rc<Expr>, right: Rc<Expr>) -> Expr {
        Expr::S(SumExpr {
            left: left,
            right: right,
        })
    }
}

#[derive(Clone)]
pub struct MulExpr {
    pub left: Rc<Expr>,
    pub right: Rc<Expr>,
}

impl MulExpr {
    pub fn new(left: Rc<Expr>, right: Rc<Expr>) -> Expr {
        Expr::M(MulExpr {
            left: left,
            right: right,
        })
    }
}
