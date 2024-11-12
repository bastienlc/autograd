/* TODO: be more general than f64 */

use std::rc::Rc;

/* Node object that holds the computation graph */
pub enum Node {
    C(Constant),
    V(Variable),
    S(Sum),
    M(Mul),
}

/* Concrete nodes */
pub struct Constant {
    pub value: f64,
}

pub struct Variable {
    pub name: String,
    pub value: Option<f64>,
}

/* Operations nodes */
pub struct Sum {
    pub left: Rc<Node>,
    pub right: Rc<Node>,
}

impl Sum {
    pub fn new(left: Rc<Node>, right: Rc<Node>) -> Node {
        Node::S(Sum {
            left: left,
            right: right,
        })
    }
}

pub struct Mul {
    pub left: Rc<Node>,
    pub right: Rc<Node>,
}

impl Mul {
    pub fn new(left: Rc<Node>, right: Rc<Node>) -> Node {
        Node::M(Mul {
            left: left,
            right: right,
        })
    }
}
