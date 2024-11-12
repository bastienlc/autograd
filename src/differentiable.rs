use crate::objects::{Constant, Mul, Node, Sum, Variable};
use std::rc::Rc;
pub trait Differentiable {
    fn backward(&self, x: &Node) -> Node;
    fn forward(&self) -> f64;
}

impl Differentiable for Node {
    /* We need to implement the trait for the enum Node */
    /* We just need to call each implementation for each type */
    fn backward(&self, x: &Node) -> Node {
        match self {
            Node::C(c) => c.backward(x),
            Node::V(v) => v.backward(x),
            Node::S(s) => s.backward(x),
            Node::M(m) => m.backward(x),
        }
    }

    fn forward(&self) -> f64 {
        match self {
            Node::C(c) => c.forward(),
            Node::V(v) => v.forward(),
            Node::S(s) => s.forward(),
            Node::M(m) => m.forward(),
        }
    }
}

impl Differentiable for Constant {
    fn backward(&self, _x: &Node) -> Node {
        Node::C(Constant { value: 0.0 })
    }

    fn forward(&self) -> f64 {
        self.value
    }
}

impl Differentiable for Variable {
    fn backward(&self, x: &Node) -> Node {
        if x == self {
            return Node::C(Constant { value: 1.0 });
        } else {
            return Node::C(Constant { value: 0.0 });
        }
    }

    fn forward(&self) -> f64 {
        self.value.unwrap()
    }
}

impl Differentiable for Sum {
    fn backward(&self, x: &Node) -> Node {
        let left = self.left.backward(x);
        let right = self.right.backward(x);
        return Sum::new(Rc::new(left), Rc::new(right));
    }

    fn forward(&self) -> f64 {
        self.left.forward() + self.right.forward()
    }
}

impl Differentiable for Mul {
    fn backward(&self, x: &Node) -> Node {
        let left = Mul::new(Rc::new(self.left.backward(x)), Rc::clone(&self.right));
        let right = Mul::new(Rc::clone(&self.left), Rc::new(self.right.backward(x)));
        return Sum::new(Rc::new(left), Rc::new(right));
    }

    fn forward(&self) -> f64 {
        self.left.forward() * self.right.forward()
    }
}
