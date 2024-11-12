use crate::objects::{Constant, Mul, Node, Sum, Variable};

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Node::C(c1), Node::C(c2)) => c1 == c2,
            (Node::V(v1), Node::V(v2)) => v1 == v2,
            (Node::S(s1), Node::S(s2)) => s1 == s2,
            (Node::M(m1), Node::M(m2)) => m1 == m2,
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

impl PartialEq for Sum {
    fn eq(&self, other: &Self) -> bool {
        self.left == other.left && self.right == other.right
    }
}

impl PartialEq for Mul {
    fn eq(&self, other: &Self) -> bool {
        self.left == other.left && self.right == other.right
    }
}

impl PartialEq<Constant> for Node {
    fn eq(&self, other: &Constant) -> bool {
        match self {
            Node::C(c) => c == other,
            _ => false,
        }
    }
}

impl PartialEq<Variable> for Node {
    fn eq(&self, other: &Variable) -> bool {
        match self {
            Node::V(v) => v == other,
            _ => false,
        }
    }
}

impl PartialEq<Sum> for Node {
    fn eq(&self, other: &Sum) -> bool {
        match self {
            Node::S(s) => s == other,
            _ => false,
        }
    }
}

impl PartialEq<Mul> for Node {
    fn eq(&self, other: &Mul) -> bool {
        match self {
            Node::M(m) => m == other,
            _ => false,
        }
    }
}
