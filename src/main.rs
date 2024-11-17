use autodiff::differentiable::Differentiable;
use autodiff::objects::{Constant, Expr, Variable};

fn main() {
    let a = Expr::V(Variable {
        name: "a".to_string(),
        value: Some(1.0),
    });
    let b = Expr::V(Variable {
        name: "b".to_string(),
        value: Some(2.0),
    });
    let c = Expr::V(Variable {
        name: "c".to_string(),
        value: Some(3.0),
    });
    let two = Expr::C(Constant { value: 2.0 });

    let x = (a.clone() + b.clone() + two) * (a.clone() + b.clone() + c.clone());

    println!("x = {}", x);
    println!("value = {}", x.forward());
    println!("d(x)/d(a) = {}", x.backward(&a));
    println!("d(x)/d(b) = {}", x.backward(&b));
    println!("d(x)/d(c) = {}", x.backward(&c));
}
