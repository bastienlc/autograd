use autodiff::backward::Backward;
use autodiff::objects::Tensor;
use autodiff::operations::reduce_sum;

fn main() {
    let a = Tensor::new(
        vec![2, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        true,
        None,
        None,
    );
    let b = Tensor::new(
        vec![2, 3],
        vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        true,
        None,
        None,
    );

    let c = a.clone() + b.clone();

    let mut d = reduce_sum(c.clone());
    d.backward(None);

    println!("a {}", a);
    println!("b {}", b);
    println!("c {}", c);
    println!("d {}", d);
    println!("d_grad {}", d.grad().unwrap());
    println!("c_grad {}", c.grad().unwrap());
    println!("b_grad {}", b.grad().unwrap());
    println!("a_grad {}", a.grad().unwrap());
}
