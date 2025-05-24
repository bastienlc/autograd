use crate::objects::Tensor;

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.shape() == other.shape() && self.data() == other.data()
    }
}
