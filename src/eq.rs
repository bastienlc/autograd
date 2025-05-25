use crate::objects::Tensor;

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.get_shape() == other.get_shape() && self.get_data() == other.get_data()
    }
}
