pub struct Model<T> {
    pub dataset: T,
    pub layers: Vec<Box<Layers>>,
    pub loss: Box<Loss>,
    pub optimizer: Box<Optimizer>,
}