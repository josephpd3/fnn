pub trait ActivationLayer {

    fn get_activation(&mut self) -> Fn(f64) -> f64;

    fn get_derivative(&mut self) -> Fn(f64) -> f64;

}