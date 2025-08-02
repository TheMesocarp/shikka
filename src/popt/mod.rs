pub mod trpo;

pub trait Policy<State, Action> {
    fn sample(&self, state: &State) -> Action;
    fn log_prob(&self, state: &State, action: &Action) -> f32; // default f32 log_prob precision for now
}
