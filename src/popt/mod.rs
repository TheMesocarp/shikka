pub mod trpo;

pub trait Policy<State, Action> {
    fn sample(&self, state: &State) -> Action;
    fn policy(&self, state: &State, action: &Action) -> f32; // default f32 policy precision for now
}
