use bytemuck::{Pod, Zeroable};

pub mod trpo;

pub trait Policy<State: Copy + Pod + Zeroable, Action: Copy + Pod + Zeroable> {
    fn sample(&self, state: &State, actions: Vec<Action>) -> Action;
    fn prob(&self, state: &State, action: &Action) -> f32;
}
