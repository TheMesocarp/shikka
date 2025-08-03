// need to implement common value, action-value, and advantage critics.

use bytemuck::{Pod, Zeroable};

use crate::error::ShikkaResult;

pub mod dae;

pub trait Value<State: Copy + Pod + Zeroable> {
    fn value(&self, state: &State) -> f32;
    fn update(&mut self, state: &State, target: f32) -> ShikkaResult<()>;
}

pub trait ActionValue<State: Copy + Pod + Zeroable, Action: Copy + Pod + Zeroable> {
    fn q_value(&self, state: &State, action: &Action) -> f32;
    fn update(&mut self, state: &State, action: &Action, target: f32) -> ShikkaResult<()>;
    fn greedy(&self, state: &State, valid_actions: &[Action]) -> Option<Action> {
        valid_actions
            .iter()
            .max_by(|a, b| {
                self.q_value(state, a)
                    .partial_cmp(&self.q_value(state, b))
                    .unwrap()
            })
            .copied()
    }
}

pub trait Advantage<State: Copy + Pod + Zeroable, Action: Copy + Pod + Zeroable> {
    fn advantage(&self, state: &State, action: &Action) -> f32;
    fn update(&mut self, state: &State, action: &Action, target: f32) -> ShikkaResult<()>;
}
