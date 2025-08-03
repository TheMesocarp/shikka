use bytemuck::{Pod, Zeroable};

use crate::{actors::Policy, env::Environment};

pub struct TRPO<State: Copy + Pod + Zeroable, Action: Copy + Pod + Zeroable, F, T>
where
    F: Fn(&State) -> f32,
    T: Fn(&State, &Action) -> State,
{
    _env: Environment<State, Action, F, T>,
    _policy: Box<dyn Policy<State, Action>>,
}
