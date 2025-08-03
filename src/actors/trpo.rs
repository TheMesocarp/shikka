use bytemuck::{Pod, Zeroable};

use crate::{env::Environment, actors::Policy};

pub struct TRPO<State: Copy + Pod + Zeroable, Action: Copy + Pod + Zeroable, F, T> 
where 
    F: Fn(&State) -> f32,
    T: Fn(&State, &Action) -> State 
{
    env: Environment<State, Action, F, T>,
    policy: Box<dyn Policy<State, Action>>,
    
}