use crate::{env::{ActionSpace, Environment, StateSpace}, popt::Policy};

pub struct TRPO<State, Action, F, T> 
where 
    F: Fn(&State) -> f32,
    T: Fn(&State, &Action) -> State 
{
    env: Environment<State, Action, F, T>,
    policy: Box<dyn Policy<State, Action>>,
    states: Box<dyn StateSpace<State = State>>,
    actions: Box<dyn ActionSpace<Action = Action>>
}