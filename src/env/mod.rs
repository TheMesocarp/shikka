use crate::error::{ShikkaError, ShikkaResult};

pub trait ActionSpace {
    type Action;
    fn sample(&self) -> Self::Action;
}

pub trait StateSpace {
    type State;
    fn sample(&self) -> Self::State;
}

pub struct Environment<State, Action, F, T> 
where 
    F: Fn(&State) -> f32,
    T: Fn(&State, &Action) -> State
    {
    pub last_action: Option<Action>,
    pub state: State,
    pub discount: f32,
    pub reward_fn: F, // No Box needed
    pub transition_fn: T,
}

impl<State, Action, F, T> Environment<State, Action, F, T>
where 
    F: Fn(&State) -> f32,
    T: Fn(&State, &Action) -> State
    {
    pub fn new(
        initial_state: State,
        discount: f32,
        reward_fn: F,
        transition_fn: T
    ) -> ShikkaResult<Self> {
        if discount < 0.0 || discount > 1.0 {
            return Err(ShikkaError::DiscountFactorOutofBounds);
        }
        Ok(Self {
            last_action: None,
            discount,
            state: initial_state,
            reward_fn,
            transition_fn
        })
    }

    pub fn step(&mut self, action: Action) -> (&State, f32) {
        self.state = (self.transition_fn)(&self.state, &action);
        self.last_action = Some(action);
        let reward = (self.reward_fn)(&self.state);
        (&self.state, reward)
    }
}