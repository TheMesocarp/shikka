use bytemuck::{Pod, Zeroable};
use mesocarp::logging::journal::Journal;

use crate::error::{ShikkaError, ShikkaResult};

pub trait ActionSpace {
    type Action: Copy + Pod + Zeroable;
    fn sample(&self) -> Self::Action;
}

pub trait StateSpace {
    type State: Copy + Pod + Zeroable;
    fn sample(&self) -> Self::State;
}

pub struct Environment<State: Clone + Pod + Zeroable, Action: Clone + Pod + Zeroable, F, T> 
where 
    F: Fn(&State) -> f32,
    T: Fn(&State, &Action) -> State
    {
    pub action_space: Box<dyn ActionSpace<Action = Action>>,
    pub state_space: Box<dyn StateSpace<State = State>>,
    pub logs: Journal,
    pub discount: f32,
    pub reward_fn: F,
    pub transition_fn: T,
    step_counter: u64, 
}

impl<State: Clone + Pod + Zeroable, Action: Clone + Pod + Zeroable, F, T> Environment<State, Action, F, T>
where 
    F: Fn(&State) -> f32,
    T: Fn(&State, &Action) -> State
    {
    pub fn new(
        states: impl StateSpace<State = State> + 'static,
        actions: impl ActionSpace<Action = Action> + 'static,
        initial_state: State,
        discount: f32,
        reward_fn: F,
        transition_fn: T
    ) -> ShikkaResult<Self> {
        if discount < 0.0 || discount > 1.0 {
            return Err(ShikkaError::DiscountFactorOutofBounds);
        }
        let state_space = Box::new(states);
        let action_space = Box::new(actions);

        let ssize = std::mem::size_of::<EnvLog<State, Action>>();
        let mut logs = Journal::init(ssize * 512);

        let init = EnvLog::<State, Action>::init(initial_state);
        logs.write(init, 0, None);
        Ok(Self {
            state_space,
            action_space,
            logs,
            discount,
            reward_fn,
            transition_fn,
            step_counter: 0
        })
    }

    pub fn state(&self) -> ShikkaResult<&State> {
        let state = &self.logs.read_state::<EnvLog<State, Action>>().map_err(ShikkaError::Mesocarp)?;
        Ok(&state.new_state)
    }
    
    pub fn env_time(&self) -> u64 {
        self.step_counter
    }

    pub fn step(&mut self, action: Action) -> ShikkaResult<()> {
        let current  = self.state()?;
        let new = (self.transition_fn)(current, &action);
        let reward = (self.reward_fn)(&new);
        let log = EnvLog::log(new, action, reward);
        self.logs.write(log, self.step_counter, None);
        self.step_counter += 1;
        Ok(())
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct EnvLog<State: Clone + Pod + Zeroable, Action: Clone + Pod + Zeroable> {
    new_state: State,
    reward: f32,
    action: Option<Action>,
}

impl<State: Clone + Pod + Zeroable, Action: Clone + Pod + Zeroable> EnvLog<State, Action> {
    fn new(new_state: State, action: Option<Action>, reward: f32) -> Self {
        Self {
            new_state,
            action,
            reward
        }
    }

    pub fn log(new_state: State, action: Action, reward: f32) -> Self {
        Self::new(new_state, Some(action), reward)
    }

    pub fn init(state: State) -> Self {
        Self::new(state, None, 0.0)
    }
}

unsafe impl<State: Clone + Pod + Zeroable, Action: Clone + Pod + Zeroable> Pod for EnvLog<State, Action> {}
unsafe impl<State: Clone + Pod + Zeroable, Action: Clone + Pod + Zeroable> Zeroable for EnvLog<State, Action> {}