use bytemuck::{Pod, Zeroable};
use mesocarp::logging::journal::Journal;

use crate::{
    error::{ShikkaError, ShikkaResult},
    actors::Policy,
    sampler::Strategy,
};

pub trait ActionSpace<State, Action> {
    fn sample(&self) -> Action;
    fn valid(&self, state: &State) -> Vec<Action>;
    fn contains(&self, action: &Action) -> bool;
}

pub trait StateSpace<State, Action> {
    fn sample(&self) -> State;
    fn reachable(&self, actions: &[Action]) -> Vec<State>;
    fn contains(&self, state: &State) -> bool;
}

pub struct Environment<State: Copy + Pod + Zeroable, Action: Copy + Pod + Zeroable, F, T>
where
    F: Fn(&State) -> f32,
    T: Fn(&State, &Action) -> State,
{
    pub action_space: Box<dyn ActionSpace<State, Action>>,
    pub state_space: Box<dyn StateSpace<State, Action>>,
    pub sampler: Strategy,
    pub logs: Journal,
    pub discount: f32,
    pub reward_fn: F,
    pub transition_fn: T,
    pub step_counter: u64,
}

impl<State: Copy + Pod + Zeroable, Action: Copy + Pod + Zeroable, F, T>
    Environment<State, Action, F, T>
where
    F: Fn(&State) -> f32,
    T: Fn(&State, &Action) -> State,
{
    pub fn new(
        states: impl StateSpace<State, Action> + 'static,
        actions: impl ActionSpace<State, Action> + 'static,
        initial_state: State,
        sampler: Strategy,
        discount: f32,
        reward_fn: F,
        transition_fn: T,
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
            sampler,
            logs,
            discount,
            reward_fn,
            transition_fn,
            step_counter: 0,
        })
    }

    pub fn state(&self) -> ShikkaResult<&State> {
        let state = &self.logs.read_state::<EnvLog<State, Action>>()?;
        Ok(&state.new_state)
    }

    pub fn get_last_action(&self) -> ShikkaResult<&Option<Action>> {
        let state = &self.logs.read_state::<EnvLog<State, Action>>()?;
        Ok(&state.action)
    }

    pub fn get_last_reward(&self) -> ShikkaResult<f32> {
        let state = &self.logs.read_state::<EnvLog<State, Action>>()?;
        Ok(state.reward)
    }

    pub fn env_time(&self) -> u64 {
        self.step_counter
    }

    pub fn step(&mut self, action: Action) -> ShikkaResult<()> {
        let current = self.state()?;
        let new = (self.transition_fn)(current, &action);
        let reward = (self.reward_fn)(&new);
        let log = EnvLog::log(new, action, reward);
        self.logs.write(log, self.step_counter, None);
        self.step_counter += 1;
        Ok(())
    }

    pub fn trajectory(&mut self) -> Vec<&EnvLog<State, Action>> {
        self.logs
            .read_all::<EnvLog<State, Action>>()
            .into_iter()
            .map(|x| x.0)
            .collect::<Vec<_>>()
    }

    pub fn cleanup(&mut self) -> Vec<EnvLog<State, Action>> {
        self.logs
            .cleanup::<EnvLog<State, Action>>()
            .into_iter()
            .map(|x| x.0)
            .collect::<Vec<_>>()
    }

    pub fn sample_forward_trajectory<P: Policy<State, Action> + 'static>(
        &self,
        horizon: usize,
        policy: P,
    ) -> ShikkaResult<Vec<EnvLog<State, Action>>> {
        let mut out = vec![];
        for _ in 0..horizon {
            let state = self.state()?;
            let valid = self.action_space.valid(state);
            let action =
                self.sampler
                    .sample(&policy, state, valid, &self.reward_fn, &self.transition_fn)?;
            let state = (self.transition_fn)(state, &action);
            let reward = (self.reward_fn)(&state);
            out.push(EnvLog::new(state, Some(action), reward))
        }
        Ok(out)
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct EnvLog<State: Copy + Pod + Zeroable, Action: Copy + Pod + Zeroable> {
    new_state: State,
    reward: f32,
    action: Option<Action>,
}

impl<State: Copy + Pod + Zeroable, Action: Copy + Pod + Zeroable> EnvLog<State, Action> {
    fn new(new_state: State, action: Option<Action>, reward: f32) -> Self {
        Self {
            new_state,
            action,
            reward,
        }
    }

    pub fn log(new_state: State, action: Action, reward: f32) -> Self {
        Self::new(new_state, Some(action), reward)
    }

    pub fn init(state: State) -> Self {
        Self::new(state, None, 0.0)
    }
}

unsafe impl<State: Copy + Pod + Zeroable, Action: Copy + Pod + Zeroable> Pod
    for EnvLog<State, Action>
{
}
unsafe impl<State: Copy + Pod + Zeroable, Action: Copy + Pod + Zeroable> Zeroable
    for EnvLog<State, Action>
{
}
