use bytemuck::{Pod, Zeroable};

use crate::{
    error::{ShikkaError, ShikkaResult},
    actors::Policy,
};

pub(crate) fn epsilon_greedy_policy<
    State: Copy + Pod + Zeroable,
    Action: Copy + Pod + Zeroable,
    P: Policy<State, Action> + 'static,
>(
    p: &P,
    state: &State,
    mut valid: Vec<Action>,
    epsilon: f32,
) -> ShikkaResult<Action> {
    if epsilon < 0.0 || epsilon > 1.0 {
        return Err(ShikkaError::EpsilonOutofBounds);
    }
    let rand = fastrand::i32(..10i32.pow(6));
    let rand = rand as f32 / 10i32.pow(6) as f32;
    if rand < epsilon {
        let mut out = valid.pop().ok_or(ShikkaError::NoValidActionsProvided)?;
        let mut old = p.prob(state, &out);
        for i in 0..valid.len() {
            let prob = p.prob(state, &valid[i]);
            if prob > old {
                old = prob;
                out = valid[i]
            }
        }
        return Ok(out);
    }
    fastrand::choice(valid).ok_or(ShikkaError::Fastrand)
}

pub(crate) fn epsilon_greedy_reward<
    State: Copy + Pod + Zeroable,
    Action: Copy + Pod + Zeroable,
    R: Fn(&State) -> f32,
    T: Fn(&State, &Action) -> State,
>(
    state: &State,
    mut valid: Vec<Action>,
    epsilon: f32,
    reward_fn: &R,
    transition_fn: &T,
) -> ShikkaResult<Action> {
    if epsilon < 0.0 || epsilon > 1.0 {
        return Err(ShikkaError::EpsilonOutofBounds);
    }
    let rand = fastrand::i32(..10i32.pow(6));
    let rand = rand as f32 / 10i32.pow(6) as f32;
    if rand < epsilon {
        let mut out = valid.pop().ok_or(ShikkaError::NoValidActionsProvided)?;
        let mut old = (reward_fn)(&(transition_fn)(state, &out));
        for i in 0..valid.len() {
            let new = (reward_fn)(&(transition_fn)(state, &valid[i]));
            if new > old {
                old = new;
                out = valid[i]
            }
        }
        let _ = valid;
        return Ok(out);
    }
    fastrand::choice(valid).ok_or(ShikkaError::Fastrand)
}

pub enum GreedyMode {
    Reward,
    Policy,
}

pub enum Strategy {
    EpsilonGreedy(f32, GreedyMode),
    Custom,
}

impl Strategy {
    pub(crate) fn sample<
        State: Copy + Pod + Zeroable,
        Action: Copy + Pod + Zeroable,
        P: Policy<State, Action> + 'static,
        R: Fn(&State) -> f32,
        T: Fn(&State, &Action) -> State,
    >(
        &self,
        policy: &P,
        state: &State,
        valid: Vec<Action>,
        reward_fn: &R,
        transition_fn: &T,
    ) -> ShikkaResult<Action> {
        Ok(match self {
            Strategy::EpsilonGreedy(epsilon, mode) => match mode {
                GreedyMode::Reward => epsilon_greedy_reward(state, valid, *epsilon, reward_fn, transition_fn)?,
                GreedyMode::Policy => epsilon_greedy_policy(policy, state, valid, *epsilon)?,
            },
            Strategy::Custom => policy.sample(state, valid),
        })
    }
}
