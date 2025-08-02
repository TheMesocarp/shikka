use bytemuck::{Pod, Zeroable};

use crate::{error::{ShikkaError, ShikkaResult}, poptim::Policy};

pub(crate) fn epsilon_greedy<State: Copy + Pod + Zeroable, Action: Copy + Pod + Zeroable, P: Policy<State, Action> + 'static>(p: &P, state: &State, mut valid: Vec<Action>, epsilon: f32) -> ShikkaResult<Action> {
    if epsilon < 0.0 || epsilon > 1.0 {
        return Err(ShikkaError::EpsilonOutofBounds)
    }
    let rand = fastrand::i32(..10i32.pow(6));
    let rand = rand as f32 / 10i32.pow(6) as f32;
    if rand < epsilon {
        let mut out = valid.pop().ok_or(ShikkaError::NoValidActionsProvided)?;
        let mut old  = p.prob(state, &out);
        for i in 0..valid.len() {
            let prob = p.prob(state, &valid[i]);
            if prob > old {
                old = prob;
                out = valid[i]
            }
        }
        return Ok(out)
    }
    fastrand::choice(valid).ok_or(ShikkaError::Fastrand)

}

pub enum SamplingStrategy {
    Greedy(f32),
    Custom, 
}

pub(crate) trait Sample<State: Copy + Pod + Zeroable, Action: Copy + Pod + Zeroable> {
    fn sample<P: Policy<State, Action> + 'static>(&self, policy: &P, state: &State, valid: Vec<Action>) -> ShikkaResult<Action>;
}

impl<State: Copy + Pod + Zeroable, Action: Copy + Pod + Zeroable> Sample<State, Action> for SamplingStrategy {
    fn sample<P: Policy<State, Action> + 'static>(&self, policy: &P, state: &State, valid: Vec<Action>) -> ShikkaResult<Action> {
        Ok(match self {
            SamplingStrategy::Greedy(epsilon) => {
                epsilon_greedy(policy, state, valid, *epsilon)?
            },
            SamplingStrategy::Custom => {
                policy.sample(state, valid)
            },
        })
    }
}