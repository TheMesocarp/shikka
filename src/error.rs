use mesocarp::MesoError;
use thiserror::Error;

pub type ShikkaResult<T> = Result<T, ShikkaError>;

#[derive(Debug, Error)]
pub enum ShikkaError {
    #[error("Discount factor is set outside of the (0.0, 1.0) interval, cannot proceed!")]
    DiscountFactorOutofBounds,
    #[error("epsilon is set outside of the [0.0, 1.0] interval, cannot proceed!")]
    EpsilonOutofBounds,
    #[error(
        "An empty list of valid actions was provided to the action sampler. The environment is now stuck."
    )]
    NoValidActionsProvided,
    #[error("`mesocarp` error: {0}")]
    Mesocarp(#[from] MesoError),
    #[error("`fastrand` sampling error caused by mismatched iterator lengths.")]
    Fastrand,
}
