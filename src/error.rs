use mesocarp::MesoError;
use thiserror::Error;

pub type ShikkaResult<T> = Result<T, ShikkaError>;

#[derive(Debug, Error)]
pub enum ShikkaError {
    #[error("Discount factor is set outside of the [0.0, 1.0] interval, cannot proceed!")]
    DiscountFactorOutofBounds,
    #[error("Mesocarp Error: {0}")]
    Mesocarp(MesoError)
}