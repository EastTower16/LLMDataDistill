

# from typing import Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

from jax.stages import Lowered
from australis import exporter


class MarketingDetectionModel(nn.Module):
  vocab_size: int
  hidden_size: int
  @nn.compact
  def __call__(self, x, training=False):
    x = nn.Embed(num_embeddings=self.vocab_size, features=self.hidden_size)(x)
    # now [batch, seqlen, hidden_size]

    x1 = nn.Conv(features=32, kernel_size=(3,),padding='VALID')(x)
    x2 = nn.Conv(features=32, kernel_size=(5,),padding='VALID')(x)
    x3 = nn.Conv(features=32, kernel_size=(7,),padding='VALID')(x)
    x4 = nn.Conv(features=32, kernel_size=(9,),padding='VALID')(x)
    x1 = nn.gelu(x1)
    x2 = nn.gelu(x2)
    x3 = nn.gelu(x3)
    x4 = nn.gelu(x4)
    x1 = nn.Conv(features=32, kernel_size=(9,),padding='VALID')(x1)
    x2 = nn.Conv(features=32, kernel_size=(7,),padding='VALID')(x2)
    x3 = nn.Conv(features=32, kernel_size=(5,),padding='VALID')(x3)
    x4 = nn.Conv(features=32, kernel_size=(3,),padding='VALID')(x4)
    x1 = nn.elu(x1)
    x2 = nn.elu(x2)
    x3 = nn.elu(x3)
    x4 = nn.elu(x4)
    x= jnp.concatenate([x1,x2,x3,x4], axis=2)
    x = nn.avg_pool(x, window_shape=(2,), strides=(2,))

    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dropout(rate=0.25)(x,deterministic=not training)
    x = nn.Dense(features=512)(x)
    x = nn.gelu(x)
    x = nn.Dense(features=1)(x)
    return x


def lower() -> Lowered:
  model = MarketingDetectionModel(vocab_size=51200, hidden_size=256)

  tx = optax.adam(0.001)

  @jax.jit
  def init():
    init_rng = jax.random.PRNGKey(42)
    params = model.init(init_rng, jnp.ones((128, 2048), dtype=jnp.int32))
    return params, tx.init(params)

  init_fn = init.lower()

  @jax.jit
  def serving(params,x):
    return model.apply(params, x, False)

  @jax.jit
  def batchServing(params,x):
    return model.apply(params, x, False)
  


  @jax.jit
  def optimizer_step(params, opt_state, x,y):
    def fwd(params):
      logits = model.apply(params, x, True,rngs={'dropout': jax.random.PRNGKey(42)})
      return optax.sigmoid_binary_cross_entropy(logits,y).mean()
      

    loss,grads = jax.value_and_grad(fwd)(params)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

  params, opt_state = jax.eval_shape(init)
  optimizer_step_lowered = optimizer_step.lower(
      params, opt_state, jax.ShapeDtypeStruct((128, 2048), jnp.int32),jax.ShapeDtypeStruct((128, 1), jnp.int32))
  
  serving_lowered = serving.lower(
      params, jax.ShapeDtypeStruct((1, 2048), jnp.int32))
  
  batch_serving_lowered = batchServing.lower(
      params, jax.ShapeDtypeStruct((128, 2048), jnp.int32))

  return [
      ("flax_init", init_fn),
      ("flax_optimizer_step", optimizer_step_lowered),
      ("flax_serving",serving_lowered),
      ("flax_batch_serving",batch_serving_lowered),
  ]


if __name__ == "__main__":
  exporter.run(lower)
