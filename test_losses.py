import os
import sys
import jax
import jax.numpy as jnp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from embodied.jax.outs import MSE, L1

def test_losses():
    mean = jnp.array([[[0.5, 0.2], [0.1, 0.8]]])
    target = jnp.array([[[0.4, 0.4], [0.1, 0.9]]])

    mse = MSE(mean)
    l1 = L1(mean)

    mse_loss = mse.loss(target)
    l1_loss = l1.loss(target)

    print("Target:", target)
    print("Mean:", mean)
    print("MSE Loss:", mse_loss)
    print("L1 Loss:", l1_loss)

    # Check correctness
    expected_mse = jnp.square(mean - target)
    expected_l1 = jnp.abs(mean - target)

    assert jnp.allclose(mse_loss, expected_mse)
    assert jnp.allclose(l1_loss, expected_l1)
    print("Assertions passed!")

if __name__ == '__main__':
    test_losses()
