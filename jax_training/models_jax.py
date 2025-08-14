import math
import string
import jax
import jax.numpy as jnp
import numpy as _np

# ------------------------------------------------------------------
# NumPy ≥ 2.0 compatibility shim for libraries referencing ComplexWarning
# ------------------------------------------------------------------
if not hasattr(_np, "ComplexWarning"):
    class _ComplexWarning(Warning):
        pass
    _np.ComplexWarning = _ComplexWarning  # type: ignore[attr-defined]

import tensorcircuit as tc
from flax import linen as nn

tc.set_backend("jax")

# ===============================================================
# Fourier Neural Network (FNN) using Flax and JAX
# ===============================================================
class FNN(nn.Module):
    num_features: int = 8            # number of input features
    num_frequencies: int = 1         # number of sin/cos frequencies per feature
    num_output: int = 1              # output dimension
    init_std: float = 0.1            # stddev for weight init
    frequency_min_init: float = 1.0  # initial scale for input frequencies
    trainable_frequency_min: bool = True

    @nn.compact
    def __call__(self, x):
        batch_size = x.shape[0]

        # --- 1) Frequency scaling parameter ---
        if self.trainable_frequency_min:
            frequency_min = self.param(
                "frequency_min",
                lambda rng: jnp.array(self.frequency_min_init, dtype=x.dtype),
            )
        else:
            frequency_min = jnp.array(self.frequency_min_init, dtype=x.dtype)
        x_scaled = x * frequency_min  # shape: (B, num_features)

        # --- 2) Build Fourier features ---
        # physical feature dimension per input: 1 + 2*num_frequencies
        D_phys = 1 + 2 * self.num_frequencies

        # prepare frequencies [1, 2, ..., num_frequencies]
        frequencies = jnp.arange(1, self.num_frequencies + 1, dtype=x.dtype)

        # compute sin and cos features: shapes (B, num_features, num_frequencies)
        sin_feats = jnp.sin(x_scaled[..., None] * frequencies)
        cos_feats = jnp.cos(x_scaled[..., None] * frequencies)

        # constant '1' channel: shape (B, num_features, 1)
        ones = jnp.ones((batch_size, self.num_features, 1), dtype=x.dtype)

        # concatenate into (B, num_features, D_phys)
        features = jnp.concatenate([ones, sin_feats, cos_feats], axis=-1)

        # --- 3) Learnable tensor of shape (D_phys, D_phys, ..., D_phys, num_output) ---
        weight_shape = (D_phys,) * self.num_features + (self.num_output,)
        W = self.param(
            "W",
            nn.initializers.normal(stddev=self.init_std),
            weight_shape,
        )
        bias = self.param("bias", nn.initializers.zeros, (self.num_output,))

        # helper to contract a single sample's features through the high-order tensor
        def contract_single(sample_features):
            # sample_features: (num_features, D_phys)
            tensor = W
            # sequentially contract each feature mode
            for i in range(self.num_features):
                tensor = jnp.tensordot(sample_features[i], tensor, axes=([0], [0]))
            # result has shape (num_output,)
            return tensor

        # apply to each element in the batch
        result = jax.vmap(contract_single)(features)  # shape: (B, num_output)

        return result + bias


import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

# ===============================================================
# Quantum Circuit Function
# ===============================================================
def quantum_circuit(x, weights, entanglement_weights, final_rotations, num_qubits, layer_depth, num_frequencies, ad = 0, pd = 0, dp = 0):
    """
    Build a quantum circuit using tensorcircuit, with parameterized RXX entanglement and final rotations.

    Args:
      x: JAX array of shape (num_qubits,) representing the input.
      weights: Array of shape (num_frequencies, layer_depth, num_qubits, 3) for single-qubit rotations.
      entanglement_weights: Array of shape (num_frequencies, layer_depth, num_qubits-1) for RXX angles.
      final_rotations: Array of shape (num_qubits, 3) for the final variational rotation layer.
      num_qubits: Number of qubits.
      layer_depth: Number of variational layers per reuploading block.
      num_frequencies: Number of reuploading blocks.

    Returns:
      JAX array of shape (num_qubits,) with the expectation values of PauliZ.
    """
    import tensorcircuit as tc  # assuming tensorcircuit is available
    c = tc.Circuit(num_qubits)
    
    # Tensorcircuit circuit simulated as density matrix
    if ad+pd+dp > 0:
        c = tc.DMCircuit( num_qubits )
    else:
        c = tc.Circuit( num_qubits )
        
    for f in range(num_frequencies):
        # Input encoding
        for j in range(num_qubits):
            c.rx(j, theta=x[j])
            apply_noise(c, [j], ad, pd, dp)
            
        # Variational layers
        for k in range(layer_depth):
            for j in range(num_qubits):
                theta, phi, alpha = weights[f, k, j]
                c.r(j, theta=theta, phi=phi, alpha=alpha)
                apply_noise(c, [j], ad, pd, dp)
                
            for j in range(num_qubits - 1):
                theta_ent = entanglement_weights[f, k, j]
                c.rxx(j, j + 1, theta=theta_ent)
                apply_noise(c, [j, j + 1], ad, pd, dp)
                
    # Final rotation layer
    for j in range(num_qubits):
        theta, phi, alpha = final_rotations[j]
        c.r(j, theta=theta, phi=phi, alpha=alpha)
        apply_noise(c, [j], ad, pd, dp)
        
#     out = [c.expectation_ps(z=[j]) for j in range(num_qubits)]
    out = [c.expectation_ps(z=[j]) for j in range(1)]
    return jnp.real(jnp.array(out))

def apply_noise(circ, inds, ad, pd, dp):
    '''
    Apply noise model to qubits at inds
    '''
    for i in inds:
        
        if ad > 0:
            circ.amplitudedamping(i, gamma=ad, p=1)
        if pd > 0:
            circ.phasedamping(i, gamma=pd)
        if dp > 0:
            circ.depolarizing(i, px = dp/3, py = dp/3, pz = dp/3)
            
# ===============================================================
# Quantum Neural Network (QNN) with tensorcircuit
# ===============================================================
class QNN(nn.Module):
    num_features: int = 8         # classical input; num_qubits computed automatically
    num_frequencies: int = 1      
    layer_depth: int = 1          
    num_output: int = 1
    init_std: float = 0.1       # used in final dense layer
    init_std_Q: float = 0.1       # used for quantum circuit weights
    frequency_min_init: float = 1.0  
    trainable_frequency_min: bool = True
    ad: float = 0.0  
    pd: float = 0.0  
    dp: float = 0.0  
    
    @property
    def num_qubits(self):
        # In this design, we assume one qubit per classical feature.
        return self.num_features

    @nn.compact
    def __call__(self, x):
        # x: shape (batch_size, num_qubits)
        num_qubits = self.num_qubits
        # Frequency scaling.
        frequency_min = (
            self.param("frequency_min", lambda rng: jnp.array(self.frequency_min_init, dtype=x.dtype))
            if self.trainable_frequency_min else self.frequency_min_init
        )
        x_scaled = x * frequency_min

        # Define shapes for the combined weights.
        shape_weights = (self.num_frequencies, self.layer_depth, num_qubits, 3)
        shape_entanglement = (self.num_frequencies, self.layer_depth, num_qubits - 1)
        shape_final = (num_qubits, 3)

        # Compute total sizes as Python integers.
        size_weights = int(np.prod(shape_weights))
        size_entanglement = int(np.prod(shape_entanglement))
        size_final = int(np.prod(shape_final))
        total_size = size_weights + size_entanglement + size_final

        # Combined parameter for all quantum circuit weights.
        quanum_weights = self.param(
            "quanum_weights",
            lambda rng, shape: jax.random.normal(rng, shape) * self.init_std_Q,
            (total_size,)
        )

        # Slice and reshape to recover individual parameters.
        weights = jnp.reshape(quanum_weights[:size_weights], shape_weights)
        entanglement_weights = jnp.reshape(
            quanum_weights[size_weights:size_weights + size_entanglement], shape_entanglement
        )
        final_rotations = jnp.reshape(
            quanum_weights[size_weights + size_entanglement:], shape_final
        )

        # Build the quantum circuit output (vectorized over the batch).
        circuit_out = jax.vmap(
            lambda single_x: quantum_circuit(
                single_x,
                weights,
                entanglement_weights,
                final_rotations,
                num_qubits,
                self.layer_depth,
                self.num_frequencies,
                self.ad,
                self.pd,
                self.dp
            )
        )(x_scaled)
        # Use a dense layer on the circuit output.
        output = nn.Dense(self.num_output,
                          kernel_init=nn.initializers.normal(stddev=self.init_std)
                         )(circuit_out)
        return output
    
    
    
    
# ===============================================================
# Kingston-style noise model (medians, July 2025 dashboard)
# ===============================================================

import math
import jax.numpy as jnp
import numpy as np
import tensorcircuit as tc

# ─── Device-wide constants (all taken from your screenshot) ────
T1_MEDIAN = 224.38e-6           # 224.38 µs  ⇒ 224.38×10⁻⁶ s
T2_MEDIAN = 120.53e-6           # 120.53 µs
EPC_1Q    = 2.335e-4            # median SX error  (used for *all* 1-q gates)
EPC_2Q    = 2.137e-3            # median CZ error  (used for RZZ)

# order-of-magnitude pulse lengths for Heron r2 processors
GATE_DUR = {                    # seconds
    "rx": 35e-9,                # 35 ns   (≈ sx length)
    "rz": 0.0,                  # virtual Z (frame shift → no deco)
    "sx": 35e-9,
    "x" : 70e-9,
    "rzz": 250e-9               # taken ~ CZ duration
}

# ---------------------------------------------------------------
# Helper: thermal-relaxation γ’s for a *single* gate of duration t
# ---------------------------------------------------------------
def _relaxation_gammas(t):
    """Return (γ_AD, γ_PD) for a gate that lasts `t` seconds."""
    if t == 0.0:
        return 0.0, 0.0
    g_ad = 1.0 - math.exp(-t / T1_MEDIAN)   # amplitude damping
    g_pd = 1.0 - math.exp(-t / T2_MEDIAN)   # phase damping
    return g_ad, g_pd


# ---------------------------------------------------------------
# Noise after a *physical* gate
# ---------------------------------------------------------------
def apply_gate_noise(circ, qubits, gate_label):
    """
    Add depolarising + thermal-relaxation noise that corresponds to one
    *physical* Kingston gate.
    """
    # 1) Depolarising channel
    if gate_label == "rzz":
        p = EPC_2Q
    else:
        p = EPC_1Q

    for q in qubits:
        circ.depolarizing(q, px=p/3, py=p/3, pz=p/3)

    # 2) Thermal-relaxation channel
    g_ad, g_pd = _relaxation_gammas(GATE_DUR.get(gate_label, 0.0))
    if g_ad > 0 or g_pd > 0:
        for q in qubits:
            if g_ad > 0:
                circ.amplitudedamping(q, gamma=g_ad, p=1.0)
            if g_pd > 0:
                circ.phasedamping(q, gamma=g_pd)


# ---------------------------------------------------------------
# Noise for an *idle* period of length t_wait (seconds)
# ---------------------------------------------------------------
def apply_idle_noise(circ, qubits, t_wait):
    """Apply only thermal-relaxation noise (no depolarising) for idle time."""
    if t_wait <= 0:
        return
    g_ad, g_pd = _relaxation_gammas(t_wait)
    for q in qubits:
        if g_ad > 0:
            circ.amplitudedamping(q, gamma=g_ad, p=1.0)
        if g_pd > 0:
            circ.phasedamping(q, gamma=g_pd)


# ===============================================================
# Quantum circuit with Kingston native gates + realistic noise
# ===============================================================
def quantum_circuit_ibm(
    x,
    weights,
    entanglement_weights,
    final_rotations,
    num_qubits,
    layer_depth,
    num_frequencies,
    t_wait=0.0,                       # ❶ NEW hyper-parameter (seconds)
):
    """
    • RXX → RZZ
    • R(θ,φ,α) → Rz(φ) Rx(θ) Rz(α)      (all three get noise)
    • After *every* physical gate → depol + relax
    • After encoding Rx, after each full Rz-Rx-Rz block, and after each RZZ
      → extra idle-time relaxation of length `t_wait`
    """
    c = tc.DMCircuit(num_qubits)      # density-matrix simulation

    qubit_range = list(range(num_qubits))

    for f in range(num_frequencies):

        # ── Input encoding ───────────────────────────────────────
        for q in qubit_range:
            c.rx(q, theta=x[q])
            apply_gate_noise(c, [q], "rx")
        apply_idle_noise(c, qubit_range, t_wait)

        # ── Variational layers ──────────────────────────────────
        for k in range(layer_depth):

            # ••• Local rotations •••
            for q in qubit_range:
                θ, φ, α = weights[f, k, q]

                c.rz(q, theta=φ)
                apply_gate_noise(c, [q], "rz")

                c.rx(q, theta=θ)
                apply_gate_noise(c, [q], "rx")

                c.rz(q, theta=α)
                apply_gate_noise(c, [q], "rz")

            apply_idle_noise(c, qubit_range, t_wait)

            # ••• Entangling ring of RZZs •••
            for q in range(num_qubits - 1):
                θ_ent = entanglement_weights[f, k, q]
                c.rzz(q, q + 1, theta=θ_ent)
                apply_gate_noise(c, [q, q + 1], "rzz")

            apply_idle_noise(c, qubit_range, t_wait)

    # ── Final rotation layer ────────────────────────────────────
    for q in qubit_range:
        θ, φ, α = final_rotations[q]

        c.rz(q, theta=φ)
        apply_gate_noise(c, [q], "rz")

        c.rx(q, theta=θ)
        apply_gate_noise(c, [q], "rx")

        c.rz(q, theta=α)
        apply_gate_noise(c, [q], "rz")

    # Example: return ⟨Z₀⟩; extend as you like
    out = [c.expectation_ps(z=[0])]
#     out = [c.expectation_ps(z=[j]) for j in range(num_qubits)]
    return jnp.real(jnp.array(out))


class QNN_IBM(nn.Module):
    num_features: int = 8         # classical input; num_qubits computed automatically
    num_frequencies: int = 1      
    layer_depth: int = 1          
    num_output: int = 1
    init_std: float = 0.1       # used in final dense layer
    init_std_Q: float = 0.1       # used for quantum circuit weights
    frequency_min_init: float = 1.0  
    trainable_frequency_min: bool = True
    t_wait: float = 0.0
    
    @property
    def num_qubits(self):
        # In this design, we assume one qubit per classical feature.
        return self.num_features

    @nn.compact
    def __call__(self, x):
        # x: shape (batch_size, num_qubits)
        num_qubits = self.num_qubits
        # Frequency scaling.
        frequency_min = (
            self.param("frequency_min", lambda rng: jnp.array(self.frequency_min_init, dtype=x.dtype))
            if self.trainable_frequency_min else self.frequency_min_init
        )
        x_scaled = x * frequency_min

        # Define shapes for the combined weights.
        shape_weights = (self.num_frequencies, self.layer_depth, num_qubits, 3)
        shape_entanglement = (self.num_frequencies, self.layer_depth, num_qubits - 1)
        shape_final = (num_qubits, 3)

        # Compute total sizes as Python integers.
        size_weights = int(np.prod(shape_weights))
        size_entanglement = int(np.prod(shape_entanglement))
        size_final = int(np.prod(shape_final))
        total_size = size_weights + size_entanglement + size_final

        # Combined parameter for all quantum circuit weights.
        quanum_weights = self.param(
            "quanum_weights",
            lambda rng, shape: jax.random.normal(rng, shape) * self.init_std_Q,
            (total_size,)
        )

        # Slice and reshape to recover individual parameters.
        weights = jnp.reshape(quanum_weights[:size_weights], shape_weights)
        entanglement_weights = jnp.reshape(
            quanum_weights[size_weights:size_weights + size_entanglement], shape_entanglement
        )
        final_rotations = jnp.reshape(
            quanum_weights[size_weights + size_entanglement:], shape_final
        )

        # Build the quantum circuit output (vectorized over the batch).
        circuit_out = jax.vmap(
            lambda single_x: quantum_circuit_ibm(
                single_x,
                weights,
                entanglement_weights,
                final_rotations,
                num_qubits,
                self.layer_depth,
                self.num_frequencies,
                self.t_wait
            )
        )(x_scaled)
        # Use a dense layer on the circuit output.
        output = nn.Dense(self.num_output,
                          kernel_init=nn.initializers.normal(stddev=self.init_std)
                         )(circuit_out)
        return output