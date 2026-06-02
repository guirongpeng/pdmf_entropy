# Gaussian-PDMF Fuzzy Number Similarity Calculation (Strictly Follow Paper)

Given two Gaussian-PDMF fuzzy numbers:
x̃₁ = ⟨x₁; d₁⁻, d₁⁺, μ₁⁻, μ₁⁺⟩
x̃₂ = ⟨x₂; d₂⁻, d₂⁺, μ₂⁻, μ₂⁺⟩
Weight λ ∈ (0,1) (e.g., λ=0.5)

## Step 1: Reuse H(μ) from entropy definition
H(μ) same as entropy: only depends on μ, σ=1 fixed.

## Step 2: Similarity formula
S_λ(x̃₁, x̃₂) = λ·exp(−|x₁−x₂|) + [(1−λ)/(2·H(0))] · [exp(−|d₁⁻−d₂⁻|)·H(μ₁⁻−μ₂⁻) + exp(−|d₁⁺−d₂⁺|)·H(μ₁⁺−μ₂⁺)
]

## Output
Return S_λ ∈ (0, 1]