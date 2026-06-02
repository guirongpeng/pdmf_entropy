# Gaussian-PDMF Fuzzy Number Entropy Calculation (Strictly Follow Paper)

Given a Gaussian-PDMF fuzzy number:
x̃ = ⟨x; d⁻, d⁺, μ⁻, μ⁺⟩

## Step 1: Define H(μ) (only depends on μ, σ=1 fixed)
For μ ∈ ℝ:
1. Let t(x) = tan(π·x − π/2), x ∈ [0,1]
2. f(x; μ) = Φ(t(x); μ) = CDF of standard normal at t(x), mean=μ, std=1
3. H(μ) = ∫₀¹ [ −f·ln(f) − (1−f)·ln(1−f) ] dx

H(0) is a constant ≈ 0.3702632680756489

## Step 2: Entropy formula
E(x̃) = [1/(2·H(0))] · [ exp(−1/d⁻)·H(μ⁻) + exp(−1/d⁺)·H(μ⁺) ]

## Output
Return E(x̃) ∈ [0, 1]