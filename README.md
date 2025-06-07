# Gradient Descent Algorithm 

## 1D Gradient Descent Algorithm (10 minutes)

=**What you'll learn:**
- Understand the fundamentals of gradient descent in 1D
- Learn the mathematical formulation
- Implement and visualize 1D gradient descent
- Analyze convergence behavior and learning rates

### What is Gradient Descent?
**Definition:**
- Iterative optimization algorithm to find minimum of a function
- Follows the steepest descent direction
- Uses the negative gradient to update parameters

**Key Intuition:**
- Imagine rolling a ball down a hill
- Ball naturally moves toward the lowest point
- Gradient descent mimics this behavior mathematically

### Mathematical Foundation - 1D Case
**For function f(x):**

**Update Rule:**
```
x_{k+1} = x_k - α × f'(x_k)
```

**Where:**
- x_k: current position
- α: learning rate (step size)
- f'(x_k): derivative (slope) at current position
- x_{k+1}: next position

### Algorithm Steps
**1D Gradient Descent Algorithm:**

1. **Initialize:** Choose starting point x₀
2. **Set:** Learning rate α
3. **Repeat until convergence:**
   - Calculate derivative: f'(x_k)
   - Update position: x_{k+1} = x_k - α × f'(x_k)
   - Check convergence: |f'(x_k)| < ε

### Example - Minimizing f(x) = x² + 4x + 3
**Function:** f(x) = x² + 4x + 3
**Derivative:** f'(x) = 2x + 4
**Starting point:** x₀ = 2
**Learning rate:** α = 0.1

**Step-by-step calculation:**
- x₁ = 2 - 0.1 × (2×2 + 4) = 2 - 0.8 = 1.2
- x₂ = 1.2 - 0.1 × (2×1.2 + 4) = 1.2 - 0.64 = 0.56
- x₃ = 0.56 - 0.1 × (2×0.56 + 4) = 0.56 - 0.512 = 0.048


### Learning Rate Impact
**Learning Rate Effects:**

**Too Small (α = 0.01):**
- Slow convergence
- Many iterations needed
- Stable but inefficient

**Optimal (α = 0.1):**
- Good convergence speed
- Reaches minimum efficiently

**Too Large (α = 0.9):**
- May overshoot minimum
- Unstable oscillations
- Possible divergence


### Applications & Summary
**1D Gradient Descent Applications:**
- Finding roots of equations
- Simple optimization problems
- Foundation for higher dimensions
- Parameter tuning in simple models

**Key Takeaways:**
- Simple yet powerful optimization technique
- Learning rate is crucial for performance
- Always converges for convex functions
- Building block for complex algorithms

---

## 2D Gradient Descent Algorithm

### Learning Objectives
**What you'll learn:**
- Extend gradient descent to 2D functions
- Understand gradient vectors and contour plots
- Implement 2D gradient descent with visualization
- Analyze convergence patterns in 2D space

### From 1D to 2D
**Extension to Two Dimensions:**

**1D:** f(x) → scalar derivative f'(x)
**2D:** f(x,y) → gradient vector ∇f = [∂f/∂x, ∂f/∂y]

**Gradient Vector:**
- Points in direction of steepest increase
- We move in opposite direction (negative gradient)
- Magnitude indicates steepness

### Mathematical Formulation
**For function f(x,y):**

**Update Rules:**
```
x_{k+1} = x_k - α × ∂f/∂x
y_{k+1} = y_k - α × ∂f/∂y
```

**Vector Form:**
```
[x_{k+1}]   [x_k]       [∂f/∂x]
[y_{k+1}] = [y_k] - α × [∂f/∂y]
```

### 2D Algorithm Steps
**2D Gradient Descent Algorithm:**

1. **Initialize:** Starting point (x₀, y₀)
2. **Set:** Learning rate α
3. **Repeat until convergence:**
   - Calculate partial derivatives: ∂f/∂x, ∂f/∂y
   - Update x: x_{k+1} = x_k - α × ∂f/∂x
   - Update y: y_{k+1} = y_k - α × ∂f/∂y
   - Check convergence: ||∇f|| < ε

### Example - f(x,y) = x² + y² + 2x - 4y + 5
**Function:** f(x,y) = x² + y² + 2x - 4y + 5

**Partial Derivatives:**
- ∂f/∂x = 2x + 2
- ∂f/∂y = 2y - 4

**Starting point:** (3, 1)
**Learning rate:** α = 0.1

**First iteration:**
- x₁ = 3 - 0.1 × (2×3 + 2) = 3 - 0.8 = 2.2
- y₁ = 1 - 0.1 × (2×1 - 4) = 1 + 0.2 = 1.2

### Visualization Techniques
**Understanding 2D Gradient Descent:**

**3D Surface Plot:**
- Shows function landscape
- Gradient descent path on surface
- Visual understanding of optimization

**Contour Plot:**
- 2D representation with level curves
- Gradient descent path projection
- Easy to see convergence pattern


### Applications & Key Insights
**2D Gradient Descent Applications:**
- Image processing optimization
- Simple machine learning models
- Engineering design problems
- Economics and finance modeling

**Key Insights:**
- Path depends on starting point
- Convergence can be slower in valleys
- Multiple starting points → same minimum (convex functions)
- Foundation for higher-dimensional problems

---

## Multivariate Gradient Descent Algorithm (10 minutes)

### Learning Objectives
**What you'll learn:**
- Understand gradient descent in n-dimensions
- Apply to real machine learning problems
- Implement linear and logistic regression
- Compare different gradient descent variants

### Multivariate Gradient Descent
**Extension to n Dimensions:**

**For function f(x₁, x₂, ..., xₙ):**

**Gradient Vector:**
```
∇f(x) = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ
```

**Update Rule:**
```
x_{k+1} = x_k - α × ∇f(x_k)
```

**Where x is now a vector of n parameters**

### Linear Regression Application
**Problem:** Find best-fit line for data points

**Cost Function (MSE):**
```
J(θ) = (1/2m) Σ(hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
```

**Hypothesis:** hθ(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ

**Gradients:**
```
∂J/∂θⱼ = (1/m) Σ(hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾)xⱼ⁽ⁱ⁾
```


### Logistic Regression Application
**Problem:** Binary classification

**Cost Function (Log-likelihood):**
```
J(θ) = -(1/m) Σ[y⁽ⁱ⁾log(hθ(x⁽ⁱ⁾)) + (1-y⁽ⁱ⁾)log(1-hθ(x⁽ⁱ⁾))]
```

**Hypothesis:** hθ(x) = 1/(1 + e^(-θᵀx))

**Gradients:**
```
∂J/∂θⱼ = (1/m) Σ(hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾)xⱼ⁽ⁱ⁾
```


### Gradient Descent Variants
**Different Types:**

**Batch Gradient Descent:**
- Uses entire dataset
- Stable convergence
- Computationally expensive

**Mini-batch Gradient Descent:**
- Uses subset of data
- Balance between stability and speed
- Most commonly used

**Stochastic Gradient Descent (SGD):**
- Uses single data point
- Fast but noisy
- Good for large datasets


### Implementation Considerations
**Practical Tips:**

**Feature Scaling:**
- Normalize features to similar ranges
- Improves convergence speed
- Prevents one feature from dominating

**Learning Rate Scheduling:**
- Start with larger learning rate
- Gradually decrease over time
- Adaptive methods (Adam, RMSprop)

**Convergence Criteria:**
- Monitor cost function
- Check gradient magnitude
- Set maximum iterations

### Real-World Applications
**Multivariate Gradient Descent in Practice:**

**Machine Learning:**
- Neural network training
- Support vector machines
- Regularized regression

**Deep Learning:**
- Backpropagation algorithm
- Computer vision models
- Natural language processing

**Other Fields:**
- Signal processing
- Control systems
- Financial modeling

**Summary:** Foundation of modern AI and optimization

---