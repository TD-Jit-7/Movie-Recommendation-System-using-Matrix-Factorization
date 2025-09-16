# Movie-Recommendation-System-using-Matrix-Factorization

---

## 📖 Main Code Overview

The system is designed to predict missing ratings in a **user–item matrix** using **latent factor models**.

* **R** = user–item matrix (sparse, most entries unknown)
* **r<sub>um</sub>** = rating assigned by user *u* to movie *m*
* **U** = number of users
* **M** = number of movies

### 🎯 Goal

Predict missing ratings using latent factors:

$$
\hat{r}_{um} \approx x_u^T w_m
$$

* $x_u \in \mathbb{R}^n$: latent vector for user *u*
* $w_m \in \mathbb{R}^n$: latent vector for movie *m*
* $n$: latent dimension

---

### 📉 Loss Function

We minimize the **mean squared error (MSE)** between actual and predicted ratings, with **L2 regularization** to prevent overfitting:

$$
\mathcal{L} = \frac{1}{|D|} \sum_{(u,m)\in D} \left(r_{um} - x_u^T w_m\right)^2 + \lambda \left( ||X||^2 + ||W||^2 \right)
$$

* **D** = set of observed ratings
* **λ** = regularization coefficient

---

### 🔄 Gradient Descent Updates

For each observed rating $(u, m, r_{um})$:

$$
\hat{r}_{um} = x_u^T w_m
$$

$$
e_{um} = r_{um} - \hat{r}_{um}
$$

Updates (using SGD):

$$
\frac{\partial L}{\partial x_u} = -2 e_{um} w_m + 2 \lambda x_u
$$

$$
\frac{\partial L}{\partial w_m} = -2 e_{um} x_u + 2 \lambda w_m
$$

**Code snippet (SGD updates):**

```python
# Gradient updates (SGD)
X[u] += lr * (err * W[m] - reg * X[u])
W[m] += lr * (err * X[u] - reg * W[m])
```


📂 Dataset

This project uses the MovieLens Small Dataset (100k ratings) provided by GroupLens Research
.
https://grouplens.org/datasets/movielens/latest/
---


