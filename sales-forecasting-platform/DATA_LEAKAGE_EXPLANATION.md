# Data Leakage Explanation and Analysis

## What is Data Leakage?

**Data leakage** occurs when information from the future or from other entities that shouldn't be available is used to predict the target variable. This makes the model appear to perform better than it actually will in production.

### Types of Data Leakage:

1. **Temporal Leakage**: Using future information to predict the past
2. **Cross-Entity Leakage**: Using information from other entities (stores/products) to predict one entity
3. **Target Leakage**: Using the target variable (or a proxy) as a feature
4. **Train-Test Contamination**: Using test/evaluation data information during training

---

## 1. Lag Features Without Grouping (FIXED ✅)

### The Problem (Before Fix):

```python
# WRONG - No grouping
df['sale_amount_lag_1'] = df['sale_amount'].shift(1)
```

**Example of the Leakage:**
```
Row 1: Store A, Product X, Date 2024-01-01, Sales = 10
Row 2: Store B, Product Y, Date 2024-01-01, Sales = 20  ← Different store/product!
Row 3: Store A, Product X, Date 2024-01-02, Sales = 15

Without grouping:
- Row 3's lag_1 = 20 (from Store B, Product Y) ❌ WRONG!

With grouping:
- Row 3's lag_1 = 10 (from Store A, Product X, previous day) ✅ CORRECT
```

**Why This is Leakage:**
- The model learns patterns from **other stores/products** that don't apply
- Store B's sales don't help predict Store A's sales
- This creates false correlations and inflates performance metrics

### The Fix (Current Code):

```python
# CORRECT - Grouped by store and product
df['sale_amount_lag_1'] = df.groupby(['store_id', 'product_id'])['sale_amount'].shift(1)
```

Now lag features only use historical data from the **same store and product**.

---

## 2. Rolling Features Without Grouping (FIXED ✅)

### The Problem (Before Fix):

```python
# WRONG - No grouping
df['sale_amount_rolling_7_mean'] = df['sale_amount'].rolling(window=7).mean()
```

**Example:**
```
If data is sorted by date globally (not by store/product):
- Store A, Product X, Day 1: Sales = 10
- Store B, Product Y, Day 1: Sales = 20
- Store C, Product Z, Day 1: Sales = 30
- Store A, Product X, Day 2: Sales = 15
  → rolling_7_mean includes Store B and C's sales ❌ WRONG!
```

**Why This is Leakage:**
- Rolling window includes sales from different stores/products
- Creates artificial patterns that don't exist in reality
- Model learns from unrelated data

### The Fix (Current Code):

```python
# CORRECT - Grouped by store and product
df['sale_amount_rolling_7_mean'] = df.groupby(['store_id', 'product_id'])['sale_amount'].transform(
    lambda x: x.rolling(window=7, min_periods=1).mean()
)
```

Now rolling features only use data from the **same store and product**.

---

## 3. Hierarchical Features - POTENTIAL LEAKAGE ⚠️

### Current Implementation:

```python
# Category-level aggregations
agg_df = df.groupby([cat_level, date_col])['sale_amount'].agg(['mean', 'sum']).reset_index()
df = df.merge(agg_df, on=[cat_level, date_col], how='left')
```

### The Problem:

**Example:**
```
Row: Store A, Product X (Category: Beverages), Date: 2024-01-15
Feature: first_category_id_avg_sales = average of ALL products in Beverages category on 2024-01-15
```

**Why This Might Be Leakage:**
- If Product X's sales are included in the category average, then:
  - The feature contains information about the target (Product X's sales)
  - This is **target leakage** at the category level
- Even if Product X is excluded, using same-day aggregations can leak future information if data isn't properly ordered

### Is This Actually Leakage?

**It depends on the use case:**
- ✅ **NOT leakage** if: You're predicting at the end of the day, and category averages are calculated from products that already sold that day
- ❌ **IS leakage** if: You're predicting at the start of the day, and category averages include the product you're predicting

**For forecasting (predicting future sales):**
- Using same-day category averages = **LEAKAGE** (you can't know other products' sales before predicting)
- Should use **previous day's** category averages instead

---

## 4. Temporal Data Leakage - PARTIALLY FIXED ⚠️

### Current Implementation:

```python
# Time-aware split
date_sort_idx = train_features['date'].argsort()
X_train = X_train.iloc[date_sort_idx]
split_idx = int(len(X_train) * 0.8)
X_train_split = X_train.iloc[:split_idx]
X_val = X_train.iloc[split_idx:]
```

### The Issue:

**Problem**: The split is done **globally** by date, but the data contains multiple stores/products with different date ranges.

**Example:**
```
Store A, Product X: Dates 2024-01-01 to 2024-06-30
Store B, Product Y: Dates 2024-03-01 to 2024-06-30

Global sort by date:
- First 80% might include: Store A (Jan-Jun), Store B (Mar-Apr)
- Last 20% might include: Store A (Jun), Store B (May-Jun)

Problem: Store B's May data might be in validation, but Store A's June data (same time period) is in training!
```

**Why This Matters:**
- If there are global trends (holidays, seasons), the model can learn from future periods of other stores
- This is **subtle temporal leakage**

### Better Approach:

Should use **time-aware split per store-product combination**, or use a cutoff date that applies to all stores/products.

---

## 5. Recovery Model Training - POTENTIAL LEAKAGE ⚠️

### Current Implementation:

```python
# Recovery model uses lag features of sale_amount
# But sale_amount is censored during stockouts
clean_data = df[df['is_any_stockout'] == 0].copy()
X_clean = self._preprocess_features(clean_data, feature_cols)
```

### The Problem:

**During prediction on stockout periods:**
- The model uses lag features computed from `sale_amount`
- But `sale_amount` during stockouts is **censored** (lower than true demand)
- Lag features from previous stockout periods are also censored
- This creates a **distribution mismatch**:
  - Training: Lag features from clean (uncensored) data
  - Prediction: Lag features from censored data

**Why This is Leakage:**
- The model learns patterns from clean lag features
- But during stockouts, it receives censored lag features
- This is a form of **covariate shift** / **data distribution leakage**

### Solution:

Use `recovered_demand` for lag features instead of `sale_amount` (iterative approach).

---

## 6. Hierarchical Features - Same-Day Aggregation ⚠️

### Current Code:

```python
agg_df = df.groupby([cat_level, date_col])['sale_amount'].agg(['mean', 'sum'])
df = df.merge(agg_df, on=[cat_level, date_col], how='left')
```

### The Leakage:

**For a prediction at time T:**
- The feature includes sales from **other products at the same time T**
- If you're predicting Product X's sales at 2 PM:
  - Category average includes Product Y's sales at 2 PM
  - But you might not know Product Y's sales yet!

**This is leakage if:**
- You're making real-time predictions
- Products don't all sell at exactly the same time
- You need to predict before all sales are known

### Fix:

Use **previous period** aggregations:
```python
# Use yesterday's category average instead of today's
agg_df = df.groupby([cat_level, date_col])['sale_amount'].agg(['mean', 'sum']).shift(1)
```

---

## Summary of Data Leakage Issues

| Issue | Status | Severity | Impact |
|-------|--------|----------|--------|
| Lag features without grouping | ✅ FIXED | High | Was causing major leakage |
| Rolling features without grouping | ✅ FIXED | High | Was causing major leakage |
| Hierarchical same-day aggregation | ⚠️ POTENTIAL | Medium | May leak future info |
| Global date sort for split | ⚠️ PARTIAL | Low | Subtle temporal leakage |
| Recovery model lag features | ⚠️ POTENTIAL | Medium | Distribution mismatch |
| Target leakage (sale_amount) | ✅ FIXED | Critical | Was causing perfect scores |

---

## Recommendations

1. ✅ **Keep**: Grouped lag/rolling features (already fixed)
2. ⚠️ **Fix**: Use previous-day aggregations for hierarchical features
3. ⚠️ **Fix**: Use recovered_demand for lag features in recovery model
4. ⚠️ **Consider**: Per-store-product time-aware splits
5. ✅ **Keep**: Time-aware global split (better than random)

---

## How to Identify Data Leakage

**Red Flags:**
- Unrealistically high performance (R² > 0.99)
- Model performs much better on validation than on real-world data
- Features that shouldn't be predictive are highly important
- Using same-time aggregations for time-series predictions
- Features computed from data that wouldn't be available at prediction time

**Questions to Ask:**
1. "Would I have this information when making the prediction?"
2. "Does this feature include information from the future?"
3. "Does this feature include information from other entities?"
4. "Does this feature include the target variable?"

