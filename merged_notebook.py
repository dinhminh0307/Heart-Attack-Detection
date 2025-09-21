# Welch's t-test: Max heart rate vs Exercise-induced angina
import pandas as pd
import numpy as np
from scipy import stats

# Load data
df = pd.read_csv("Heart_Attack_Cleaned.csv")

# Identify required columns
ang_col = next((c for c in df.columns if c.strip().lower() == "exercise angina"), None)
hr_col  = next((c for c in df.columns if c.strip().lower() == "max heart rate"), None)
if ang_col is None or hr_col is None:
    raise ValueError("Required columns not found: exercise angina, max heart rate")

# Prepare fields
df[ang_col] = pd.to_numeric(df[ang_col], errors="coerce")
df[hr_col]  = pd.to_numeric(df[hr_col],  errors="coerce")
df = df.dropna(subset=[ang_col, hr_col]).copy()

# Define groups
df["Group"] = np.where(df[ang_col] == 1, "With exercise-induced angina", "Without exercise-induced angina")

# Arrays for Welch
with_ang    = df.loc[df["Group"] == "With exercise-induced angina", hr_col].to_numpy()
without_ang = df.loc[df["Group"] == "Without exercise-induced angina", hr_col].to_numpy()

# Table 1: Descriptive statistics
desc = (
    df.groupby("Group")[hr_col]
      .agg(n="count",
           mean="mean",
           sd=lambda s: s.std(ddof=1),
           median="median",
           min="min",
           max="max")
      .reset_index()
      .rename(columns={
          "mean":   "Mean HR (bpm)",
          "sd":     "SD (bpm)",
          "median": "Median (bpm)",
          "min":    "Min (bpm)",
          "max":    "Max (bpm)"
      })
)
for col in ["Mean HR (bpm)", "SD (bpm)"]:
    desc[col] = desc[col].round(2)

# Welch's t-test (two-tailed)
t_stat, p_val = stats.ttest_ind(with_ang, without_ang, equal_var=False, alternative="two-sided")
t_stat, p_val = float(t_stat), float(p_val)

n_with, n_without = len(with_ang), len(without_ang)
s1_sq, s2_sq = float(np.var(with_ang, ddof=1)), float(np.var(without_ang, ddof=1))
v1, v2 = s1_sq/n_with, s2_sq/n_without
df_welch = (v1 + v2)**2 / ((v1**2)/(n_with-1) + (v2**2)/(n_without-1))
mean_with, mean_without = float(np.mean(with_ang)), float(np.mean(without_ang))
mean_diff = mean_with - mean_without
se_diff = float(np.sqrt(v1 + v2))
tcrit = stats.t.ppf(0.975, df_welch)
ci_low, ci_high = mean_diff - tcrit*se_diff, mean_diff + tcrit*se_diff

# Minimal effect sizes
sp2 = ((n_with-1)*s1_sq + (n_without-1)*s2_sq) / (n_with + n_without - 2)
sp = float(np.sqrt(sp2))
hedges_g = (1 - (3/(4*(n_with + n_without) - 9))) * (mean_diff / sp)
r_pb = t_stat / np.sqrt(t_stat**2 + df_welch)

# Print tables
print("Table 1. Maximum heart rate by exercise-induced angina status\n")
print(desc[["Group","n","Mean HR (bpm)","SD (bpm)","Median (bpm)","Min (bpm)","Max (bpm)"]]
      .to_string(index=False))

print("\nTable 2. Welch's t-test results (Max HR: with angina − without angina)\n")
res = pd.DataFrame([{
    "Mean difference (bpm)": round(mean_diff, 2),
    "95% CI (bpm)": f"[{ci_low:.2f}, {ci_high:.2f}]",
    "t (Welch)": round(t_stat, 2),
    "df (Welch)": round(df_welch, 2),
    "p-value (two-tailed)": f"{p_val:.2e}",
    "Hedges' g": round(hedges_g, 2),
    "Point-biserial r": round(r_pb, 3)
}])
print(res.to_string(index=False))

# Simple decision line
alpha = 0.05
decision = "Reject H0: max HR differs by angina status" if p_val < alpha else "Fail to reject H0"
print(f"\nDecision (alpha=0.05): {decision}.")

# Welch's t-test: Age vs Heart-attack Risk
import pandas as pd
import numpy as np
from scipy import stats

# Load
df = pd.read_csv("Heart_Attack_Cleaned.csv")

# Identify columns
age_col = next((c for c in df.columns if c.strip().lower() == "age"), None)
target_col = next((c for c in df.columns if c.strip().lower() == "target"), None)
if age_col is None or target_col is None:
    raise ValueError("Required columns not found: age, target")

# Prepare fields
df[age_col] = pd.to_numeric(df[age_col], errors="coerce")
df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
df = df.dropna(subset=[age_col, target_col]).copy()

# Define groups: risk-positive := target != 0; risk-negative := target == 0
df["Risk group"] = np.where(df[target_col] != 0, "Risk positive", "Risk negative")

# Arrays
pos = df.loc[df["Risk group"] == "Risk positive", age_col].to_numpy()
neg = df.loc[df["Risk group"] == "Risk negative", age_col].to_numpy()

# Table 1: Descriptive statistics
desc = (
    df.groupby("Risk group")[age_col]
      .agg(n="count",
           mean="mean",
           sd=lambda s: s.std(ddof=1),
           median="median",
           min="min",
           max="max")
      .reset_index()
      .rename(columns={
          "mean": "Mean age (years)",
          "sd": "SD (years)",
          "median": "Median (years)",
          "min": "Min (years)",
          "max": "Max (years)"
      })
)
for col in ["Mean age (years)", "SD (years)"]:
    desc[col] = desc[col].round(2)

# Welch's t-test (two-tailed)
t_stat, p_val = stats.ttest_ind(pos, neg, equal_var=False, alternative="two-sided")
t_stat, p_val = float(t_stat), float(p_val)

n_pos, n_neg = len(pos), len(neg)
s1_sq, s2_sq = float(np.var(pos, ddof=1)), float(np.var(neg, ddof=1))
v1, v2 = s1_sq/n_pos, s2_sq/n_neg
df_welch = (v1 + v2)**2 / ((v1**2)/(n_pos-1) + (v2**2)/(n_neg-1))
mean_pos, mean_neg = float(np.mean(pos)), float(np.mean(neg))
mean_diff = mean_pos - mean_neg
se_diff = float(np.sqrt(v1 + v2))
tcrit = stats.t.ppf(0.975, df_welch)
ci_low, ci_high = mean_diff - tcrit*se_diff, mean_diff + tcrit*se_diff

# Minimal effect size
sp2 = ((n_pos-1)*s1_sq + (n_neg-1)*s2_sq) / (n_pos + n_neg - 2)
sp = float(np.sqrt(sp2))
hedges_g = (1 - (3/(4*(n_pos + n_neg) - 9))) * (mean_diff / sp)

# Print tables
print("Table 1. Descriptive statistics of age by risk group\n")
print(desc[["Risk group","n","Mean age (years)","SD (years)","Median (years)","Min (years)","Max (years)"]]
      .to_string(index=False))

print("\nTable 2. Welch's t-test results (Age: risk-positive − risk-negative)\n")
res = pd.DataFrame([{
    "Mean difference (years)": round(mean_diff, 2),
    "95% CI (years)": f"[{ci_low:.2f}, {ci_high:.2f}]",
    "t (Welch)": round(t_stat, 2),
    "df (Welch)": round(df_welch, 2),
    "p-value (two-tailed)": f"{p_val:.2e}",
    "Hedges' g": round(hedges_g, 2)
}])
print(res.to_string(index=False))

# Conclusion
alpha = 0.05
decision = "Reject H0" if p_val < alpha else "Fail to reject H0"
print(f"\nDecision (alpha=0.05): {decision}.")

# Two-proportion test: Heart-attack risk (target != 0) by sex — clean console tables (no exports)
import pandas as pd
import numpy as np
from scipy import stats

# Load
df = pd.read_csv("Heart_Attack_Cleaned.csv")

# Identify columns
sex_col = next((c for c in df.columns if c.strip().lower() == "sex"), None)
target_col = next((c for c in df.columns if c.strip().lower() == "target"), None)
if sex_col is None or target_col is None:
    raise ValueError("Required columns not found: sex, target")

# Prepare
df[sex_col] = pd.to_numeric(df[sex_col], errors="coerce")
df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
df = df.dropna(subset=[sex_col, target_col]).copy()

# Define: risk-positive := target != 0; risk-negative := target == 0
df["Risk positive"] = (df[target_col] != 0).astype(int)
df["Sex"] = np.where(df[sex_col] == 1, "Male", "Female")

# Counts and rates by sex
tab = (
    df.groupby("Sex")["Risk positive"]
      .agg(n="count", risk_pos="sum")
      .reset_index()
)
tab["risk_neg"] = tab["n"] - tab["risk_pos"]
tab["risk_rate"] = (tab["risk_pos"] / tab["n"]) * 100

# Print Table 1
print("Table 1. Heart-attack risk by sex\n")
print(tab[["Sex","n","risk_pos","risk_neg","risk_rate"]]
      .rename(columns={
          "risk_pos":"Risk-positive",
          "risk_neg":"Risk-negative",
          "risk_rate":"Risk rate (%)"
      })
      .assign(**{"Risk rate (%)": lambda d: d["Risk rate (%)"].round(2)})
      .to_string(index=False))

# Extract male/female counts
row_m = tab.loc[tab["Sex"] == "Male"].iloc[0]
row_f = tab.loc[tab["Sex"] == "Female"].iloc[0]
n_male, x_male = int(row_m["n"]), int(row_m["risk_pos"])
n_female, x_female = int(row_f["n"]), int(row_f["risk_pos"])

p_male = x_male / n_male
p_female = x_female / n_female

# Two-proportion z-test (pooled SE for test)
p_pool = (x_male + x_female) / (n_male + n_female)
se_pool = np.sqrt(p_pool * (1 - p_pool) * (1/n_male + 1/n_female))
z_stat = (p_male - p_female) / se_pool
p_val_two = 2 * (1 - stats.norm.cdf(abs(z_stat)))

# 95% CI for difference (unpooled SE)
se_unpooled = np.sqrt(p_male*(1-p_male)/n_male + p_female*(1-p_female)/n_female)
zcrit = stats.norm.ppf(0.975)
diff = p_male - p_female
ci_low, ci_high = diff - zcrit*se_unpooled, diff + zcrit*se_unpooled

# Risk ratio and odds ratio with 95% CIs (Wald on log scale, continuity if needed)
A, B = x_male, n_male - x_male
C, D = x_female, n_female - x_female
if min(A,B,C,D) == 0:
    A += 0.5; B += 0.5; C += 0.5; D += 0.5

rr = (A/(A+B)) / (C/(C+D))
se_log_rr = np.sqrt(1/A - 1/(A+B) + 1/C - 1/(C+D))
rr_low, rr_high = np.exp(np.log(rr) - zcrit*se_log_rr), np.exp(np.log(rr) + zcrit*se_log_rr)

odds_ratio = (A*D) / (B*C)
se_log_or = np.sqrt(1/A + 1/B + 1/C + 1/D)
or_low, or_high = np.exp(np.log(odds_ratio) - zcrit*se_log_or), np.exp(np.log(odds_ratio) + zcrit*se_log_or)

# Print Table 2
res = pd.DataFrame([{
    "Risk difference (Male − Female)": round(diff, 4),
    "95% CI (difference)": f"[{ci_low:.4f}, {ci_high:.4f}]",
    "z statistic": round(z_stat, 2),
    "p-value (two-tailed)": "< 1e-15" if p_val_two < 1e-15 else f"{p_val_two:.2e}",
    "Risk ratio": f"{rr:.2f} [{rr_low:.2f}, {rr_high:.2f}]",
    "Odds ratio": f"{odds_ratio:.2f} [{or_low:.2f}, {or_high:.2f}]"
}])

print("\nTable 2. Two-proportion z-test summary (Male − Female)\n")
print(res.to_string(index=False))

# Simple decision line
alpha = 0.05
decision = "Reject H0: different risk rates by sex" if p_val_two < alpha else "Fail to reject H0"
print(f"\nDecision (alpha=0.05): {decision}.")

# Welch's t-test: Resting blood pressure (trestbps) vs Heart-attack risk
import pandas as pd
import numpy as np
from scipy import stats

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
except Exception:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
    from rich.console import Console
    from rich.table import Table
    from rich import box

console = Console(force_jupyter=True)

# --- Load data (unchanged logic) ---
df = pd.read_csv("Heart_Attack_Cleaned.csv")

# Identify required columns
bp_col = next((c for c in df.columns if c.strip().lower() == "trestbps"), None)
target_col = next((c for c in df.columns if c.strip().lower() == "target"), None)
if bp_col is None or target_col is None:
    raise ValueError("Required columns not found: trestbps, target")

# Prepare fields
df[bp_col] = pd.to_numeric(df[bp_col], errors="coerce")
df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
df = df.dropna(subset=[bp_col, target_col]).copy()

# Define groups: risk-positive := target != 0; risk-negative := target == 0
df["Risk group"] = np.where(df[target_col] != 0, "Risk positive", "Risk negative")

# Arrays for Welch
pos = df.loc[df["Risk group"] == "Risk positive", bp_col].to_numpy()
neg = df.loc[df["Risk group"] == "Risk negative", bp_col].to_numpy()

# Table 1: Descriptive statistics by group
desc = (
    df.groupby("Risk group")[bp_col]
      .agg(n="count",
           mean="mean",
           sd=lambda s: s.std(ddof=1),
           median="median",
           min="min",
           max="max")
      .reset_index()
      .rename(columns={
          "mean": "Mean BP (mmHg)",
          "sd": "SD (mmHg)",
          "median": "Median (mmHg)",
          "min": "Min (mmHg)",
          "max": "Max (mmHg)"
      })
)
for col in ["Mean BP (mmHg)", "SD (mmHg)", "Median (mmHg)", "Min (mmHg)", "Max (mmHg)"]:
    desc[col] = pd.to_numeric(desc[col], errors="coerce").round(2)

# Welch's t-test (two-tailed)
t_stat, p_val = stats.ttest_ind(pos, neg, equal_var=False, alternative="two-sided")
t_stat, p_val = float(t_stat), float(p_val)

n_pos, n_neg = len(pos), len(neg)
s1_sq, s2_sq = float(np.var(pos, ddof=1)), float(np.var(neg, ddof=1))
v1, v2 = s1_sq/n_pos, s2_sq/n_neg
df_welch = (v1 + v2)**2 / ((v1**2)/(n_pos-1) + (v2**2)/(n_neg-1))
mean_pos, mean_neg = float(np.mean(pos)), float(np.mean(neg))
mean_diff = mean_pos - mean_neg
se_diff = float(np.sqrt(v1 + v2))
tcrit = stats.t.ppf(0.975, df_welch)
ci_low, ci_high = mean_diff - tcrit*se_diff, mean_diff + tcrit*se_diff

# Minimal effect sizes
sp2 = ((n_pos-1)*s1_sq + (n_neg-1)*s2_sq) / (n_pos + n_neg - 2)
sp = float(np.sqrt(sp2))
hedges_g = (1 - (3/(4*(n_pos + n_neg) - 9))) * (mean_diff / sp)
r_pb = t_stat / np.sqrt(t_stat**2 + df_welch)

def rg_num(value, positive_is_good=True, fmt="{:.2f}", threshold=None):
    """
    Return value as plain black text, except:
      - green for 'good' (positive if positive_is_good, or < threshold if provided),
      - red for the opposite.
    """
    try:
        v = float(value)
    except Exception:
        return str(value)

    if threshold is not None:
        # e.g., p-value with alpha threshold
        return f"[green]{fmt.format(v)}[/]" if v < threshold else f"[red]{fmt.format(v)}[/]"
    else:
        if positive_is_good and v > 0:
            return f"[green]{fmt.format(v)}[/]"
        if positive_is_good and v < 0:
            return f"[red]{fmt.format(v)}[/]"
        if not positive_is_good and v < 0:
            return f"[green]{fmt.format(v)}[/]"
        if not positive_is_good and v > 0:
            return f"[red]{fmt.format(v)}[/]"
    return fmt.format(v)

# --- Renderers ---
def render_table1(desc_df: pd.DataFrame):
    t = Table(
        title="Table 1. Descriptive statistics of resting blood pressure by risk group",
        box=box.SIMPLE_HEAVY,
        header_style="bold",        # black text (theme default), bold for emphasis
        row_styles=["none", "dim"]  # zebra without color
    )
    t.add_column("Risk group", justify="left", no_wrap=True)
    t.add_column("n", justify="right")
    t.add_column("Mean BP (mmHg)", justify="right")
    t.add_column("SD (mmHg)", justify="right")
    t.add_column("Median (mmHg)", justify="right")
    t.add_column("Min (mmHg)", justify="right")
    t.add_column("Max (mmHg)", justify="right")

    cols = ["Risk group","n","Mean BP (mmHg)","SD (mmHg)","Median (mmHg)","Min (mmHg)","Max (mmHg)"]
    for _, r in desc_df[cols].iterrows():
        t.add_row(
            str(r["Risk group"]),
            f"{int(r['n'])}",
            f"{r['Mean BP (mmHg)']:.2f}",
            f"{r['SD (mmHg)']:.2f}",
            f"{r['Median (mmHg)']:.2f}",
            f"{r['Min (mmHg)']:.2f}",
            f"{r['Max (mmHg)']:.2f}",
        )
    console.print(t)

def render_table2(mean_diff, ci_low, ci_high, t_stat, df_welch, p_val, hedges_g, r_pb, alpha=0.05):
    t = Table(
        title="Table 2. Welch's t-test results (Resting BP: risk-positive − risk-negative)",
        box=box.SIMPLE_HEAVY,
        header_style="bold",        # black text headers
        row_styles=["none", "dim"]  # zebra without color
    )
    t.add_column("Metric", justify="left", no_wrap=True)
    t.add_column("Value", justify="right")

    # Targeted red/green only for mean difference and p-value
    mean_diff_txt = rg_num(mean_diff, positive_is_good=True, fmt="{:.2f}")
    p_txt = rg_num(p_val, fmt="{:.2e}", threshold=alpha)

    t.add_row("Mean difference (mmHg)", mean_diff_txt)
    t.add_row("95% CI (mmHg)", f"[{ci_low:.2f}, {ci_high:.2f}]")
    t.add_row("t (Welch)", f"{t_stat:.2f}")
    t.add_row("df (Welch)", f"{df_welch:.2f}")
    t.add_row("p-value (two-tailed)", p_txt)
    t.add_row("Hedges' g", f"{hedges_g:.2f}")
    t.add_row("Point-biserial r", f"{r_pb:.3f}")
    console.print(t)

    decision = "Reject H0: resting BP differs by risk group" if p_val < alpha else "Fail to reject H0"
    console.print(f"[bold]Decision (alpha={alpha:.2f}): {decision}[/]")

# --- Render both tables  ---
alpha = 0.05
render_table1(desc)
render_table2(mean_diff, ci_low, ci_high, t_stat, df_welch, p_val, hedges_g, r_pb, alpha=alpha)

# Welch's t-test: Cholesterol vs Heart-attack occurrence 

# --- Core imports (stats unchanged) ---
import pandas as pd
import numpy as np
from scipy import stats

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
except Exception:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
    from rich.console import Console
    from rich.table import Table
    from rich import box

console = Console(force_jupyter=True)

df = pd.read_csv("Heart_Attack_Cleaned.csv")

# Identify columns
chol_col = next((c for c in df.columns if c.strip().lower() == "cholesterol"), None)
target_col = next((c for c in df.columns if c.strip().lower() == "target"), None)
if chol_col is None or target_col is None:
    raise ValueError("Required columns not found: cholesterol, target")

# Prepare fields
df[chol_col] = pd.to_numeric(df[chol_col], errors="coerce")
df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
df = df.dropna(subset=[chol_col, target_col]).copy()

# Define groups: risk-positive := target != 0; risk-negative := target == 0
df["Risk group"] = np.where(df[target_col] != 0, "Risk positive", "Risk negative")

# Arrays for Welch
pos = df.loc[df["Risk group"] == "Risk positive", chol_col].to_numpy()
neg = df.loc[df["Risk group"] == "Risk negative", chol_col].to_numpy()

# Table 1: Descriptive statistics
desc = (
    df.groupby("Risk group")[chol_col]
      .agg(n="count",
           mean="mean",
           sd=lambda s: s.std(ddof=1),
           median="median",
           min="min",
           max="max")
      .reset_index()
      .rename(columns={
          "mean": "Mean Chol (mg/dL)",
          "sd": "SD (mg/dL)",
          "median": "Median (mg/dL)",
          "min": "Min (mg/dL)",
          "max": "Max (mg/dL)"
      })
)
# Round for display only
for col in ["Mean Chol (mg/dL)", "SD (mg/dL)", "Median (mg/dL)", "Min (mg/dL)", "Max (mg/dL)"]:
    desc[col] = pd.to_numeric(desc[col], errors="coerce").round(2)

# Welch's t-test (two-tailed)
t_stat, p_val = stats.ttest_ind(pos, neg, equal_var=False, alternative="two-sided")
t_stat, p_val = float(t_stat), float(p_val)

n_pos, n_neg = len(pos), len(neg)
s1_sq, s2_sq = float(np.var(pos, ddof=1)), float(np.var(neg, ddof=1))
v1, v2 = s1_sq/n_pos, s2_sq/n_neg
df_welch = (v1 + v2)**2 / ((v1**2)/(n_pos-1) + (v2**2)/(n_neg-1))
mean_pos, mean_neg = float(np.mean(pos)), float(np.mean(neg))
mean_diff = mean_pos - mean_neg
se_diff = float(np.sqrt(v1 + v2))
tcrit = stats.t.ppf(0.975, df_welch)
ci_low, ci_high = mean_diff - tcrit*se_diff, mean_diff + tcrit*se_diff

# Minimal effect sizes
sp2 = ((n_pos-1)*s1_sq + (n_neg-1)*s2_sq) / (n_pos + n_neg - 2)
sp = float(np.sqrt(sp2))
hedges_g = (1 - (3/(4*(n_pos + n_neg) - 9))) * (mean_diff / sp)
r_pb = t_stat / np.sqrt(t_stat**2 + df_welch)

# --- Helper: red/green only where it matters; otherwise black ---
def rg_num(value, positive_is_good=True, fmt="{:.2f}"):
    """
    Return value as plain black text, except:
      - green for 'good' (positive if positive_is_good, negative if not),
      - red for the opposite.
    """
    try:
        v = float(value)
    except Exception:
        return str(value)

    if positive_is_good:
        if v > 0:
            return f"[green]{fmt.format(v)}[/]"
        elif v < 0:
            return f"[red]{fmt.format(v)}[/]"
    else:
        # e.g., p-value where smaller is better
        if v < 0.05:  # default threshold; caller can format with alpha elsewhere
            return f"[green]{fmt.format(v)}[/]"
        else:
            return f"[red]{fmt.format(v)}[/]"
    return fmt.format(v)

# --- Renderers ---
def render_table1(desc_df: pd.DataFrame):
    # All-black headers (bold only) and dim zebra row for tracking; no colored text here
    t = Table(
        title="Table 1. Descriptive statistics of cholesterol by risk group",
        box=box.SIMPLE_HEAVY,
        header_style="bold",        # black text (theme default), bold for emphasis
        row_styles=["none", "dim"]  # zebra without color
    )
    # Columns
    t.add_column("Risk group", justify="left", no_wrap=True)
    t.add_column("n", justify="right")
    t.add_column("Mean Chol (mg/dL)", justify="right")
    t.add_column("SD (mg/dL)", justify="right")
    t.add_column("Median (mg/dL)", justify="right")
    t.add_column("Min (mg/dL)", justify="right")
    t.add_column("Max (mg/dL)", justify="right")

    cols = ["Risk group","n","Mean Chol (mg/dL)","SD (mg/dL)","Median (mg/dL)","Min (mg/dL)","Max (mg/dL)"]
    for _, r in desc_df[cols].iterrows():
        t.add_row(
            str(r["Risk group"]),
            f"{int(r['n'])}",
            f"{r['Mean Chol (mg/dL)']:.2f}",
            f"{r['SD (mg/dL)']:.2f}",
            f"{r['Median (mg/dL)']:.2f}",
            f"{r['Min (mg/dL)']:.2f}",
            f"{r['Max (mg/dL)']:.2f}",
        )
    console.print(t)

def render_table2(mean_diff, ci_low, ci_high, t_stat, df_welch, p_val, hedges_g, r_pb, alpha=0.05):
    t = Table(
        title="Table 2. Welch's t-test results (Cholesterol: risk-positive − risk-negative)",
        box=box.SIMPLE_HEAVY,
        header_style="bold",        # black text headers
        row_styles=["none", "dim"]  # zebra without color
    )
    t.add_column("Metric", style="", justify="left", no_wrap=True)
    t.add_column("Value", justify="right")

    # Only color values where helpful
    mean_diff_txt = rg_num(mean_diff, positive_is_good=True, fmt="{:.2f}")
    p_txt = f"{p_val:.2e}"
    # Color p-value green if < alpha else red
    p_txt = f"[green]{p_txt}[/]" if p_val < alpha else f"[red]{p_txt}[/]"

    # Optional: sign-driven color for g and r
    g_txt = rg_num(hedges_g, positive_is_good=True, fmt="{:.2f}")
    r_txt = rg_num(r_pb, positive_is_good=True, fmt="{:.3f}")

    t.add_row("Mean difference (mg/dL)", mean_diff_txt)
    t.add_row("95% CI (mg/dL)", f"[{ci_low:.2f}, {ci_high:.2f}]")
    t.add_row("t (Welch)", f"{t_stat:.2f}")
    t.add_row("df (Welch)", f"{df_welch:.2f}")
    t.add_row("p-value (two-tailed)", p_txt)
    t.add_row("Hedges' g", g_txt)
    t.add_row("Point-biserial r", r_txt)
    console.print(t)

    # Emphasized conclusion: bold only (no color coding)
    decision = "Reject H0: cholesterol differs by risk group" if p_val < alpha else "Fail to reject H0"
    console.print(f"[bold]Decision (alpha={alpha:.2f}): {decision}[/]")

# --- Render both tables (presentation only; calculations above unchanged) ---
render_table1(desc)
render_table2(mean_diff, ci_low, ci_high, t_stat, df_welch, p_val, hedges_g, r_pb, alpha=0.05)
