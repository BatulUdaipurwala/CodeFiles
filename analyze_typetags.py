#!/usr/bin/env python3
import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ColumnConfig:
	date: str
	tag: str
	clicks: str
	revenue: Optional[str] = None
	downgrade_reason: Optional[str] = None


@dataclass
class Params:
	lookback_days: int = 28
	ma_short: int = 3
	ma_long: int = 7
	disappear_days: int = 7
	new_tag_days: int = 14
	bucket_scheme: str = "80_15_5"  # or "decile"
	min_active_clicks: int = 1
	come_back_days: int = 3  # consider comeback if recovered within last N days
	min_days_for_volatility: int = 10  # for pct-change std
	# Dataset-level volatility labeling thresholds
	dataset_pct_volatile_thresh: float = 0.30
	dataset_volatile_share_thresh: float = 0.40
	# Insight thresholds for new tags
	insight_new_tags_high_pct: float = 0.80
	insight_new_tags_low_share: float = 0.40


def read_input(input_path: str) -> pd.DataFrame:
	ext = os.path.splitext(input_path)[1].lower()
	if ext in (".csv", ".tsv"):
		delim = "," if ext == ".csv" else "\t"
		return pd.read_csv(input_path, sep=delim)
	elif ext in (".parquet", ".pq"):
		return pd.read_parquet(input_path)
	else:
		raise ValueError(f"Unsupported file extension: {ext}")


def normalize_columns(df: pd.DataFrame, cols: ColumnConfig) -> pd.DataFrame:
	for col in [cols.date, cols.tag, cols.clicks]:
		if col not in df.columns:
			raise KeyError(f"Missing required column: {col}")
	df = df.copy()
	# Normalize date to datetime at day resolution
	df[cols.date] = pd.to_datetime(df[cols.date]).dt.tz_localize(None).dt.normalize()
	# Ensure types
	df[cols.tag] = df[cols.tag].astype(str)
	df[cols.clicks] = pd.to_numeric(df[cols.clicks], errors="coerce").fillna(0).astype(float)
	if cols.revenue and cols.revenue in df.columns:
		df[cols.revenue] = pd.to_numeric(df[cols.revenue], errors="coerce").fillna(0.0)
	return df


def resample_to_daily(df: pd.DataFrame, cols: ColumnConfig) -> pd.DataFrame:
	"""Resample per typetag to daily frequency filling missing dates with zeros for clicks (and revenue if present)."""
	grouped_frames: List[pd.DataFrame] = []
	for tag_value, tag_df in df.groupby(cols.tag, sort=False):
		idx = pd.date_range(tag_df[cols.date].min(), tag_df[cols.date].max(), freq="D")
		tag_df = tag_df.set_index(cols.date).sort_index()
		resampled = tag_df.reindex(idx)
		resampled.index.name = cols.date
		resampled[cols.tag] = tag_value
		# Fill numeric columns
		resampled[cols.clicks] = resampled[cols.clicks].fillna(0.0)
		if cols.revenue and cols.revenue in resampled.columns:
			resampled[cols.revenue] = resampled[cols.revenue].fillna(0.0)
		grouped_frames.append(resampled.reset_index())
	return pd.concat(grouped_frames, ignore_index=True)


def add_rolling_features(df: pd.DataFrame, cols: ColumnConfig, params: Params) -> pd.DataFrame:
	df = df.sort_values([cols.tag, cols.date]).copy()
	group = df.groupby(cols.tag)[cols.clicks]
	# Previous-day rolling means to avoid same-day leakage
	df["ma_short_prev"] = group.transform(lambda s: s.rolling(params.ma_short, min_periods=1).mean().shift(1))
	df["ma_long_prev"] = group.transform(lambda s: s.rolling(params.ma_long, min_periods=1).mean().shift(1))
	# 7-day averages: last vs previous 7
	df["avg_last7"] = group.transform(lambda s: s.rolling(7, min_periods=1).mean())
	df["avg_prev7"] = group.transform(lambda s: s.shift(7).rolling(7, min_periods=1).mean())
	# Daily percent change and rolling std for volatility
	pct_change = group.transform(lambda s: s.replace(0, np.nan).pct_change())
	df["pct_change"] = pct_change
	# Volatility over last 14 days (std of pct changes)
	df["volatility14"] = df.groupby(cols.tag)["pct_change"].transform(lambda s: s.rolling(14, min_periods=1).std())
	return df


def compute_contribution_buckets(df: pd.DataFrame, cols: ColumnConfig, window_days: int, scheme: str, metric_col: str) -> pd.DataFrame:
	"""Compute contribution share per tag over the last window_days from global last date and assign buckets based on metric_col (e.g., clicks or revenue)."""
	last_date = df[cols.date].max()
	window_start = last_date - pd.Timedelta(days=window_days - 1)
	mask = (df[cols.date] >= window_start) & (df[cols.date] <= last_date)
	val_col = metric_col
	agg = (
		df.loc[mask, [cols.tag, val_col]]
		.groupby(cols.tag, as_index=False)
		.sum()
		.rename(columns={val_col: f"{val_col}_last_window"})
	)
	window_sum_col = f"{val_col}_last_window"
	total_val = float(agg[window_sum_col].sum()) or 1.0
	agg[f"{val_col}_share"] = agg[window_sum_col] / total_val
	agg = agg.sort_values(f"{val_col}_share", ascending=False).reset_index(drop=True)
	agg[f"{val_col}_cum_share"] = agg[f"{val_col}_share"].cumsum()
	if scheme == "80_15_5":
		conditions = [
			agg[f"{val_col}_cum_share"] <= 0.80,
			(agg[f"{val_col}_cum_share"] > 0.80) & (agg[f"{val_col}_cum_share"] <= 0.95),
		]
		choices = ["head", "torso"]
		agg["bucket"] = np.select(conditions, choices, default="tail")
	elif scheme == "decile":
		agg["bucket"] = pd.qcut(agg[f"{val_col}_share"], 10, labels=[f"Q{i}" for i in range(1, 11)])
	else:
		raise ValueError(f"Unknown bucket scheme: {scheme}")
	return agg[[cols.tag, f"{val_col}_share", f"{val_col}_cum_share", "bucket", window_sum_col]]


def _first_nonzero_date(s: pd.Series, dates: pd.Series) -> Optional[pd.Timestamp]:
	mask = s > 0
	if mask.any():
		return dates.loc[mask].iloc[0]
	return None


def derive_signals(df: pd.DataFrame, cols: ColumnConfig, params: Params) -> Tuple[pd.DataFrame, Dict[str, float]]:
	last_date = df[cols.date].max()
	lookback_start = last_date - pd.Timedelta(days=params.lookback_days - 1)
	disappear_start = last_date - pd.Timedelta(days=params.disappear_days - 1)
	come_back_start = last_date - pd.Timedelta(days=params.ma_long * 2 - 1)

	# Aggregate for last/prev 7-day comparisons per tag
	def tag_summary(tag_df: pd.DataFrame) -> Dict[str, float]:
		# Ensure sorted by date
		tag_df = tag_df.sort_values(cols.date)
		# Windows
		last7_mask = tag_df[cols.date] > last_date - pd.Timedelta(days=7)
		prev7_mask = (tag_df[cols.date] <= last_date - pd.Timedelta(days=7)) & (
			tag_df[cols.date] > last_date - pd.Timedelta(days=14)
		)
		avg_last7 = tag_df.loc[last7_mask, cols.clicks].mean() if last7_mask.any() else 0.0
		avg_prev7 = tag_df.loc[prev7_mask, cols.clicks].mean() if prev7_mask.any() else 0.0
		current_clicks = float(tag_df.loc[tag_df[cols.date] == last_date, cols.clicks].iloc[0])
		ma_long_prev = float(tag_df.loc[tag_df[cols.date] == last_date, "ma_long_prev"].iloc[0])
		ratio = (current_clicks + 1.0) / (ma_long_prev + 1.0)
		trend_ratio = (avg_last7 + 1.0) / (avg_prev7 + 1.0)
		# Volatility: std of pct_change over last 14 days
		vol14 = float(tag_df.loc[tag_df[cols.date] == last_date, "volatility14"].iloc[0])

		# Disappearance: zero clicks in last N days but had activity before
		recent_window = tag_df[tag_df[cols.date] >= disappear_start]
		prior_window = tag_df[tag_df[cols.date] < disappear_start]
		disappeared = (recent_window[cols.clicks].sum() <= 0) and (prior_window[cols.clicks].sum() > 0)

		# Drop and comeback: had low vs ma_long_prev then recovered within last N days
		lookback_df = tag_df[tag_df[cols.date] >= come_back_start]
		low_days = (lookback_df[cols.clicks] <= 0.5 * lookback_df["ma_long_prev"]) & (
			lookback_df["ma_long_prev"] > 0
		)
		come_back_days = (lookback_df[cols.clicks] >= 0.8 * lookback_df["ma_long_prev"]) & (
			lookback_df["ma_long_prev"] > 0
		)
		come_back = low_days.any() and lookback_df.tail(params.come_back_days).index.isin(lookback_df.index[come_back_days]).any()

		# Score: log-ratio based, bounded
		base = 80.0 * math.log(max(ratio, 1e-9)) + 20.0 * math.log(max(trend_ratio, 1e-9))
		# Clamp score
		score = float(np.clip(base, -100.0, 100.0))
		if disappeared:
			score = min(score, -80.0)

		delta7 = (avg_last7 - avg_prev7) / (avg_prev7 + 1.0)
		is_highly_volatile = (
			abs(score) >= 40.0
			or disappeared
			or come_back
			or abs(delta7) >= 0.5
		)

		return {
			"current_clicks": current_clicks,
			"ma_long_prev": ma_long_prev,
			"ratio_curr_to_ma": ratio,
			"avg_last7": avg_last7,
			"avg_prev7": avg_prev7,
			"trend_ratio": trend_ratio,
			"volatility14": vol14 if not np.isnan(vol14) else 0.0,
			"delta7": delta7,
			"disappeared_recently": bool(disappeared),
			"drop_and_comeback": bool(come_back),
			"volatility_score": score,
			"is_highly_volatile": bool(is_highly_volatile),
		}

	# Build summary per tag (for tags present on last_date)
	last_day_df = df[df[cols.date] == last_date]
	active_tags = last_day_df[cols.tag].unique().tolist()
	per_tag_records: List[Dict] = []
	for tag_value in active_tags:
		tag_df = df[df[cols.tag] == tag_value]
		rec = tag_summary(tag_df)
		rec[cols.tag] = tag_value
		per_tag_records.append(rec)
	per_tag = pd.DataFrame(per_tag_records)

	# Contribution buckets based on clicks
	contrib_clicks = compute_contribution_buckets(
		df, cols, window_days=params.ma_long, scheme=params.bucket_scheme, metric_col=cols.clicks
	)
	per_tag = per_tag.merge(contrib_clicks, on=cols.tag, how="left")
	# If revenue present, compute revenue share too (no rebucketing; keep click-based bucket)
	if cols.revenue and cols.revenue in df.columns:
		contrib_rev = compute_contribution_buckets(
			df, cols, window_days=params.ma_long, scheme=params.bucket_scheme, metric_col=cols.revenue
		)
		# Suffix revenue columns to avoid overlap
		contrib_rev = contrib_rev.rename(
			columns={
				f"{cols.revenue}_share": "revenue_share",
				f"{cols.revenue}_cum_share": "revenue_cum_share",
				f"{cols.revenue}_last_window": "revenue_last_window",
			}
		)
		per_tag = per_tag.merge(contrib_rev[[cols.tag, "revenue_share", "revenue_cum_share", "revenue_last_window"]], on=cols.tag, how="left")

	# New vs old tags
	first_seen = (
		df[df[cols.clicks] > 0]
		.groupby(cols.tag)[cols.date]
		.min()
		.rename("first_seen_date")
		.reset_index()
	)
	per_tag = per_tag.merge(first_seen, on=cols.tag, how="left")
	per_tag["is_new_tag"] = per_tag["first_seen_date"] >= (last_date - pd.Timedelta(days=params.new_tag_days - 1))

	# Dataset-level metrics
	volatile_mask = per_tag["is_highly_volatile"].fillna(False)
	clicks_last_window_sum = float(per_tag.get(f"{cols.clicks}_last_window", pd.Series(dtype=float)).sum() or 1.0)
	volatile_share_of_clicks = float(per_tag.loc[volatile_mask, f"{cols.clicks}_last_window"].sum()) / clicks_last_window_sum
	pct_volatile_tags = float(volatile_mask.mean())

	new_mask = per_tag["is_new_tag"].fillna(False)
	pct_new_tags = float(new_mask.mean())
	new_tags_clicks_share = float(per_tag.loc[new_mask, f"{cols.clicks}_last_window"].sum()) / clicks_last_window_sum

	# Revenue shares if present
	revenue_metrics: Dict[str, float] = {}
	if cols.revenue and "revenue_last_window" in per_tag.columns:
		revenue_last_window_sum = float(per_tag["revenue_last_window"].sum() or 1.0)
		revenue_metrics["volatile_share_of_revenue"] = float(per_tag.loc[volatile_mask, "revenue_last_window"].sum()) / revenue_last_window_sum
		revenue_metrics["new_tags_revenue_share"] = float(per_tag.loc[new_mask, "revenue_last_window"].sum()) / revenue_last_window_sum
	else:
		revenue_last_window_sum = 0.0

	# Dataset High Volatility label
	dataset_is_highly_volatile = (
		pct_volatile_tags >= params.dataset_pct_volatile_thresh or volatile_share_of_clicks >= params.dataset_volatile_share_thresh
	)

	insights: List[str] = []
	# High churn insight for clicks
	if pct_new_tags >= params.insight_new_tags_high_pct and new_tags_clicks_share <= params.insight_new_tags_low_share:
		insights.append(
			f"High churn: {pct_new_tags:.0%} of typetags are new but contribute only {new_tags_clicks_share:.0%} of clicks"
		)
	# High churn for revenue if available
	if revenue_metrics.get("new_tags_revenue_share") is not None:
		new_tags_rev_share = revenue_metrics["new_tags_revenue_share"]
		if pct_new_tags >= params.insight_new_tags_high_pct and new_tags_rev_share <= params.insight_new_tags_low_share:
			insights.append(
				f"High churn: {pct_new_tags:.0%} of typetags are new but contribute only {new_tags_rev_share:.0%} of revenue"
			)
	# Head bucket disappearance
	head = per_tag[per_tag["bucket"] == "head"]
	if not head.empty:
		head_disappear_frac = float(head["disappeared_recently"].mean())
		if head_disappear_frac >= 0.10:
			insights.append(
				f"Head bucket instability: {head_disappear_frac:.0%} of head tags disappeared in the last {params.disappear_days} days"
			)

	metrics = {
		"dataset": {
			"last_date": str(last_date.date()),
			"pct_volatile_tags": pct_volatile_tags,
			"volatile_share_of_clicks": volatile_share_of_clicks,
			"pct_new_tags": pct_new_tags,
			"new_tags_clicks_share": new_tags_clicks_share,
			"is_highly_volatile": bool(dataset_is_highly_volatile),
			"insights": insights,
		},
	}
	if revenue_metrics:
		metrics["dataset"].update(revenue_metrics)
	return per_tag, metrics


def merge_buckets_counts(per_tag: pd.DataFrame, cols: ColumnConfig) -> pd.DataFrame:
	counts = (
		per_tag.groupby(["bucket", "is_new_tag"], dropna=False)
		.size()
		.rename("num_typetags")
		.reset_index()
	)
	return counts


def write_outputs(per_tag: pd.DataFrame, metrics: Dict, output_dir: str, cols: ColumnConfig) -> None:
	os.makedirs(output_dir, exist_ok=True)
	per_tag_path = os.path.join(output_dir, "per_tag_metrics.csv")
	per_tag.to_csv(per_tag_path, index=False)
	with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
		json.dump(metrics, f, indent=2)
	# Human-readable insights
	lines: List[str] = []
	lines.append(f"Last date: {metrics['dataset']['last_date']}")
	lines.append(
		f"Volatile tags: {metrics['dataset']['pct_volatile_tags']:.0%} contributing {metrics['dataset']['volatile_share_of_clicks']:.0%} of clicks"
	)
	if "volatile_share_of_revenue" in metrics["dataset"]:
		lines.append(
			f"Volatile tags contribute {metrics['dataset']['volatile_share_of_revenue']:.0%} of revenue"
		)
	lines.append(
		f"New tags: {metrics['dataset']['pct_new_tags']:.0%} contributing {metrics['dataset']['new_tags_clicks_share']:.0%} of clicks"
	)
	if "new_tags_revenue_share" in metrics["dataset"]:
		lines.append(
			f"New tags contribute {metrics['dataset']['new_tags_revenue_share']:.0%} of revenue"
		)
	if metrics["dataset"]["is_highly_volatile"]:
		lines.append("Dataset labeled: HIGH VOLATILITY")
	else:
		lines.append("Dataset labeled: normal volatility")
	if metrics["dataset"]["insights"]:
		lines.append("")
		lines.append("Insights:")
		for msg in metrics["dataset"]["insights"]:
			lines.append(f"- {msg}")
	with open(os.path.join(output_dir, "alerts.md"), "w", encoding="utf-8") as f:
		f.write("\n".join(lines))


def generate_synthetic_dataset(num_tags: int = 40, num_days: int = 90, seed: int = 7) -> pd.DataFrame:
	rng = np.random.default_rng(seed)
	start_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=num_days - 1)
	dates = pd.date_range(start_date, periods=num_days, freq="D")
	rows: List[Dict] = []
	for i in range(num_tags):
		tag = f"tag_{i:03d}"
		base = rng.lognormal(mean=2.0, sigma=0.7)  # base clicks
		trend = rng.normal(loc=0.0, scale=0.01)
		season = rng.normal(loc=0.0, scale=0.05)
		series = []
		level = base
		for d in range(num_days):
			noise = rng.normal(0, 0.15)
			seasonal = 1.0 + season * math.sin(2 * math.pi * d / 7.0)
			level *= (1.0 + trend + noise)
			val = max(0.0, level * seasonal)
			series.append(val)
		# Inject patterns: some disappear, some drop & comeback
		if i % 10 == 0:  # disappear in last 7 days
			series[-7:] = [0.0] * 7
		elif i % 10 == 1:  # drop then recover
			drop_idx = -14
			series[drop_idx:-7] = [s * 0.2 for s in series[drop_idx:-7]]
		elif i % 10 == 2:  # surge
			series[-10:] = [s * 2.0 for s in series[-10:]]
		for d, val in enumerate(series):
			rows.append({
				"date": dates[d],
				"typetag": tag,
				"clicks": float(val),
				"revenue": float(val * rng.uniform(0.05, 0.15)),
			})
	return pd.DataFrame(rows)


def run_analysis(df: pd.DataFrame, cols: ColumnConfig, params: Params, output_dir: str) -> Tuple[pd.DataFrame, Dict]:
	df = normalize_columns(df, cols)
	df = resample_to_daily(df, cols)
	df = add_rolling_features(df, cols, params)
	per_tag, metrics = derive_signals(df, cols, params)
	# Add bucket new/old counts table for convenience
	bucket_counts = merge_buckets_counts(per_tag, cols)
	metrics["buckets"] = {
		"counts": bucket_counts.to_dict(orient="records")
	}
	write_outputs(per_tag, metrics, output_dir, cols)
	return per_tag, metrics


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Typetag volatility analysis: moving averages, buckets, new/old stats, scores, disappearance flags, and labels.",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument("--input", type=str, default="", help="Path to input CSV/Parquet file")
	parser.add_argument("--output-dir", type=str, default="/workspace/output_typetags", help="Directory to write outputs")
	parser.add_argument("--date-col", type=str, default="date")
	parser.add_argument("--tag-col", type=str, default="typetag")
	parser.add_argument("--clicks-col", type=str, default="clicks")
	parser.add_argument("--revenue-col", type=str, default="", help="Optional revenue column")
	parser.add_argument("--downgrade-col", type=str, default="", help="Optional downgrade reason label column")
	parser.add_argument("--lookback-days", type=int, default=Params.lookback_days)
	parser.add_argument("--ma-short", type=int, default=Params.ma_short)
	parser.add_argument("--ma-long", type=int, default=Params.ma_long)
	parser.add_argument("--disappear-days", type=int, default=Params.disappear_days)
	parser.add_argument("--new-tag-days", type=int, default=Params.new_tag_days)
	parser.add_argument("--bucket-scheme", type=str, choices=["80_15_5", "decile"], default=Params.bucket_scheme)
	# Thresholds
	parser.add_argument("--dataset-pct-volatile-thresh", type=float, default=Params.dataset_pct_volatile_thresh)
	parser.add_argument("--dataset-volatile-share-thresh", type=float, default=Params.dataset_volatile_share_thresh)
	parser.add_argument("--insight-new-tags-high-pct", type=float, default=Params.insight_new_tags_high_pct)
	parser.add_argument("--insight-new-tags-low-share", type=float, default=Params.insight_new_tags_low_share)
	parser.add_argument("--demo", action="store_true", help="Run on synthetic demo data")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	cols = ColumnConfig(
		date=args.date_col,
		tag=args.tag_col,
		clicks=args.clicks_col,
		revenue=(args.revenue_col or None),
		downgrade_reason=(args.downgrade_col or None),
	)
	params = Params(
		lookback_days=args.lookback_days,
		ma_short=args.ma_short,
		ma_long=args.ma_long,
		disappear_days=args.disappear_days,
		new_tag_days=args.new_tag_days,
		bucket_scheme=args.bucket_scheme,
		dataset_pct_volatile_thresh=args.dataset_pct_volatile_thresh,
		dataset_volatile_share_thresh=args.dataset_volatile_share_thresh,
		insight_new_tags_high_pct=args.insight_new_tags_high_pct,
		insight_new_tags_low_share=args.insight_new_tags_low_share,
	)

	if args.demo:
		df = generate_synthetic_dataset()
	else:
		if not args.input:
			raise SystemExit("--input is required unless --demo is provided")
		df = read_input(args.input)

	_ = run_analysis(df, cols, params, args.output_dir)
	print(f"Wrote outputs to: {args.output_dir}")


if __name__ == "__main__":
	main()