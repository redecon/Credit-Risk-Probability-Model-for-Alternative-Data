import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def create_proxy_target(df: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    """
    Create proxy target variable 'is_high_risk' using RFM clustering.
    """

    # Ensure TransactionStartTime is datetime and drop timezone info
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"]).dt.tz_localize(None)

    # Build RFM table
    rfm = df.groupby("CustomerId").agg(
        Recency=("TransactionStartTime", lambda x: (snapshot_date - x.max()).days),
        Frequency=("TransactionStartTime", "count"),
        Monetary=("Amount", "sum"),
    ).reset_index()

    # Scale features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    # Cluster customers
    kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

    # Identify high-risk cluster (low Frequency, low Monetary, high Recency)
    cluster_summary = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    high_risk_cluster = cluster_summary.sort_values(["Frequency", "Monetary"]).index[0]
    rfm["is_high_risk"] = (rfm["Cluster"] == high_risk_cluster).astype(int)

    # Merge target back into main dataset
    df_with_target = df.merge(rfm[["CustomerId", "is_high_risk"]], on="CustomerId", how="inner")

    return df_with_target
