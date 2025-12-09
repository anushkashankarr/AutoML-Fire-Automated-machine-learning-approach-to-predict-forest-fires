from imports import *
from config.config import CFG

def generate_zone_report(zone_dir, out_html):
    print("[i] Generating zone report:", zone_dir)

    try:
        meta = json.load(open(os.path.join(zone_dir,"classifier_meta.json")))
    except:
        meta = {}

    try:
        meta_r = json.load(open(os.path.join(zone_dir,"regressor_meta.json")))
    except:
        meta_r = {}

    imgs = [f for f in os.listdir(zone_dir) if f.endswith(".png")]

    img_html = ""
    for f in imgs:
        label = f.replace(".png","").replace("_"," ").title()
        img_html += f"<h3>{label}</h3><img src='{f}' width='600'/><br>"

    html = f"""
    <html>
    <body>
    <h1>Zone Report: {os.path.basename(zone_dir)}</h1>

    <h2>Classifier Summary</h2>
    <pre>{json.dumps(meta, indent=2)}</pre>

    <h2>Regressor Summary</h2>
    <pre>{json.dumps(meta_r, indent=2)}</pre>

    <h2>Diagnostics & Training Curves</h2>
    {img_html}
    </body>
    </html>
    """

    with open(out_html,"w",encoding="utf-8") as f:
        f.write(html)

    print("[i] Report written:", out_html)

def generate_global_reports(registry, outdir):
    print("[i] Building ULTIMATE global performance reports...")

    import numpy as np
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt

    global_dir = os.path.join(outdir, "global")
    os.makedirs(global_dir, exist_ok=True)

    summary_rows = []
    
    all_y_true = []
    all_y_prob = []
    spatial_data = []

    # Extract stored centroid structure safely
    centroids = registry.get("centroids", {})

    for zone in sorted([z for z in registry.keys() if isinstance(z, (int, np.integer))]):
        entry = registry[zone] or {}

        mclf = entry.get("metrics_clf", {}) or {}
        mreg = entry.get("metrics_reg", {}) or {}

        tn = mclf.get("tn", 0)
        fp = mclf.get("fp", 0)
        fn = mclf.get("fn", 0)
        tp = mclf.get("tp", 0)
        total_samples = tn + fp + fn + tp
        fire_rate = (tp + fn) / max(1, total_samples)

        # Collect prediction distributions
        if "y_true" in entry and "y_prob" in entry:
            yt, yp = entry["y_true"], entry["y_prob"]
            if yt is not None and yp is not None:
                all_y_true.extend(list(yt))
                all_y_prob.extend(list(yp))

        # Safe centroid lookup
        if isinstance(centroids, dict) and zone in centroids:
            lat, lon = centroids[zone]
        else:
            lat, lon = (np.nan, np.nan)

        if not np.isnan(lat):
            spatial_data.append({
                "zone": zone, "lat": lat, "lon": lon,
                "pr_auc": mclf.get("pr_auc", np.nan)
            })

        summary_rows.append({
            "zone": zone,
            "total_samples": total_samples,
            "fire_rate": fire_rate,
            "latitude": lat,

            # CLASSIFICATION
            "pr_auc": mclf.get("pr_auc", np.nan),
            "f2": mclf.get("f2", np.nan),
            "accuracy": mclf.get("accuracy", np.nan),
            "precision": mclf.get("precision", np.nan),
            "recall": mclf.get("recall", np.nan),

            # REGRESSION
            "rmse": mreg.get("rmse", np.nan),
            "r2": mreg.get("r2", np.nan),

            # VALIDATION STABILITY
            "cv_mean": mclf.get("cv_pr_auc_mean", np.nan),
            "cv_std": mclf.get("cv_pr_auc_std", np.nan),

            # STORE counts so global CM works!
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(global_dir, "performance_summary.csv"), index=False)

    # ============================================================================
    # 1. METRIC RANGE BOXPLOTS
    # ============================================================================
    print("[i] Generating metric range boxplots...")
    try:
        melted = summary_df.melt(
            id_vars=["zone"],
            value_vars=["pr_auc", "f2", "accuracy", "recall", "precision"],
            var_name="Metric", value_name="Score"
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=melted, x="Metric", y="Score", ax=ax, palette="Set2")
        sns.stripplot(data=melted, x="Metric", y="Score",
                      color="black", jitter=True, alpha=0.3, ax=ax)

        ax.set_title("Global Performance Distribution (Range Summary)")
        ax.set_ylim(0, 1.05)
        save_fig(fig, os.path.join(global_dir, "global_metric_ranges.png"))
    except Exception as e:
        print("[!] Boxplots failed:", e)

    # ============================================================================
    # 2. META CORR HEATMAP
    # ============================================================================
    print("[i] Generating meta-correlation heatmap...")
    try:
        meta_cols = ["total_samples", "fire_rate", "latitude",
                     "pr_auc", "f2", "cv_std"]
        corr_df = summary_df[meta_cols].dropna().corr()

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f",
                    center=0, ax=ax)
        ax.set_title("Meta-Analysis: Drivers of Zone Performance")
        save_fig(fig, os.path.join(global_dir, "meta_correlation_matrix.png"))
    except Exception as e:
        print("[!] Heatmap failed:", e)

    # ============================================================================
    # 3. SAMPLE SIZE VS PERFORMANCE
    # ============================================================================
    try:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(data=summary_df,
                        x="total_samples", y="pr_auc",
                        alpha=0.6, s=120, ax=ax)

        ax.set_xscale("log")
        ax.set_title("Data Efficiency: Sample Size vs PR-AUC")
        ax.set_xlabel("Log(Samples)")
        save_fig(fig, os.path.join(global_dir, "samples_vs_performance.png"))
    except Exception as e:
        print("[!] Scatter failed:", e)

    # ============================================================================
    # 4. GLOBAL PROBABILITY HISTOGRAM
    # ============================================================================
    try:
        if len(all_y_prob) > 0:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(all_y_prob, bins=50, kde=True, ax=ax)
            ax.set_title("Global Fire Probability Distribution")
            ax.set_xlabel("Predicted Fire Probability")
            save_fig(fig, os.path.join(global_dir, "global_prediction_dist.png"))
    except Exception as e:
        print("[!] Global histogram failed:", e)

    # ============================================================================
    # 5. SPATIAL MAP
    # ============================================================================
    if len(spatial_data) > 0:
        try:
            sdf = pd.DataFrame(spatial_data)
            fig, ax = plt.subplots(figsize=(8, 6))
            pts = ax.scatter(sdf["lon"], sdf["lat"], c=sdf["pr_auc"],
                             cmap="RdYlGn", s=80, edgecolor="k")
            plt.colorbar(pts, label="PR-AUC")
            ax.set_title("Geographic Performance Map")
            save_fig(fig, os.path.join(global_dir, "spatial_performance_map.png"))
        except Exception as e:
            print("[!] Spatial plot failed:", e)

    # ============================================================================
    # 6. GLOBAL CONFUSION MATRIX
    # ============================================================================
    all_tn = summary_df["tn"].sum()
    all_fp = summary_df["fp"].sum()
    all_fn = summary_df["fn"].sum()
    all_tp = summary_df["tp"].sum()

    global_cm = np.array([[all_tn, all_fp],
                          [all_fn, all_tp]])

    pd.DataFrame(global_cm,
                 index=["Actual 0", "Actual 1"],
                 columns=["Pred 0", "Pred 1"]).to_csv(
        os.path.join(global_dir, "global_confusion_matrix.csv")
    )

    try:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(global_cm, annot=True, fmt=".0f",
                    cmap="Blues", ax=ax)
        ax.set_title("Global Confusion Matrix")
        save_fig(fig, os.path.join(global_dir, "global_confusion_matrix.png"))
    except Exception:
        pass

    # ============================================================================
    # 7. GLOBAL HTML DASHBOARD
    # ============================================================================
    print("[i] Writing global index page...")

    index_html = """
    <html><head><title>AutoML-ECOFIRE Global Analysis</title></head><body>
    <h1>Global Performance Summary</h1>
    <ul>
    """

    for z in sorted([z for z in registry.keys() if isinstance(z, (int, np.integer))]):
        index_html += f"<li><a href='../zone_{z}/report.html'>Zone {z} Report</a></li>"

    index_html += """
    </ul><hr>
    <p><a href='performance_summary.csv'>Download CSV Summary</a></p>
    </body></html>
    """

    with open(os.path.join(global_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(index_html)

    print("[i] DONE Global reporting.\n")

def save_zone_map(df_zone, save_path):
    try:
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.kdeplot(
            x=df_zone["longitude"],
            y=df_zone["latitude"],
            fill=True,
            cmap="Reds",
            thresh=0.05,
            ax=ax
        )
        ax.set_title("Spatial Heatmap of Samples")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

        print("[i] Saved zone map:", save_path)

    except Exception as e:
        print("[!] Failed zone map:", e)

def save_gis_overlay(df_zone, save_path, value_column="fire_detected"):
    """
    Creates geospatial scatter/heat overlay maps.

    value_column:
        - None          → plain geospatial map (no heatmap)
        - fire_detected → default classifier signal
        - shap_risk     → explainability overlay
    """
    try:
        print(f"[i] GIS overlay requested → {save_path} using '{value_column}'")

        # Validate coordinates exist
        if "latitude" not in df_zone or "longitude" not in df_zone:
            print("[!] GIS overlay aborted: missing latitude/longitude columns")
            return False

        df_zone = df_zone.copy()
        df_zone["latitude"]  = pd.to_numeric(df_zone["latitude"],  errors="coerce")
        df_zone["longitude"] = pd.to_numeric(df_zone["longitude"], errors="coerce")
        df_zone = df_zone.dropna(subset=["latitude", "longitude"])

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            df_zone,
            geometry=[Point(xy) for xy in zip(df_zone["longitude"], df_zone["latitude"])],
            crs="EPSG:4326"
        )

        fig, ax = plt.subplots(figsize=(7, 6))

        # Case 1: plain scatter if value_column=None
        if value_column is None:
            gdf.plot(
                ax=ax,
                markersize=6,
                color="red",
                alpha=0.6
            )
        else:
            # If column missing, fail gracefully
            if value_column not in gdf.columns:
                print(f"[!] GIS overlay aborted: missing value column '{value_column}'")
                return False

            df_zone[value_column] = pd.to_numeric(df_zone[value_column], errors="coerce")

            gdf.plot(
                ax=ax,
                column=value_column,
                cmap="inferno",
                legend=True,
                markersize=6,
                alpha=0.7
            )

        title = f"GIS Overlay — {value_column}" if value_column else "GIS Overlay"
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"[✓] GIS overlay actually saved → {save_path}")
        return True

    except Exception as e:
        print("[!] GIS overlay failed:", e)
        return False

def plot_india_risk(df, risk_col="fire_prob", save_path="india_risk_map.png"):
    """
    India ke map par risk_col ke basis pe heatmap (scatter) banata hai.
    Example: risk_col = 'fire_prob' ya 'shap_risk'
    """
    required = {"latitude", "longitude", risk_col}
    if not required <= set(df.columns):
        print(f"[!] plot_india_risk: df missing columns: {required - set(df.columns)}")
        return

    df = df.copy()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df[risk_col] = pd.to_numeric(df[risk_col], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude", risk_col])

    india = _get_india_shape()

    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df["longitude"], df["latitude"])],
        crs="EPSG:4326",
    )

    fig, ax = plt.subplots(figsize=(7, 9))

    # base India
    india.plot(ax=ax, color="white", edgecolor="black", linewidth=0.8)

    # risk overlay
    gdf.plot(
        ax=ax,
        column=risk_col,
        cmap="inferno",
        markersize=5,
        alpha=0.8,
        legend=True,
    )

    ax.set_title(f"High-Risk Areas over India ({risk_col})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"[i] Saved India risk map → {save_path}")
 
def plot_india_zones(df, save_path="india_ecozones.png"):
    """
    India ke map par har sample ka ecozone dikhata hai.
    df: columns required -> ['latitude', 'longitude', 'ecozone']
    """
    # safety
    if not {"latitude", "longitude", "ecozone"} <= set(df.columns):
        print("[!] plot_india_zones: df missing latitude/longitude/ecozone")
        return

    # clean coords
    df = df.copy()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])

    india = _get_india_shape()

    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df["longitude"], df["latitude"])],
        crs="EPSG:4326",
    )

    fig, ax = plt.subplots(figsize=(7, 9))

    # base India boundary
    india.plot(ax=ax, color="white", edgecolor="black", linewidth=0.8)

    # ecozone points
    gdf.plot(
        ax=ax,
        column="ecozone",
        markersize=5,
        alpha=0.7,
        legend=True,
        cmap="tab20",
    )

    ax.set_title("Ecozones over India")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"[i] Saved India ecozone map → {save_path}")

def save_curve(data, title, xlabel, ylabel, save_path):
    try:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(range(1, len(data) + 1), data, marker="o")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        save_fig(fig, save_path)
    except Exception as e:
        print("[!] Curve save failed:", e)

def save_gis_overlay(df_zone, save_path, value_column="fire_detected"):
    """
    Robust GIS overlay. 
    1. Tries GeoPandas (Heatmap).
    2. Falls back to Matplotlib Scatter if GeoPandas fails.
    """
    # Safety Check: Data existence
    if "latitude" not in df_zone.columns or "longitude" not in df_zone.columns:
        print("[!] GIS Aborted: Missing 'latitude' or 'longitude' columns.")
        return
    if value_column not in df_zone.columns:
        print(f"[!] GIS Aborted: Missing value column '{value_column}'.")
        return

    # 1. Try GeoPandas
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            df_zone,
            geometry=[Point(xy) for xy in zip(df_zone["longitude"], df_zone["latitude"])],
            crs="EPSG:4326"
        )
        
        fig, ax = plt.subplots(figsize=(8, 6))
        gdf.plot(
            ax=ax,
            column=value_column,
            markersize=20,
            cmap="inferno",
            legend=True,
            alpha=0.7,
            edgecolor='none'
        )
        ax.set_title(f"GIS Overlay: {value_column}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        
        # Save
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        return # Success

    except ImportError:
        print("[!] GeoPandas not installed. Falling back to Scatter Plot.")
    except Exception as e:
        print(f"[!] GeoPandas overlay failed: {e}. Falling back to Scatter Plot.")

    # 2. Fallback: Standard Scatter Plot
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(
            df_zone["longitude"], 
            df_zone["latitude"], 
            c=df_zone[value_column], 
            cmap="inferno", 
            s=20, 
            alpha=0.7
        )
        plt.colorbar(sc, label=value_column)
        ax.set_title(f"Spatial Map: {value_column}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"[i] Saved GIS fallback map: {save_path}")

    except Exception as e:
        print(f"[!] GIS Fallback failed: {e}")

def _get_india_shape():
    """
    Natural Earth dataset se India polygon le aata hai.
    CRS = EPSG:4326 (lat/lon)
    """
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    india = world[world["name"] == "India"].to_crs(epsg=4326)
    return india
