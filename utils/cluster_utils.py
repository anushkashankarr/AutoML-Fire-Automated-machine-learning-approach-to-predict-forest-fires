from imports import *
from config.config import CFG

def extract_ecozones(df, eps=None, min_samples=None):
    print("[i] hybrid clustering (probabilistic routing)...")

    # ========= 1) COARSE CLUSTERING (fast) ==========
    k = max(9, min(15, int(np.sqrt(len(df)/40000))))
    coords = df[['latitude', 'longitude']].values

    km = KMeans(n_clusters=k, random_state=42)
    df['macro_zone'] = km.fit_predict(coords)
    print(f"[i] KMeans macro clusters = {k}")

    zones_full = np.full(len(df), -1)
    zone_counter = 0

    # ========= 2) DBSCAN inside each macro region ==========
    for mz in range(k):
        sub = df[df['macro_zone'] == mz]

        if len(sub) < 2000:
            continue

        coords_rad = np.radians(sub[['latitude', 'longitude']])

        radius_km = eps if eps is not None else 15
        min_pts = min_samples if min_samples is not None else 80

        db = DBSCAN(
            eps=radius_km / 6371.0,
            min_samples=min_pts,
            metric="haversine"
        ).fit(coords_rad)

        labels = db.labels_
        if labels is None:
            continue

        for cid in np.unique(labels):
            if cid == -1:  # skip noise
                continue

            mask = (labels == cid)
            if mask.sum() < 150:  # small ignored
                continue
            zones_full[sub.index[mask]] = zone_counter
            zone_counter += 1

    print(f"[i] DBSCAN refined ecozones =", len(np.unique(zones_full[zones_full != -1])))

    # ========= 3) Assign remaining using probabilistic router ==========
    valid = df[zones_full != -1]

    if len(valid) == 0:
        print("[!] No DBSCAN ecozones — falling back to macro KNN clusters.")
        df['ecozone'] = df['macro_zone']
        final_count = df['ecozone'].nunique()
        print(f"[i] Final ecozone count = {final_count}")
        return df, None, None

    # ===== Spatially-corrected coordinates (longitude shrinkage) =====
    valid_xy = valid.copy()
    valid_xy['x'] = valid['longitude'] * np.cos(np.radians(valid['latitude']))
    valid_xy['y'] = valid['latitude']

    df_xy = df.copy()
    df_xy['x'] = df['longitude'] * np.cos(np.radians(df['latitude']))
    df_xy['y'] = df['latitude']

    # ========= routing remaining points using kNN ==========
    knn = KNeighborsClassifier(n_neighbors=40, weights="distance")

    valid_labels = zones_full[valid.index]
    knn.fit(valid_xy[['x', 'y']], valid_labels)

    df['ecozone'] = knn.predict(df_xy[['x', 'y']])

    print("[i] Final ecozone count =", df['ecozone'].nunique())

    return df, None, knn

def assign_zones(df, router):
    print("[i] Assigning ecozones via router…")
    df["ecozone"] = router.predict(df[["latitude","longitude"]])
    return df

def train_ecozone_knn(df, outdir):
    print("[i] Training ecozone KNN router …")
    
    k = max(20, int(np.sqrt(len(df)) / 3))
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
    knn.fit(df[['latitude','longitude']], df['ecozone'])
    
    joblib.dump(knn, os.path.join(outdir, "ecozone_knn.pkl"))
    return knn

def compute_centroids(df):
    centroids = {}
    for z in sorted(df["ecozone"].unique()):
        if z == -1: 
            continue
        part = df[df["ecozone"] == z]
        centroids[int(z)] = (
            float(part["latitude"].mean()),
            float(part["longitude"].mean())
        )
    return centroids



