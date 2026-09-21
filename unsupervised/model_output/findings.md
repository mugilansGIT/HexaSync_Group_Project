# Clustering & ML Findings — Music Listener Segmentation

**Chosen k:** 3 (selected via Elbow Method + Silhouette Analysis)

**Final silhouette score:** 0.5530

## Cluster Centers (original units)

|   cluster |   listening_hours_per_week |   songs_per_day |   skip_rate |   playlist_count |
|----------:|---------------------------:|----------------:|------------:|-----------------:|
|         0 |                      -0.07 |           -0.06 |       -0.25 |             0.06 |
|         1 |                       1.43 |            1.43 |       -1.19 |             1.33 |
|         2 |                      -0.97 |           -0.98 |        1.12 |            -1.03 |

## Segment Interpretation

### Cluster 0 — Regular Listeners (11 listeners)
- Listening hours/week: -0.1
- Songs/day: -0.1
- Skip rate: -0.25
- Playlist count: 0.1

### Cluster 1 — Power Listeners (8 listeners)
- Listening hours/week: 1.4
- Songs/day: 1.4
- Skip rate: -1.19
- Playlist count: 1.3

### Cluster 2 — Casual Listeners (11 listeners)
- Listening hours/week: -1.0
- Songs/day: -1.0
- Skip rate: 1.12
- Playlist count: -1.0

