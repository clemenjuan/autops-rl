# Phase 2: JEPA Maneuver Classification + Orbit Visualization
## Executive Summary

## 8 Subphases

1. **Propagation pipeline** - Orekit/GMAT orbit refinement → fused time series
2. **Synthetic maneuvers** - 50k GMAT/Orekit-simulated sequences (10k per class)
3. **Data preparation** - Merge real + synthetic, normalize, split 70/15/15
4. **JEPA training** - Temporal encoder → 5-class classifier (>85% accuracy)
5. **Validation** - Benchmark vs Phase 1 threshold baseline
6. **CoALA integration** - /maneuvers API endpoint for predictions
7. **Orbit visualization** - /orbits/* endpoints + Cesium.js 3D UI (< 100ms latency)
8. **Monitoring** - Drift detection, monthly retraining

## Subphase 7: Orbit Visualization (NEW)

**Why:** Operators validate JEPA predictions visually, not numerically.

**What:**
- Render pre-maneuver orbit (green) + post-maneuver orbit (yellow, opacity=confidence)
- Interactive 3D globe (Cesium.js) with time slider
- Maneuver timeline panel + detailed prediction modal
- API latency: < 100ms (single sat), < 1s (batch 100)

**How:**
- orbit_renderer.py: Convert JEPA predictions → GeoJSON trajectories
- cesium_formatter.py: Add styling (colors, opacity based on confidence)
- Cesium.js: Render 3D globe, handle user interactions
- PostgreSQL: Store orbital state + predictions

**Result:**
Operators can query "Which STARLINK sats maneuvered?" → CoALA responds → Operator clicks → **Sees 3D orbits in Cesium → Validates prediction → Proceeds with mission planning**

## Timeline: ~17 weeks
- Subphase 1-6: 14 weeks
- **Subphase 7 (Visualization): 2 weeks**
- Subphase 8: 1 week

## Success Criteria
- JEPA > 85% accuracy
- Visualization latency < 100ms (single), < 1s (batch)
- Trajectory accuracy ± 1 km (LEO)
- Cesium.js interactive + mobile-responsive
- Operator workflow validated (timeline → details → orbit view)
