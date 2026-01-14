# Phase 2: Autonomous Maneuver Classification via JEPA
**Implementation Specification (Grosvenor et al. Methodology)**

---

## Overview

Phase 2 replaces Phase 1's threshold-based maneuver detection (Δa > 0.01 km, Δi > 0.005°) with a **Joint Embedding Predictive Architecture (JEPA)** model trained on high-fidelity orbital time series. Following the Grosvenor et al. methodology, the system combines **TLE catalog data + numerical orbit propagation + synthetic GMAT maneuvers** to train a robust classifier that learns to distinguish propulsive maneuvers from decay/perturbations/noise.

**Primary objectives:**
- Integrate GMAT/Orekit orbit propagator for high-fidelity orbital snapshots
- Collect 6+ months TLE history + synthetic maneuver simulations (~50k training sequences)
- Train JEPA encoder on standardized orbital element time series
- Implement predictive classifier for maneuver type and confidence
- **Visualize predicted orbits (pre/post-maneuver) for operator validation**
- Integrate predictions into CoALA reasoning for mission analysis
- Benchmark against Phase 1 threshold baseline

**Data sources:**
- TLE catalog: PostgreSQL `tle_history` from Phase 1 (real observations)
- Numerical propagation: GMAT/Orekit for orbit refinement and synthetic maneuvers
- Synthetic maneuvers: ~50k simulated station-keeping/reboost/decay sequences

**ML framework:** PyTorch with temporal JEPA encoder  
**Visualization:** Cesium.js for 3D orbit rendering  
**Integration:** REST API `/maneuvers` and `/orbits` endpoints  
**Deployment:** CPU inference in API layer (no GPU required initially)  

---

## Implementation Subphases

### Subphase 1: Orbit Propagation Pipeline & Data Fusion

[Content identical to previous specification...]

### Subphase 2: Synthetic Maneuver Generation via GMAT/Orekit

[Content identical to previous specification...]

### Subphase 3: Data Preparation & Feature Engineering

[Content identical to previous specification...]

### Subphase 4: JEPA Model Architecture & Training

[Content identical to previous specification...]

### Subphase 5: Validation & Baseline Comparison

[Content identical to previous specification...]

### Subphase 6: Integration with CoALA Tools

[Content identical to previous specification...]

### Subphase 7: Orbit Visualization & Prediction Rendering

**Objectives:**
- Render predicted orbit states as 3D trajectories (Cesium.js frontend)
- Show pre-maneuver vs. post-maneuver orbits for high-confidence predictions
- Enable operators to visually validate JEPA classifications
- Support historical maneuver timeline visualization

**Deliverables:**
- `agent/visualization/orbit_renderer.py`
  - Convert orbital elements → Cartesian position (propagate t0 to t0+days)
  - Compute position at discrete time steps (e.g., every 6 hours)
  - Return list of [lat, lon, alt, timestamp] for Cesium rendering
  - Support pre/post-maneuver trajectories (separate color-coded paths)
  - Use Orekit/GMAT propagator from Subphase 1

- `agent/api/main.py` - New visualization endpoints:
  - `GET /orbits/render/{norad_id}?days=7&include_prediction=true` → trajectory GeoJSON
  - `GET /orbits/maneuver_signature/{norad_id}?days=30` → pre/post-maneuver side-by-side
  - `GET /orbits/constellation/{constellation}?days=1` → batch rendering (up to 500 sats)
  - Response format: GeoJSON FeatureCollection with lat/lon/alt/timestamp

- Frontend enhancement (`templates/visualization.html`):
  - **Cesium.js 3D map:** Interactive globe with orbit trajectories
    - Green path: pre-maneuver (stable) orbit
    - Yellow/orange path: maneuver-induced change (intensity = confidence)
    - Real-time time-slider to replay orbital evolution
  - **Maneuver Timeline Panel:** 
    - Detected maneuvers sorted by detection date
    - JEPA classification, confidence score, Δa/Δi estimates
    - Click to view detailed orbit visualization
  - **Prediction Details Modal:**
    - Maneuver type + confidence (with uncertainty visualization)
    - Pre/post orbital elements (a, e, i, RAAN, AOP, mean anomaly)
    - Estimated Δv magnitude (derived from JEPA latent features or orbital changes)
    - Temporal signature plot: Δa, Δi vs. time (days)
    - Comparison to Phase 1 threshold detector result

- Visualization tests: `tests/test_orbit_visualization.py`
  - Verify trajectory coordinates satisfy orbital mechanics (apogee/perigee altitudes)
  - Test coordinate system transformations (ECI → lat/lon/alt, SGP4 ↔ Orekit)
  - Validate GeoJSON output format and timestamp precision
  - Latency benchmarks: single satellite < 100ms, batch (100) < 1s

**Acceptance criteria:**
- Orbit rendering latency: single satellite < 100ms, batch (100 sats) < 1 second
- Pre/post-maneuver trajectories clearly distinguishable via color and opacity
- Cesium.js visualization supports zoom, rotation, time-slider, constellation toggle
- Maneuver confidence displayed prominently (color intensity: red=low, green=high)
- Operators can click maneuver in timeline → detailed prediction visualization
- Historical maneuver timeline interactive (filter by class, confidence threshold)
- Trajectory accuracy: within 1 km altitude for LEO orbits (validated vs. GMAT)
- Export capability: save visualization as image/GeoJSON for reports

---

### Subphase 8: Monitoring & Drift Detection

**Objectives:**
- Monitor model predictions on real-time TLE data
- Detect model drift (confidence degradation, class distribution shift)
- Enable periodic retraining on recent data
- Log predictions for analysis and debugging

**Deliverables:**
- `agent/ml/monitoring.py`
  - Hook into Phase 1 hourly sync
  - Batch inference on all new TLE records
  - Compute prediction statistics: class distribution, confidence histogram, mean confidence
  - Trend analysis: detect confidence drop > 5% over 7 days
  - Class distribution analysis: alert if major shift (e.g., maneuver rate doubles)
- Update `agent/api/main.py`:
  - `/ml/status` endpoint returns: model_version, last_prediction_time, mean_confidence_7d, drift_alert
  - Optional: `/ml/predictions/recent` for debugging
- Maintenance script: `agent/ml/retrain.py`
  - Collect real maneuvers from last 3 months
  - Merge with synthetic data (50% real, 50% synthetic for stability)
  - Retrain from checkpoint (transfer learning)
  - Validate on held-out 1-month test set
  - If metrics improve, promote to production
- Documentation: `docs/drift_detection.md` and `docs/retraining.md`

**Acceptance criteria:**
- Predictions logged with satellite_id, timestamp, predicted_class, confidence, model_version
- Drift alert triggered automatically if mean confidence drops > 5% over 7 days
- System detects significant maneuver class distribution shifts
- Retraining procedure fully automated; runs on monthly schedule (optional)
- No inference interruption during model updates
- Confidence trend visualizations in monitoring dashboard (optional)

---

## Success Criteria for Phase 2 Completion

1. **Data Preparation:**
   - ✅ TLE history extracted: ≥ 6 months, ≥ 30k satellites
   - ✅ Synthetic maneuvers generated: 50k sequences, 10k per class
   - ✅ Feature normalization: zero-mean, unit-variance
   - ✅ Train/val/test split: 70%/15%/15%, stratified by satellite and maneuver type

2. **Model Training:**
   - ✅ JEPA model achieves > 85% accuracy on test set
   - ✅ Precision > 0.80 for 'station_keeping' class
   - ✅ Recall > 0.75 for all propulsive maneuver classes
   - ✅ Outperforms Phase 1 threshold baseline

3. **Visualization:**
   - ✅ Orbit trajectories render < 100ms per satellite
   - ✅ Pre/post-maneuver orbits visually distinguishable
   - ✅ Cesium.js 3D interactive map functional
   - ✅ Maneuver timeline panel shows recent predictions
   - ✅ Operators can drill-down to detailed prediction modal

4. **Integration:**
   - ✅ Inference latency < 50ms per satellite (single), < 500ms per batch (100)
   - ✅ REST API endpoints updated; ML predictions accessible
   - ✅ CoALA tools registered and functional
   - ✅ Confidence filtering works (min_confidence parameter)

5. **Monitoring:**
   - ✅ Predictions logged to database with timestamps and model version
   - ✅ Drift detection active; alerts triggered on >5% confidence drop
   - ✅ Retraining procedure documented and tested
   - ✅ Model versioning and checkpoint management in place

6. **Documentation:**
   - ✅ Complete training curves and evaluation metrics
   - ✅ Maneuver class definitions documented
   - ✅ Integration guide for operators
   - ✅ Drift detection and retraining procedures explained
   - ✅ Visualization user guide (Cesium.js controls, interpretation)

---

## Next Steps: Phase 3 & Beyond

**Phase 3: Autonomous Mission Planning**
- Use JEPA maneuver classifications to predict future orbital changes
- Implement collision avoidance and fuel-optimal scheduling
- Integrate with CoALA for autonomous reboost/deorbit planning

**Phase 4: Multi-Agent Reinforcement Learning** (Your MARL expertise)
- Extend to constellation-level coordination (e.g., STARLINK phasing)
- Train agents on learned JEPA embeddings (low-dimensional state)
- Minimize fuel consumption while maintaining coverage objectives
- Build on autops-rl foundation with JEPA state representations

---

## Implementation Timeline

| Subphase | Duration | Dependencies | Key Output |
|----------|----------|--------------|------------|
| 1. Propagation pipeline | 2 weeks | Phase 1 TLE history | Fused orbital time series |
| 2. Synthetic maneuvers | 3 weeks | Propagator ready | 50k training sequences |
| 3. Data preparation | 2 weeks | Real + synthetic data | Labeled dataset (70/15/15) |
| 4. Model training | 3 weeks | Dataset ready | JEPA checkpoint, metrics |
| 5. Validation | 2 weeks | Trained model | Evaluation report |
| 6. CoALA integration | 2 weeks | Model + API | Functional tools |
| 7. **Visualization** | **2 weeks** | **Subphase 6 complete** | **Interactive 3D orbits** |
| 8. Monitoring | 1 week | All subphases | Drift detection active |
| **Total** | **~17 weeks** | — | **Production-ready Phase 2** |

---

**Ready to proceed with Subphase 1 (propagator integration)?**