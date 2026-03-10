# Decisions: Motion-Driven Mask Sampling

## Architectural Choices

### Flow Model Selection
- **Decision**: Use GMFlow (already in codebase at `gmflow/`)
- **Rationale**: Production-validated in `demo_flow_guided_mpc.py`, checkpoint available
- **Alternative Rejected**: RAFT (would require new dependency)

### Threshold Strategy
- **Decision**: Percentile-based (70th percentile) with minimum 0.5px
- **Rationale**: Adaptive to scene dynamics, from `demo_flow_guided_mpc.py:312`
- **Alternative Rejected**: Fixed pixel threshold (fails across diverse scenes)

### Sampling Strategy
- **Decision**: 70% motion regions + 30% Shi-Tomasi corners
- **Rationale**: Balance coverage with texture quality for robust tracking

