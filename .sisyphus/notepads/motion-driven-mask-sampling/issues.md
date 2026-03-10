# Issues & Gotchas: Motion-Driven Mask Sampling

## Known Issues

### GMFlow Device Handling
- **Issue**: GMFlow model stays in GPU memory after use
- **Solution**: Explicit `del flownet` + `torch.cuda.empty_cache()` required
- **Source**: Metis gap analysis

### Coverage Edge Cases
- **Issue**: Camera motion → coverage >80% → mask useless
- **Solution**: Fallback to Shi-Tomasi sampling
- **Issue**: Static scene → coverage <1% → no points
- **Solution**: Fallback to uniform grid

