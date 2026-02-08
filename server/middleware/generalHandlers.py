import time
from constants import *

def update_long_term_variance(state, engagement_score):
    now = time.time()

    # Only run every 10 minutes
    if now - state.last_variance_update_time < 600:
        return

    state.last_variance_update_time = now

    for proxy, stats in state.session_stats.items():
        fast_std = stats["std"]
        slow_std = state.long_term_variance[proxy]

        # First time: initialize slow variance
        if slow_std is None:
            state.long_term_variance[proxy] = fast_std
            continue

        # Compute improvement
        if fast_std is not None and slow_std is not None:
            improvement = slow_std - fast_std

            # If variance dropped significantly AND engagement is high
            if improvement > VARIANCE_IMPROVEMENT_THRESHOLD and engagement_score > HIGH_ENGAGEMENT_THRESHOLD:
                tighten_threshold(state, proxy)

        # Update slow variance for next cycle
        state.long_term_variance[proxy] = fast_std

    state.save_profile()
