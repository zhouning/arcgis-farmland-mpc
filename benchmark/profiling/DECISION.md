# Decision Gate Result

Source: Colab T4, plain_small_cons seed 0 (800 blocks).
Rule: GPU when throughput >= 5.0x CPU AND SM util mean >= 50.0%.

## MPC
- CPU baseline    : 0.286 steps/s (3.493 s/step)
- GPU T4          : 0.165 steps/s (6.077 s/step)
- Throughput gain : 0.57x  (bar 5.0x) FAIL
- SM util mean    : 0.0%    (bar 50.0%) FAIL
- Decision        : CPU

## PPO
- CPU baseline    : 4.546 steps/s (500 ts smoke)
- GPU T4          : 8.725 steps/s (25000 ts, 2865.3 s)
- Throughput gain : 1.92x  (bar 5.0x) FAIL
- SM util mean    : 1.1%    (bar 50.0%) FAIL
- Decision        : CPU

## CPU reference (cpu_profile.json)
- generator       : 81.6 s
- random          : 0.429 s
- greedy          : 3.5 s
- ga-500gen (est) : 52.7 s