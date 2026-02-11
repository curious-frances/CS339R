import numpy as np
def parity(x):
    p = 0
    while(x):
        p ^= (x & 1)
        x >>= 1
    return p

def lfsr_prbs(n_bits, tap_mask, seed=0x1):
    if seed == 0:
        raise ValueError("seed needs to be greater than 0")
    if seed >= (1 << n_bits):
        raise ValueError("seed needs to fit in n_bits")

    state = seed & ((1 << n_bits) - 1)
    out = []
    period = (1 << n_bits) - 1

    for _ in range(period):
        lsb = state & 1
        out.append(lsb)
        state >>= 1
        if lsb:
            state ^= tap_mask
            
    return out
def bits_to_pm1(bits):
     return np.array([1 if b == 0 else -1 for b in bits], dtype=int)

# Hw3 a 
prbs7 = lfsr_prbs(n_bits=3, tap_mask=0x6)
prbs127 = lfsr_prbs(n_bits=7, tap_mask=0x60)
prbs511  = lfsr_prbs(n_bits=9,  tap_mask=0x110)
prbs1023 = lfsr_prbs(n_bits=10, tap_mask=0x240)

# print(f"{len(prbs7)} | {len(prbs127)} | {len(prbs511)} | {len(prbs1023)}")
# print(f"{prbs511} and the number of 1s {sum(prbs511)}")


