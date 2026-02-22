from operator import xor
import numpy as np
def lfsr_prbs(n_bits, tap_mask, seed=None):
    if seed is None:
        seed = (1 << n_bits) - 1 # Initial state of the shift register (all bits set to 1)
    state = seed & ((1 << n_bits) - 1)
    period = (1 << n_bits) - 1
    out = []

    for _ in range(period):
        msb = (state >> (n_bits - 1)) & 1
        out.append(msb)

        
        temp = state & tap_mask
        feedback = temp & 1
        temp >>= 1
        while temp:
            feedback ^= (temp & 1)
            temp >>= 1

        state = ((state << 1) & ((1 << n_bits) - 1)) | feedback

    return out
def bits_to_pm1(bits):
     return np.array([1 if b == 0 else -1 for b in bits])

# Hw3 a
prbs7 = lfsr_prbs(n_bits=3, tap_mask=0x6)
prbs127 = lfsr_prbs(n_bits=7, tap_mask=0x60)
prbs511  = lfsr_prbs(n_bits=9,  tap_mask=0x110)
prbs1023 = lfsr_prbs(n_bits=10, tap_mask=0x240)

pbtest = lfsr_prbs(n_bits=4, tap_mask=0xC)
print("pbtest:", pbtest)
# print(f"{len(prbs7)} | {len(prbs127)} | {len(prbs511)} | {len(prbs1023)}")
# print(f"{prbs511} and the number of 1s {sum(prbs511)}")


