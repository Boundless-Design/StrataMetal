// QSA gate. One thread per output element.
// q_full is (n_head, 2 * head_dim). The gate is the second head_dim of each head.
// out = attn * sigmoid(gate). This GPU has no fp64, so the sigmoid replays host libm
// expf with integer arithmetic. The q8 codes then match the host reference.
#include <metal_stdlib>
using namespace metal;

constant ulong kExpA = 0x40671547652b82feul;
constant ulong kExpB = 0x4338000000000000ul;
constant ulong kExpC0 = 0x3eeebfbdff30d656ul;
constant ulong kExpC1 = 0x3f762e4453e10daeul;
constant ulong kExpTab[128] = {
    0x3ff0000000000000ul,
    0x3feff63da9fb3335ul,
    0x3fefec9a3e778061ul,
    0x3fefe315e86e7f85ul,
    0x3fefd9b0d3158574ul,
    0x3fefd06b29ddf6deul,
    0x3fefc74518759bc8ul,
    0x3fefbe3ecac6f383ul,
    0x3fefb5586cf9890ful,
    0x3fefac922b7247f7ul,
    0x3fefa3ec32d3d1a2ul,
    0x3fef9b66affed31bul,
    0x3fef9301d0125b51ul,
    0x3fef8abdc06c31ccul,
    0x3fef829aaea92de0ul,
    0x3fef7a98c8a58e51ul,
    0x3fef72b83c7d517bul,
    0x3fef6af9388c8deaul,
    0x3fef635beb6fcb75ul,
    0x3fef5be084045cd4ul,
    0x3fef54873168b9aaul,
    0x3fef4d5022fcd91dul,
    0x3fef463b88628cd6ul,
    0x3fef3f49917ddc96ul,
    0x3fef387a6e756238ul,
    0x3fef31ce4fb2a63ful,
    0x3fef2b4565e27cddul,
    0x3fef24dfe1f56381ul,
    0x3fef1e9df51fdee1ul,
    0x3fef187fd0dad990ul,
    0x3fef1285a6e4030bul,
    0x3fef0cafa93e2f56ul,
    0x3fef06fe0a31b715ul,
    0x3fef0170fc4cd831ul,
    0x3feefc08b26416fful,
    0x3feef6c55f929ff1ul,
    0x3feef1a7373aa9cbul,
    0x3feeecae6d05d866ul,
    0x3feee7db34e59ff7ul,
    0x3feee32dc313a8e5ul,
    0x3feedea64c123422ul,
    0x3feeda4504ac801cul,
    0x3feed60a21f72e2aul,
    0x3feed1f5d950a897ul,
    0x3feece086061892dul,
    0x3feeca41ed1d0057ul,
    0x3feec6a2b5c13cd0ul,
    0x3feec32af0d7d3deul,
    0x3feebfdad5362a27ul,
    0x3feebcb299fddd0dul,
    0x3feeb9b2769d2ca7ul,
    0x3feeb6daa2cf6642ul,
    0x3feeb42b569d4f82ul,
    0x3feeb1a4ca5d920ful,
    0x3feeaf4736b527daul,
    0x3feead12d497c7fdul,
    0x3feeab07dd485429ul,
    0x3feea9268a5946b7ul,
    0x3feea76f15ad2148ul,
    0x3feea5e1b976dc09ul,
    0x3feea47eb03a5585ul,
    0x3feea34634ccc320ul,
    0x3feea23882552225ul,
    0x3feea155d44ca973ul,
    0x3feea09e667f3bcdul,
    0x3feea012750bdabful,
    0x3fee9fb23c651a2ful,
    0x3fee9f7df9519484ul,
    0x3fee9f75e8ec5f74ul,
    0x3fee9f9a48a58174ul,
    0x3fee9feb564267c9ul,
    0x3feea0694fde5d3ful,
    0x3feea11473eb0187ul,
    0x3feea1ed0130c132ul,
    0x3feea2f336cf4e62ul,
    0x3feea427543e1a12ul,
    0x3feea589994cce13ul,
    0x3feea71a4623c7adul,
    0x3feea8d99b4492edul,
    0x3feeaac7d98a6699ul,
    0x3feeace5422aa0dbul,
    0x3feeaf3216b5448cul,
    0x3feeb1ae99157736ul,
    0x3feeb45b0b91ffc6ul,
    0x3feeb737b0cdc5e5ul,
    0x3feeba44cbc8520ful,
    0x3feebd829fde4e50ul,
    0x3feec0f170ca07baul,
    0x3feec49182a3f090ul,
    0x3feec86319e32323ul,
    0x3feecc667b5de565ul,
    0x3feed09bec4a2d33ul,
    0x3feed503b23e255dul,
    0x3feed99e1330b358ul,
    0x3feede6b5579fdbful,
    0x3feee36bbfd3f37aul,
    0x3feee89f995ad3adul,
    0x3feeee07298db666ul,
    0x3feef3a2b84f15fbul,
    0x3feef9728de5593aul,
    0x3feeff76f2fb5e47ul,
    0x3fef05b030a1064aul,
    0x3fef0c1e904bc1d2ul,
    0x3fef12c25bd71e09ul,
    0x3fef199bdd85529cul,
    0x3fef20ab5fffd07aul,
    0x3fef27f12e57d14bul,
    0x3fef2f6d9406e7b5ul,
    0x3fef3720dcef9069ul,
    0x3fef3f0b555dc3faul,
    0x3fef472d4a07897cul,
    0x3fef4f87080d89f2ul,
    0x3fef5818dcfba487ul,
    0x3fef60e316c98398ul,
    0x3fef69e603db3285ul,
    0x3fef7321f301b460ul,
    0x3fef7c97337b9b5ful,
    0x3fef864614f5a129ul,
    0x3fef902ee78b3ff6ul,
    0x3fef9a51fbc74c83ul,
    0x3fefa4afa2a490daul,
    0x3fefaf482d8e67f1ul,
    0x3fefba1bee615a27ul,
    0x3fefc52b376bba97ul,
    0x3fefd0765b6e4540ul,
    0x3fefdbfdad9cbe14ul,
    0x3fefe7c1819e90d8ul,
    0x3feff3c22b8f71f1ul,
};

// Six limbs. Aligning B = 2^52 with the smallest float (2^-149) puts B's
// leading bit at index 298. Four limbs stop at 255 and drop that addend.
constant int kLimbs = 6;
constant int kBits = 384;

struct W {
    ulong l[6];
    int sticky;
};

struct Num {
    int sign;
    int exp;
    ulong sig;
    int kind;
};

void w_clear(thread W& a) {
    for (int i = 0; i < kLimbs; ++i) a.l[i] = 0;
    a.sticky = 0;
}

int w_is_zero(thread const W& a) {
    for (int i = 0; i < kLimbs; ++i)
        if (a.l[i]) return 0;
    return 1;
}

int w_msb(thread const W& a) {
    for (int i = kLimbs - 1; i >= 0; --i) {
        if (a.l[i]) return i * 64 + 63 - (int) clz(a.l[i]);
    }
    return -1;
}

// noinline is load-bearing. Inlining these into soft_fma makes the Metal
// compiler drop a limb during a subtract and round_pack then returns zero.
__attribute__((noinline))
void w_shr(thread W& a, int s) {
    if (s <= 0) return;
    if (s >= kBits) {
        a.sticky |= !w_is_zero(a);
        w_clear(a);
        return;
    }
    int limb = s >> 6;
    int bit = s & 63;
    ulong lost = 0;
    for (int i = 0; i < limb; ++i) lost |= a.l[i];
    if (bit && limb < kLimbs) lost |= a.l[limb] & ((1ul << bit) - 1ul);
    W r;
    w_clear(r);
    for (int i = 0; i < kLimbs - limb; ++i) {
        ulong v = a.l[i + limb] >> bit;
        if (bit && i + limb + 1 < kLimbs) v |= a.l[i + limb + 1] << (64 - bit);
        r.l[i] = v;
    }
    r.sticky = a.sticky | (lost != 0);
    a = r;
}

__attribute__((noinline))
void w_shl(thread W& a, int s) {
    if (s <= 0) return;
    if (s >= kBits) {
        w_clear(a);
        return;
    }
    int limb = s >> 6;
    int bit = s & 63;
    W r;
    w_clear(r);
    r.sticky = a.sticky;
    for (int i = kLimbs - 1; i >= limb; --i) {
        ulong v = a.l[i - limb] << bit;
        if (bit && i - limb - 1 >= 0) v |= a.l[i - limb - 1] >> (64 - bit);
        r.l[i] = v;
    }
    a = r;
}

__attribute__((noinline))
int w_add(thread W& a, thread const W& b) {
    ulong carry = 0;
    for (int i = 0; i < kLimbs; ++i) {
        ulong s = a.l[i] + b.l[i];
        ulong c1 = s < a.l[i] ? 1ul : 0ul;
        ulong s2 = s + carry;
        ulong c2 = s2 < s ? 1ul : 0ul;
        a.l[i] = s2;
        carry = c1 + c2;
    }
    return (int) carry;
}

__attribute__((noinline))
void w_sub(thread W& a, thread const W& b) {
    ulong br = 0;
    for (int i = 0; i < kLimbs; ++i) {
        ulong bv = b.l[i] + br;
        ulong br1 = bv < b.l[i] ? 1ul : 0ul;
        if (a.l[i] < bv) {
            a.l[i] = a.l[i] - bv;
            br = br1 + 1ul;
        } else {
            a.l[i] = a.l[i] - bv;
            br = br1;
        }
    }
}

__attribute__((noinline))
int w_cmp(thread const W& a, thread const W& b) {
    for (int i = kLimbs - 1; i >= 0; --i) {
        if (a.l[i] < b.l[i]) return -1;
        if (a.l[i] > b.l[i]) return 1;
    }
    return 0;
}

Num unpack64(ulong u) {
    Num n;
    n.sign = (int) (u >> 63);
    int e = (int) ((u >> 52) & 0x7ff);
    ulong m = u & ((1ul << 52) - 1ul);
    if (e == 0) {
        n.kind = 0;
        n.sig = 0;
        n.exp = 0;
        return n;
    }
    if (e == 0x7ff) {
        n.kind = m ? 2 : 1;
        n.sig = 0;
        n.exp = 0;
        return n;
    }
    n.kind = 0;
    n.sig = m | (1ul << 52);
    n.exp = e - 1023;
    return n;
}

ulong pack64(int sign, int exp, ulong sig53) {
    int be = exp + 1023;
    if (be >= 0x7ff) return ((ulong) sign << 63) | (0x7fful << 52);
    if (be <= 0) return (ulong) sign << 63;
    return ((ulong) sign << 63) | ((ulong) be << 52) | (sig53 & ((1ul << 52) - 1ul));
}

__attribute__((noinline))
ulong round_pack(int sign, thread const W& mag_in, int lsb_exp) {
    W mag = mag_in;
    if (w_is_zero(mag)) return (ulong) sign << 63;
    int msb = w_msb(mag);
    int sh = msb - 52;
    if (sh > 0) {
        int guard = 0;
        int sticky = mag.sticky;
        int rb = sh - 1;
        int limb = rb >> 6;
        int bit = rb & 63;
        if (limb < kLimbs) guard = (int) ((mag.l[limb] >> bit) & 1ul);
        if (bit && limb < kLimbs && (mag.l[limb] & ((1ul << bit) - 1ul))) sticky = 1;
        for (int i = 0; i < limb; ++i)
            if (mag.l[i]) sticky = 1;
        w_shr(mag, sh);
        ulong sig = mag.l[0];
        if (guard && (sticky || (sig & 1ul))) {
            sig += 1;
            if (sig >> 53) {
                sig >>= 1;
                msb += 1;
            }
        }
        return pack64(sign, lsb_exp + msb, sig);
    }
    w_shl(mag, -sh);
    return pack64(sign, lsb_exp + msb, mag.l[0]);
}

__attribute__((noinline))
void exact_mul(thread const Num& a, thread const Num& b, thread W& mag, thread int& lsb_exp, thread int& sign) {
    w_clear(mag);
    sign = a.sign ^ b.sign;
    ulong a0 = a.sig & 0xfffffffful, a1 = a.sig >> 32;
    ulong b0 = b.sig & 0xfffffffful, b1 = b.sig >> 32;
    ulong p0 = a0 * b0, p1 = a0 * b1, p2 = a1 * b0, p3 = a1 * b1;
    ulong mid = (p0 >> 32) + (p1 & 0xfffffffful) + (p2 & 0xfffffffful);
    mag.l[0] = (p0 & 0xfffffffful) | (mid << 32);
    mag.l[1] = p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32);
    lsb_exp = (a.exp - 52) + (b.exp - 52);
}

__attribute__((noinline))
ulong soft_fma(ulong ua, ulong ub, ulong uc) {
    Num a = unpack64(ua);
    Num b = unpack64(ub);
    Num c = unpack64(uc);
    if (a.kind == 2 || b.kind == 2 || c.kind == 2) return 0x7ff8000000000000ul;
    if (a.kind || b.kind || c.kind) return 0x7ff8000000000000ul;
    if (a.sig == 0 || b.sig == 0) {
        if (c.sig == 0) {
            int sign = a.sign ^ b.sign;
            if (c.sign == sign) return (ulong) sign << 63;
            return 0;
        }
        return uc;
    }
    W prod;
    int pexp;
    int psign;
    exact_mul(a, b, prod, pexp, psign);
    W sum = prod;
    int sexp = pexp;
    int ssign = psign;
    if (c.sig != 0) {
        W cm;
        w_clear(cm);
        cm.l[0] = c.sig;
        int cexp = c.exp - 52;
        if (cexp >= pexp) {
            int d = cexp - pexp;
            if (d >= kBits) {
                sum = cm;
                sexp = cexp;
                ssign = c.sign;
                sum.sticky = 0;
            } else {
                w_shl(cm, d);
                sexp = pexp;
                if (psign == c.sign) {
                    int carry = w_add(sum, cm);
                    if (carry) {
                        w_shr(sum, 1);
                        sexp += 1;
                    }
                } else {
                    int cmp = w_cmp(sum, cm);
                    if (cmp == 0) return 0;
                    if (cmp < 0) {
                        W tmp = cm;
                        w_sub(tmp, sum);
                        tmp.sticky = cm.sticky | sum.sticky;
                        sum = tmp;
                        ssign = c.sign;
                    } else {
                        w_sub(sum, cm);
                        sum.sticky |= cm.sticky;
                    }
                }
            }
        } else {
            int d = pexp - cexp;
            if (d >= kBits) {
                sum.sticky |= 1;
            } else {
                w_shr(cm, d);
                if (psign == c.sign) {
                    int carry = w_add(sum, cm);
                    if (carry) {
                        w_shr(sum, 1);
                        sexp += 1;
                    }
                } else {
                    int cmp = w_cmp(sum, cm);
                    if (cmp == 0 && !sum.sticky && !cm.sticky) return 0;
                    if (cmp < 0 || (cmp == 0 && cm.sticky && !sum.sticky)) {
                        W tmp = cm;
                        w_sub(tmp, sum);
                        tmp.sticky = cm.sticky | sum.sticky;
                        sum = tmp;
                        ssign = c.sign;
                    } else {
                        w_sub(sum, cm);
                        sum.sticky |= cm.sticky;
                    }
                }
            }
        }
    }
    return round_pack(ssign, sum, sexp);
}

ulong f32_to_f64(float x) {
    uint u = as_type<uint>(x);
    uint sign = u >> 31;
    uint exp = (u >> 23) & 0xff;
    uint mant = u & 0x7fffffu;
    if (exp == 255) return ((ulong) sign << 63) | (0x7fful << 52) | ((ulong) mant << 29);
    if (exp == 0) {
        if (mant == 0) return (ulong) sign << 63;
        int cl = 0;
        uint m = mant;
        while ((m & 0x800000u) == 0) {
            m <<= 1;
            ++cl;
        }
        int ub = -126 - cl;
        int be = ub + 1023;
        return ((ulong) sign << 63) | ((ulong) be << 52) | ((ulong) (m & 0x7fffffu) << 29);
    }
    int be = (int) exp - 127 + 1023;
    return ((ulong) sign << 63) | ((ulong) be << 52) | ((ulong) mant << 29);
}

float f64_to_f32(ulong u) {
    uint sign = (uint) (u >> 63);
    int exp = (int) ((u >> 52) & 0x7ff);
    ulong mant = u & ((1ul << 52) - 1ul);
    uint out;
    if (exp == 0x7ff) {
        out = (sign << 31) | (0xffu << 23) | (mant ? 0x400000u : 0u);
    } else if (exp == 0) {
        out = sign << 31;
    } else {
        int ub = exp - 1023;
        if (ub > 127) {
            out = (sign << 31) | (0xffu << 23);
        } else if (ub >= -126) {
            uint top = (uint) (mant >> 29);
            uint roundb = (uint) ((mant >> 28) & 1ul);
            uint sticky = (mant & ((1ul << 28) - 1ul)) != 0;
            if (roundb && (sticky || (top & 1u))) {
                top += 1;
                if (top == (1u << 23)) {
                    top = 0;
                    ub += 1;
                }
            }
            if (ub > 127) out = (sign << 31) | (0xffu << 23);
            else out = (sign << 31) | ((uint) (ub + 127) << 23) | top;
        } else {
            ulong sig = mant | (1ul << 52);
            int R = -97 - ub;
            if (R >= 64) {
                out = sign << 31;
            } else {
                uint keep = (uint) (sig >> R);
                int guard = (int) ((sig >> (R - 1)) & 1ul);
                int sticky = (sig & ((1ul << (R - 1)) - 1ul)) != 0;
                if (guard && (sticky || (keep & 1u))) keep += 1;
                if (keep >= 0x800000u) out = (sign << 31) | (1u << 23);
                else out = (sign << 31) | keep;
            }
        }
    }
    return as_type<float>(out);
}

// Host libm expf. The GPU has no fp64, so this is the same sequence libm runs,
// on integer limbs: scale by 128/ln(2), a 128-entry table, a degree-2 polynomial.
float apple_expf(float x) {
    uint xi = as_type<uint>(x);
    if (((xi >> 23) & 0xff) >= 0x86) {
        if (!(x < 0.0f)) return x * as_type<float>(0x7f7fffffu);
        return 0.0f;
    }
    ulong xd = f32_to_f64(x);
    ulong z = soft_fma(kExpA, xd, kExpB);
    ulong kd = soft_fma(z, 0x3ff0000000000000ul, kExpB ^ (1ul << 63));
    ulong r = soft_fma(kExpA, xd, kd ^ (1ul << 63));
    ulong tbits = kExpTab[z & 127ul] + (z << 45);
    ulong p = soft_fma(kExpC0, r, kExpC1);
    ulong pr = soft_fma(p, r, 0);
    ulong y = soft_fma(pr, tbits, tbits);
    return f64_to_f32(y);
}

struct QgP {
    int n_head;
    int head_dim;
    int pad0;
    int pad1;
};

kernel void qsa_gate_apply_f32(device const float* attn [[buffer(0)]],
                               device const float* q_full [[buffer(1)]],
                               device float* out [[buffer(2)]],
                               constant QgP& dim [[buffer(3)]],
                               uint i [[thread_position_in_grid]]) {
    const int n = dim.n_head * dim.head_dim;
    if ((int) i >= n) return;
    const int head_dim = dim.head_dim;
    const int h = (int) i / head_dim;
    const int d = (int) i - h * head_dim;
    const float g = q_full[(size_t) h * (size_t) (head_dim * 2) + (size_t) head_dim + (size_t) d];
    const float e = apple_expf(-g);
    const float s = 1.0f / (1.0f + e);
    out[i] = attn[i] * s;
}
