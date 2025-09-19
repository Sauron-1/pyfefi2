#include <cstdint>

#pragma once

namespace rational {

constexpr uint64_t gcd(uint64_t a, uint64_t b) {
    if (a == 0)
        return b;
    else if (b == 0)
        return a;
    else if (a == b)
        return a;
    else if (a > b)
        return gcd(a % b, b);
    else
        return gcd(b % a, a);
}

struct Rational {

    uint64_t numerator, denominator;
    bool sign;

    constexpr Rational(bool sign, uint64_t numerator, uint64_t denominator):
        numerator(numerator), denominator(denominator), sign(sign) {}

    constexpr Rational(int64_t numerator, uint64_t denominator):
        Rational(numerator>0?0:1, numerator>0?numerator:-numerator, denominator) {}

    constexpr Rational(bool sign, uint64_t numerator):
        Rational(sign, numerator, 1) {}

    constexpr Rational(int64_t numerator):
        Rational(numerator>0?0:1, numerator>0?numerator:-numerator, 1) {}

    constexpr Rational() : Rational(0, 0, 1) {}

    constexpr Rational reducted() const {
        uint64_t _gcd = gcd(numerator, denominator);
        return Rational(sign, numerator/_gcd, denominator/_gcd);
    }

    template<typename Float=double>
    constexpr Float to_float() const {
        Float result = Float((double)numerator / (double)denominator);
        if (sign and numerator != 0) result = -result;
        return result;
    }

    constexpr Rational operator-() const {
        return Rational(not sign, numerator, denominator);
    }

};

constexpr Rational operator+(const Rational& r1, const Rational& r2) {
    uint64_t a1 = r1.numerator*r2.denominator,
             a2 = r2.numerator*r1.denominator;
    uint64_t b = r1.denominator * r2.denominator;
    if (r1.sign == r2.sign)
        return Rational(r1.sign, a1+a2, b).reducted();
    else {
        if (r1.sign)
            return r2 + r1;
        else {
            if (a1 > a2)
                return Rational(0, a1-a2, b).reducted();
            else
                return Rational(1, a2-a1, b).reducted();
        }
    }
}

constexpr Rational operator-(const Rational& r1, const Rational& r2) {
    return r1 + (-r2);
}

constexpr Rational operator*(const Rational& r1, const Rational& r2) {
    uint64_t a = r1.numerator*r2.numerator;
    uint64_t b = r1.denominator*r2.denominator;
    return Rational(r1.sign xor r2.sign, a, b).reducted();
}

constexpr Rational operator/(const Rational& r1, const Rational& r2) {
    uint64_t a = r1.numerator*r2.denominator;
    uint64_t b = r1.denominator*r2.numerator;
    return Rational(r1.sign xor r2.sign, a, b).reducted();
}

template<typename T>
inline auto constexpr powi(T val, std::size_t N) {
    if (N == 0)
        return T{1};
    else if (N == 1)
        return val;
    else {
        auto half = powi(val, N/2);
        if (N % 2 == 0)
            return half * half;
        else
            return half * half * val;
    }
}

constexpr Rational pow(const Rational& r, std::size_t N) {
    auto new_sign = N % 2 == 0 ? r.sign xor r.sign : r.sign;
    return Rational(
            new_sign,
            powi(r.numerator, N),
            powi(r.denominator, N));
}

}
