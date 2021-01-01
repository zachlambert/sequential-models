#ifndef CPP_ML_HMM_H
#define CPP_ML_HMM_H

#include <chrono>
#define _USE_MATH_DEFINES
#include <cmath>
#include <random>


class Hmm {
public:
    virtual double p0(double x)=0;
    virtual double fn(double x_prev, double x, int n)=0;
    virtual double gn(double x, double y, int n)=0;
};


class HmmSampler {
public:
    HmmSampler();
    void initialise();
    void step();

    int n;
    double xn;
    double yn;

private:
    virtual double sample_p0() = 0;
    virtual double sample_fn(double xn_prev, int n) = 0;
    virtual double sample_gn(double xn, int n) = 0;
};


class HmmGaussian: public Hmm {
public:
    HmmGaussian(double a, double b, double c, double d);

private:
    double p0(double x);
    double fn(double x_prev, double x, int n);
    double gn(double x, double y, int n);

    double a, b, c, d;
};


class HmmSamplerGaussian: public HmmSampler {
public:
    HmmSamplerGaussian(double a, double b, double c, double d);

private:
    double sample_p0();
    double sample_fn(double xn_prev, int n);
    double sample_gn(double xn, int n);

    const double a, b, c, d;
    std::default_random_engine generator;
    std::normal_distribution<double> distribution;
};

#endif
