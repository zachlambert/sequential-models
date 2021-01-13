#ifndef CPP_ML_HMM_SAMPLER_H
#define CPP_ML_HMM_SAMPLER_H

#include <random>
#include <Eigen/Dense>

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


class HmmMultivariateSampler {
public:
    typedef Eigen::VectorXd x_t;
    typedef Eigen::VectorXd y_t;

    HmmMultivariateSampler();
    void initialise();
    void step();

    int n;
    x_t xn;
    y_t yn;

private:
    virtual x_t sample_p0() = 0;
    virtual x_t sample_fn(const x_t &xn_prev, int n) = 0;
    virtual y_t sample_gn(const x_t &xn, int n) = 0;
};

#endif
