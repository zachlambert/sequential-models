#ifndef CPP_ML_HMM_MULTIVARIATE_H
#define CPP_ML_HMM_MULTIVARIATE_H

#include <random>
#include <Eigen/Dense>

class HmmMultivariate {
public:
    typedef Eigen::VectorXd x_t;
    typedef Eigen::VectorXd y_t;

    virtual double p0(const x_t &x)=0;
    virtual double fn(const x_t &x_prev, const x_t &x, int n)=0;
    virtual double gn(const x_t &x, const y_t &y, int n)=0;
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


class HmmMultivariateStateSpace: public HmmMultivariate {
public:
    HmmMultivariateStateSpace(Eigen::MatrixXd F,

private:
    double p0(x_t x);
    double fn(x_t x_prev, x_t x, int n);
    double gn(y_t x, y_t y, int n);

    Eigen::VectorXd mean;
    Eigen::MatrixXd covariance;
};


class HmmMultivariateSamplerStateSpace: public HmmMultivariateSampler {
public:
    HmmMultivariateSamplerStateSpace(
        Eigen::MatrixXd F, Eigen::MatrixXd Q,
        Eigen::MatrixXd G, Eigen::MatrixXd R);

private:
    x_t sample_p0();
    x_t sample_fn(x_t xn_prev, int n);
    y_t sample_gn(x_t xn, int n);

    Eigen::MatrixXd F, G, Q, R;
    Eigen::MatrixXd Q_S, R_S;

    std::default_random_engine generator;
    std::normal_distribution<double> distribution;

    // To draw from a N-dimensional gaussian:
    // Draw N iid samples U from standard normal N(0, 1)
    // Transform with X = mean + S*U
    // X ~ N(mean, covariance)
    // Where covariance = SS^T

    // State transition and emission equations have covariance Q and R
    // with S matrices Q_S and R_S
};

#endif
