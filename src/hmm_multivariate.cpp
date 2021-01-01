#include "hmm_multivariate.h"

typedef HmmMultivariateSamplerStateSpace::x_t x_t;
typedef HmmMultivariateSamplerStateSpace::y_t y_t;

Eigen::MatrixXd get_S_matrix(const Eigen::MatrixXd &cov)
{
    Eigen::EigenSolver<Eigen::MatrixXd> eig(cov);
    std::size_t n = eig.eigenvalues().size();
    Eigen::MatrixXd sqrt_eigenvalues(n, n);
    for (std::size_t i = 0; i < n; i++) {
        sqrt_eigenvalues(i, i) = sqrt(eig.eigenvalues()(i).real());
    }
    return sqrt_eigenvalues * eig.eigenvectors();
}

HmmMultivariateSamplerStateSpace::HmmMultivariateSamplerStateSpace(
        Eigen::MatrixXd F, Eigen::MatrixXd Q,
        Eigen::MatrixXd G, Eigen::MatrixXd R):
    F(F), Q(Q), G(G), R(R), Q_S(get_S_matrix(Q)), R_S(get_S_matrix(R))
{

}

x_t sample_p0()
{

}

x_t sample_fn(x_t xn_prev, int n)
{

}

y_t sample_gn(x_t xn, int n)
{

}
